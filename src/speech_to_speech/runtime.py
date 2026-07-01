from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from anytrain.codec import LongCatAudioCodec
from anytrain.idspace import IdSpace, Modality, ModalityBlock
from anytrain.tokenizer import CodecBPE
from .bpe_types import BPEArtifactMeta
from .config import BPEConfig, ModelConfig
from .datamodule.types import SpeechPair, TranslationExample
from .types import AudioBoundary, SpecialToken

if TYPE_CHECKING:
    from anytrain.codec import LongCatDecoderName
    from transformers import PreTrainedTokenizerBase
    from torch import Tensor

_QWEN3_TOKENIZERS: dict[tuple[str, bool], PreTrainedTokenizerBase] = {}
_LONGCAT_TOKENIZERS: dict[Path, CodecBPE] = {}
_LONGCAT_CODEC: LongCatAudioCodec | None = None


def qwen3_tokenizer(
    config: ModelConfig | None = None,
    *,
    model_name_or_path: str | None = None,
    trust_remote_code: bool | None = None,
) -> PreTrainedTokenizerBase:
    config = config or ModelConfig()
    name = model_name_or_path or config.backbone.model_name_or_path
    trust = config.backbone.trust_remote_code if trust_remote_code is None else trust_remote_code
    key = (name, trust)
    if key not in _QWEN3_TOKENIZERS:
        from transformers import AutoTokenizer

        _QWEN3_TOKENIZERS[key] = AutoTokenizer.from_pretrained(
            name, trust_remote_code=trust
        )
    return _QWEN3_TOKENIZERS[key]


def qwen3_longcat_idspace(
    *,
    tokenizer: object | None = None,
    bpe_vocab_size: int,
    config: ModelConfig | None = None,
    qwen_vocab_size: int | None = None,
) -> IdSpace:
    tokenizer = tokenizer or qwen3_tokenizer(config)
    text_vocab_size = _qwen_vocab_size(tokenizer, explicit=qwen_vocab_size)
    audio_start = text_vocab_size + len(AudioBoundary)
    return IdSpace(
        special_token_ids={
            **_qwen3_special_token_ids(tokenizer),
            AudioBoundary.BOA.value: text_vocab_size,
            AudioBoundary.EOA.value: text_vocab_size + 1,
        },
        modality_blocks=(
            ModalityBlock(
                modality=Modality.TEXT,
                start=0,
                vocab_size=text_vocab_size,
            ),
            ModalityBlock(
                modality=Modality.AUDIO,
                start=audio_start,
                vocab_size=bpe_vocab_size,
            ),
        ),
    )


def longcat_bpe_path(
    config: BPEConfig | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    config = config or BPEConfig()
    root = _cache_dir(config, cache_dir)
    return config.artifact_path(root).expanduser()


def longcat_tokenizer(
    config: BPEConfig | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> CodecBPE:
    config = config or BPEConfig()
    path = longcat_bpe_path(config, cache_dir=cache_dir)
    if path not in _LONGCAT_TOKENIZERS:
        _validate_cached_bpe(path, config)
        _LONGCAT_TOKENIZERS[path] = CodecBPE.from_pretrained(path)
    return _LONGCAT_TOKENIZERS[path]


def prepare_longcat_tokenizer(
    pairs: Iterable[SpeechPair | TranslationExample]
    | Callable[[], Iterable[SpeechPair | TranslationExample]],
    *,
    datasets: Iterable[Mapping[str, object]] = (),
    config: BPEConfig | None = None,
    cache_dir: str | Path | None = None,
) -> CodecBPE:
    config = config or BPEConfig()
    path = longcat_bpe_path(config, cache_dir=cache_dir)
    datasets = tuple(datasets)
    if _bpe_state_path(path).exists():
        if datasets:
            _validate_cached_bpe(path, config, datasets=datasets)
        return longcat_tokenizer(config, cache_dir=cache_dir)

    bpe = CodecBPE.train(
        _pair_corpus_factory(pairs),
        codebook_sizes=config.codebook_sizes,
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        max_token_length=config.max_token_length,
    )
    path.mkdir(parents=True, exist_ok=True)
    bpe.save_pretrained(path)
    _write_bpe_meta(
        path,
        BPEArtifactMeta(
            codec_name=config.codec_name,
            requested_vocab_size=config.vocab_size,
            actual_vocab_size=bpe.vocab_size,
            min_frequency=config.min_frequency,
            max_token_length=config.max_token_length,
            codebook_sizes=config.codebook_sizes,
            datasets=datasets,
        ),
    )
    _LONGCAT_TOKENIZERS[path] = bpe
    return bpe


def longcat_codec() -> LongCatAudioCodec:
    global _LONGCAT_CODEC
    if _LONGCAT_CODEC is None:
        _LONGCAT_CODEC = LongCatAudioCodec.from_pretrained()
    return _LONGCAT_CODEC


def longcat_acoustic_features(
    acoustic_codes: Tensor,
    *,
    codec: LongCatAudioCodec | None = None,
    decoder: LongCatDecoderName = "16k_4codebooks",
) -> Tensor:
    acoustic_codes = _batched_acoustic_codes(acoustic_codes)
    return (codec or longcat_codec()).acoustic_codes_to_features(
        acoustic_codes,
        decoder=decoder,
    )


def _cache_dir(config: BPEConfig, cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    value = os.environ.get(config.cache_dir_env)
    if value is None:
        raise KeyError(
            f"{config.cache_dir_env} is required to locate LongCat BPE artifacts."
        )
    return Path(value)


def _qwen3_special_token_ids(tokenizer: object) -> dict[str, int]:
    return {member.value: _token_id(tokenizer, member.value) for member in SpecialToken}


def _qwen_vocab_size(tokenizer: object, *, explicit: int | None) -> int:
    if explicit is not None:
        _validate_positive_int(explicit, name="qwen_vocab_size")
        return explicit

    value = getattr(tokenizer, "vocab_size", None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value

    length = getattr(tokenizer, "__len__", None)
    if callable(length):
        value = length()
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value

    raise AttributeError("qwen3 tokenizer must expose a positive vocab_size or __len__.")


def _token_id(tokenizer: object, token: str) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert):
        token_id = convert(token)
        if isinstance(token_id, int) and token_id >= 0:
            return token_id

    encode = getattr(tokenizer, "encode")
    ids = encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"special token {token!r} must map to exactly one token id.")
    return int(ids[0])


def _validate_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _batched_acoustic_codes(acoustic_codes: Tensor) -> Tensor:
    if acoustic_codes.dim() == 2:
        acoustic_codes = acoustic_codes.unsqueeze(0)
    if acoustic_codes.dim() != 3:
        raise ValueError("LongCat acoustic_codes must have shape [nq, time] or [batch, nq, time].")
    if (
        acoustic_codes.dtype == torch.bool
        or torch.is_floating_point(acoustic_codes)
        or torch.is_complex(acoustic_codes)
    ):
        raise TypeError("LongCat acoustic_codes must contain integer ids.")
    return acoustic_codes


def _pair_corpus(pairs: Iterable[SpeechPair | TranslationExample]) -> Iterable[list[list[int]]]:
    for pair in pairs:
        source = _unit_sequence(pair.source_ids)
        if source:
            yield source
        target = _unit_sequence(pair.target_ids)
        if target:
            yield target


def _pair_corpus_factory(
    pairs: Iterable[SpeechPair | TranslationExample]
    | Callable[[], Iterable[SpeechPair | TranslationExample]],
) -> Callable[[], Iterable[list[list[int]]]]:
    if callable(pairs):
        return partial(_pair_corpus_from_factory, pairs)
    if isinstance(pairs, Iterator):
        raise TypeError("pairs must be re-iterable or a callable returning a fresh iterator.")
    return partial(_pair_corpus, pairs)


def _pair_corpus_from_factory(
    pairs: Callable[[], Iterable[SpeechPair | TranslationExample]],
) -> Iterable[list[list[int]]]:
    return _pair_corpus(pairs())


def _unit_sequence(ids: Tensor | Sequence[int]) -> list[list[int]]:
    if hasattr(ids, "reshape") and hasattr(ids, "tolist"):
        values = ids.reshape(-1).tolist()
    else:
        values = list(ids)
    return [[int(value)] for value in values]


def _validate_cached_bpe(
    path: Path,
    config: BPEConfig,
    *,
    datasets: tuple[Mapping[str, object], ...] = (),
) -> None:
    state_path = _bpe_state_path(path)
    if not state_path.exists():
        raise FileNotFoundError(
            f"LongCat BPE state not found at {state_path}; run tokenizer preparation first."
        )

    meta_path = _bpe_meta_path(path)
    if not meta_path.exists():
        raise FileNotFoundError(f"LongCat BPE metadata not found at {meta_path}.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {
        "codec_name": config.codec_name,
        "requested_vocab_size": config.vocab_size,
        "min_frequency": config.min_frequency,
        "max_token_length": config.max_token_length,
        "codebook_sizes": list(config.codebook_sizes),
    }
    mismatches = {
        key: (meta.get(key), value)
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if "requested_vocab_size" not in meta and meta.get("vocab_size") == config.vocab_size:
        mismatches.pop("requested_vocab_size", None)
    if mismatches:
        details = ", ".join(
            f"{key}: cached={cached!r}, requested={requested!r}"
            for key, (cached, requested) in mismatches.items()
        )
        raise ValueError(f"LongCat BPE cache config mismatch at {path}: {details}.")
    if "actual_vocab_size" not in meta:
        meta["requested_vocab_size"] = meta.get("requested_vocab_size", meta.get("vocab_size"))
        meta["actual_vocab_size"] = CodecBPE.from_pretrained(path).vocab_size
        meta.pop("vocab_size", None)
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if datasets and tuple(meta.get("datasets", ())) != datasets:
        raise ValueError(f"LongCat BPE cache dataset mismatch at {path}.")


def _write_bpe_meta(path: Path, meta: BPEArtifactMeta) -> None:
    payload = json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n"
    _bpe_meta_path(path).write_text(payload, encoding="utf-8")


def _bpe_state_path(path: Path) -> Path:
    return path / "codec_bpe.json"


def _bpe_meta_path(path: Path) -> Path:
    return path / "meta.json"
