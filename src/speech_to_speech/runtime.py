from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from anytrain.codec import LongCatAudioCodec
from anytrain.tokenizer import IntBPE
from .config import BPEConfig, DatasetInput, ModelConfig
from .types import BPEArtifactMeta, SpeechPair, TranslationExample

if TYPE_CHECKING:
    from anytrain.codec import LongCatDecoderName
    from transformers import PreTrainedTokenizerBase
    from torch import Tensor

_QWEN3_TOKENIZERS: dict[tuple[str, bool], PreTrainedTokenizerBase] = {}
_LONGCAT_TOKENIZERS: dict[Path, IntBPE] = {}
_LONGCAT_CODEC: LongCatAudioCodec | None = None


def qwen3_tokenizer(
    config: ModelConfig | None = None,
    *,
    model_name_or_path: str | None = None,
    trust_remote_code: bool | None = None,
) -> PreTrainedTokenizerBase:
    config = config or ModelConfig()
    name = model_name_or_path or config.model_name_or_path
    trust = config.trust_remote_code if trust_remote_code is None else trust_remote_code
    key = (name, trust)
    if key not in _QWEN3_TOKENIZERS:
        from transformers import AutoTokenizer

        _QWEN3_TOKENIZERS[key] = AutoTokenizer.from_pretrained(
            name, trust_remote_code=trust
        )
    return _QWEN3_TOKENIZERS[key]


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
) -> IntBPE:
    config = config or BPEConfig()
    path = longcat_bpe_path(config, cache_dir=cache_dir)
    if path not in _LONGCAT_TOKENIZERS:
        _validate_cached_bpe(path, config)
        _LONGCAT_TOKENIZERS[path] = IntBPE.from_pretrained(path)
    return _LONGCAT_TOKENIZERS[path]


def prepare_longcat_tokenizer(
    pairs: Iterable[SpeechPair | TranslationExample],
    *,
    datasets: Iterable[DatasetInput] = (),
    config: BPEConfig | None = None,
    cache_dir: str | Path | None = None,
) -> IntBPE:
    config = config or BPEConfig()
    path = longcat_bpe_path(config, cache_dir=cache_dir)
    if _bpe_state_path(path).exists():
        return longcat_tokenizer(config, cache_dir=cache_dir)

    bpe = IntBPE.train(
        _chunked_pair_corpus(pairs, config.max_piece_frames),
        vocab_size=config.vocab_size,
    )
    path.mkdir(parents=True, exist_ok=True)
    bpe.save_pretrained(path)
    _write_bpe_meta(
        path,
        BPEArtifactMeta(
            codec_name=config.codec_name,
            vocab_size=config.vocab_size,
            max_piece_frames=config.max_piece_frames,
            datasets=tuple(_dataset_meta(dataset) for dataset in datasets),
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


def _chunked_pair_corpus(
    pairs: Iterable[SpeechPair | TranslationExample],
    max_piece_frames: int,
) -> Iterable[list[int]]:
    if max_piece_frames <= 0:
        raise ValueError("max_piece_frames must be positive.")

    for pair in pairs:
        yield from _chunks(_unit_sequence(pair.source_ids), max_piece_frames)
        yield from _chunks(_unit_sequence(pair.target_ids), max_piece_frames)


def _unit_sequence(ids: Tensor | Sequence[int]) -> list[int]:
    if hasattr(ids, "reshape") and hasattr(ids, "tolist"):
        values = ids.reshape(-1).tolist()
    else:
        values = list(ids)
    return [int(value) for value in values]


def _chunks(units: Sequence[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(units), size):
        chunk = [int(unit) for unit in units[start : start + size]]
        if chunk:
            yield chunk


def _validate_cached_bpe(path: Path, config: BPEConfig) -> None:
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
        "vocab_size": config.vocab_size,
        "max_piece_frames": config.max_piece_frames,
    }
    mismatches = {
        key: (meta.get(key), value)
        for key, value in expected.items()
        if meta.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: cached={cached!r}, requested={requested!r}"
            for key, (cached, requested) in mismatches.items()
        )
        raise ValueError(f"LongCat BPE cache config mismatch at {path}: {details}.")


def _dataset_meta(dataset: DatasetInput) -> Mapping[str, object]:
    from anydataset import resolve_dataset

    return resolve_dataset(dataset).to_dict()


def _write_bpe_meta(path: Path, meta: BPEArtifactMeta) -> None:
    payload = json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n"
    _bpe_meta_path(path).write_text(payload, encoding="utf-8")


def _bpe_state_path(path: Path) -> Path:
    return path / "int_bpe.json"


def _bpe_meta_path(path: Path) -> Path:
    return path / "meta.json"
