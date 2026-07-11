from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, cast

from anydataset.types import AudioView
from anytrain.codec.longcat import LongCat
from anytrain.idspace import Layout
from transformers import AutoModelForCausalLM, AutoTokenizer

from .audio_tokenizer import NativeAudioTokenizer, TorchCodecBPE
from .special_tokens import AudioSpecialToken, TextSpecialToken
from .types import AudioTokenizer, Backbone, Codec, TextTokenizer

if TYPE_CHECKING:
    from anytrain.framework.flow_matching import ContinuousFlowRuntime


@dataclass(frozen=True)
class Config:
    codec: str = "longcat"
    backbone: str = "Qwen/Qwen3-0.6B"
    audio_tokenizer: str | None = None
    device: str | None = None

    @property
    def audio_view(self):
        if self.codec == "longcat":
            return AudioView.LONGCAT
        else:
            raise ValueError()


class _CodecContract:
    """Expose the runtime codec through the local model-facing contract."""

    def __init__(self, codec: LongCat) -> None:
        self._codec = codec
        decoders = list(codec.decoders.values())
        if not decoders or not isinstance(
            getattr(decoders[0], "latent_dim", None), int
        ):
            raise TypeError("LongCat decoder must expose an integer latent_dim.")
        self._acoustic_feature_dim = decoders[0].latent_dim

    @property
    def acoustic_feature_dim(self) -> int:
        return self._acoustic_feature_dim

    @property
    def semantic_codebook(self):
        return self._codec.semantic_codebook

    def encode(self, audio, sample_rate):
        return self._codec.encode(audio, sample_rate)

    def decode(self, semantic_codes, acoustic_codes):
        return self._codec.decode(semantic_codes, acoustic_codes)

    def acoustic_codes_to_features(self, acoustic_codes):
        return self._codec.acoustic_codes_to_features(acoustic_codes)

    def decode_features(self, semantic_codes, acoustic_features):
        return self._codec.decode_features(semantic_codes, acoustic_features)


@dataclass(frozen=True)
class Runtime:
    config: Config

    @cached_property
    def text_tokenizer(self) -> TextTokenizer:
        return cast(TextTokenizer, AutoTokenizer.from_pretrained(self.config.backbone))

    @cached_property
    def backbone(self) -> Backbone:
        return cast(
            Backbone,
            AutoModelForCausalLM.from_pretrained(self.config.backbone),
        )

    @cached_property
    def codec(self) -> Codec:
        if self.config.codec != "longcat":
            raise NotImplementedError(f"unsupported codec: {self.config.codec}")
        return cast(
            Codec, _CodecContract(LongCat.from_pretrained(device=self.config.device))
        )

    @cached_property
    def audio_tokenizer(self) -> AudioTokenizer:
        if self.config.audio_tokenizer is None:
            return NativeAudioTokenizer(
                vocab_size=int(self.codec.semantic_codebook.size(0))
            )
        return _audio_tokenizer(self.config.audio_tokenizer)

    @cached_property
    def layout(self) -> Layout:
        text_vocab_size = len(self.text_tokenizer)
        audio_vocab_size = self.audio_tokenizer.vocab_size + 2
        return Layout(
            text=(0, text_vocab_size),
            audio=(text_vocab_size, text_vocab_size + audio_vocab_size),
        )

    @cached_property
    def flow_matching(self) -> ContinuousFlowRuntime:
        from anytrain.framework.flow_matching import ContinuousFlowRuntime

        return ContinuousFlowRuntime()

    def _text_special_id(self, token: TextSpecialToken) -> int:
        ids = self.text_tokenizer.encode(token.value, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"text token {token.value!r} must map to one id.")
        return ids[0]

    @cached_property
    def text_special_ids(self) -> dict[TextSpecialToken, int]:
        return {token: self._text_special_id(token) for token in TextSpecialToken}

    @property
    def pad_token_id(self):
        return self.text_special_ids[TextSpecialToken.PAD]

    @property
    def bos_token_id(self):
        return self.text_special_ids[TextSpecialToken.BOS]

    @property
    def eos_token_id(self):
        return self.text_special_ids[TextSpecialToken.EOS]

    @property
    def boa_token_id(self):
        return len(self.text_tokenizer) + self.audio_tokenizer.vocab_size

    @property
    def eoa_token_id(self):
        return self.boa_token_id + 1


_runtime: Runtime | None = None


def init_runtime(config: Config) -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = Runtime(config=config)
    elif _runtime.config != config:
        raise RuntimeError("runtime is already initialized with a different config.")
    return _runtime


def runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        raise RuntimeError("runtime is not initialized")
    return _runtime


def _audio_vocab_size(name: str) -> int:
    suffix = name.rsplit("_", maxsplit=1)[-1]
    if suffix.endswith("k") and suffix[:-1].isdigit():
        return int(suffix[:-1]) * 1000
    if suffix.isdigit():
        return int(suffix)
    raise ValueError(f"unable to infer audio tokenizer vocab size from {name!r}.")


def _audio_tokenizer(name: str) -> AudioTokenizer:
    from zhuyin.env import bpe_cache_dir, configure_environment
    from zhuyin.tokenizers.longcat import longcat_bpe

    configure_environment()
    path = bpe_cache_dir() / name
    if path.exists():
        return cast(AudioTokenizer, TorchCodecBPE.from_pretrained(path))
    if "/" in name:
        return cast(AudioTokenizer, TorchCodecBPE.from_pretrained(Path(name).expanduser()))
    return cast(
        AudioTokenizer,
        TorchCodecBPE.wrap(longcat_bpe(vocab_size=_audio_vocab_size(name))),
    )
