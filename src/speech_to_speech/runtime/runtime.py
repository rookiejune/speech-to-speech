from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, cast

import torch
from anydataset.types import AudioView, Modality
from anytrain.module.idspace import Layout
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .audio_tokenizer import FlattenedAudioTokenizer, NativeAudioTokenizer, TorchCodecBPE
from .codec import load_codec
from .special_tokens import Qwen3SpecialToken
from .types import AudioTokenizer, Backbone, Codec, SemanticCodec, TextTokenizer
from .._compat import StrEnum, auto

if TYPE_CHECKING:
    from anytrain.framework.flow_matching import ContinuousFlowRuntime
    from semantic_acoustic_codec.runtime import CodecBackend

_FLOW_METHODS = frozenset(
    {
        "adaptive_heun",
        "bosh3",
        "dopri5",
        "dopri8",
        "euler",
        "explicit_adams",
        "fehlberg2",
        "fixed_adams",
        "heun2",
        "heun3",
        "implicit_adams",
        "midpoint",
        "rk4",
        "scipy_solver",
    }
)


class AudioRepresentation(StrEnum):
    DECOUPLED = auto()
    FULL_CODEC_SEQUENCE = auto()


@dataclass(frozen=True)
class Config:
    codec: str = "longcat"
    backbone: str = "Qwen/Qwen3-0.6B"
    audio_representation: AudioRepresentation = AudioRepresentation.DECOUPLED
    audio_tokenizer: Optional[Union[str, Path]] = None
    semantic_codec_artifact: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    attn_implementation: Optional[str] = None
    flow_method: str = "midpoint"
    flow_nfe: int = 20
    flow_num_steps: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.audio_representation, AudioRepresentation):
            raise TypeError("audio_representation must be an AudioRepresentation.")
        if (
            self.audio_representation is AudioRepresentation.FULL_CODEC_SEQUENCE
            and self.audio_tokenizer is not None
        ):
            raise ValueError(
                "full codec sequence representation cannot use a BPE audio tokenizer."
            )
        if (
            self.codec == "bicodec"
            and self.audio_representation is not AudioRepresentation.FULL_CODEC_SEQUENCE
        ):
            raise ValueError("BiCodec requires the full codec sequence representation.")
        if self.semantic_codec_artifact is not None:
            if not self.semantic_codec_artifact:
                raise ValueError("semantic_codec_artifact must not be empty.")
            if self.audio_representation is AudioRepresentation.FULL_CODEC_SEQUENCE:
                raise ValueError(
                    "semantic codec artifacts require the decoupled audio representation."
                )
            if self.codec != "longcat":
                raise ValueError(
                    "semantic codec artifacts currently require the longcat codec."
                )
        if self.flow_method not in _FLOW_METHODS:
            raise ValueError(f"unsupported flow method: {self.flow_method}")
        if self.flow_nfe <= 0:
            raise ValueError(f"flow_nfe must be positive, got {self.flow_nfe}.")
        if self.flow_num_steps < 2:
            raise ValueError(
                "flow_num_steps must be at least 2, "
                f"got {self.flow_num_steps}."
            )

    @property
    def audio_view(self) -> AudioView:
        try:
            return AudioView(self.codec)
        except ValueError as error:
            raise ValueError(f"unsupported codec: {self.codec}") from error


@dataclass(frozen=True)
class Runtime:
    config: Config

    @property
    def codec_name(self) -> str:
        return self.config.codec

    @property
    def audio_view(self) -> AudioView:
        return self.config.audio_view

    @property
    def codec_frame_rate(self) -> float:
        frame_rate = float(self.codec.frame_rate)
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("codec frame_rate must be finite and positive.")
        return frame_rate

    @property
    def audio_representation(self) -> AudioRepresentation:
        return self.config.audio_representation

    @property
    def acoustic_side_channel(self) -> bool:
        return (
            self.audio_representation is AudioRepresentation.DECOUPLED
            and bool(self.codec.acoustic_codebook_sizes)
        )

    @cached_property
    def text_tokenizer(self) -> TextTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.config.backbone)
        return cast(TextTokenizer, cast(object, tokenizer))

    @cached_property
    def backbone(self) -> Backbone:
        kwargs = {}
        if self.config.dtype is not None:
            kwargs["dtype"] = dtype(self.config.dtype)
        if self.config.attn_implementation is not None:
            kwargs["attn_implementation"] = self.config.attn_implementation
        backbone = AutoModelForCausalLM.from_pretrained(self.config.backbone, **kwargs)
        if self.config.device is not None:
            backbone = cast(nn.Module, cast(object, backbone)).to(self.config.device)
        return cast(Backbone, cast(object, backbone))

    @cached_property
    def codec(self) -> Codec:
        return load_codec(self.config.codec, self.config.device)

    @cached_property
    def semantic_codec(self) -> SemanticCodec:
        artifact = self.config.semantic_codec_artifact
        if artifact is None:
            return self.codec
        from semantic_acoustic_codec.runtime import SemanticCodecRuntime, load_artifact

        support = load_artifact(
            Path(artifact).expanduser(),
            device=self.config.device,
        )
        runtime = SemanticCodecRuntime(
            support,
            cast("CodecBackend", cast(object, self.codec)),
        )
        return cast(SemanticCodec, cast(object, runtime))

    @cached_property
    def audio_tokenizer(self) -> AudioTokenizer:
        if self.config.audio_representation is AudioRepresentation.FULL_CODEC_SEQUENCE:
            return FlattenedAudioTokenizer(
                codebook_sizes=self.codec.codebook_sizes,
                codec_name=self.codec_name,
            )
        if self.config.audio_tokenizer is None:
            return NativeAudioTokenizer(vocab_size=int(self.codec.semantic_codebook.size(0)))
        return audio_tokenizer(self.config.audio_tokenizer)

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
        from anytrain.framework.flow_matching import ContinuousFlowRuntime, ODESampler

        return ContinuousFlowRuntime(
            sampler=ODESampler(
                method=self.config.flow_method,
                nfe=self.config.flow_nfe,
                num_steps=self.config.flow_num_steps,
                return_intermediates=False,
            ),
        )

    def _text_special_id(self, token: Qwen3SpecialToken) -> int:
        ids = self.text_tokenizer.encode(token.value, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"text token {token.value!r} must map to one id.")
        return ids[0]

    @cached_property
    def pad_token_id(self) -> int:
        return self._text_special_id(Qwen3SpecialToken.PAD)

    @cached_property
    def bos_token_id(self) -> int:
        return self._text_special_id(Qwen3SpecialToken.BOS)

    @cached_property
    def eos_token_id(self) -> int:
        return self._text_special_id(Qwen3SpecialToken.EOS)

    @property
    def boa_token_id(self) -> int:
        return len(self.text_tokenizer) + self.audio_tokenizer.vocab_size

    @property
    def eoa_token_id(self) -> int:
        return self.boa_token_id + 1

    @property
    def audio_head_range(self) -> tuple[int, int]:
        return self.layout.blocks[Modality.AUDIO.value]

    @property
    def codec_audio_range(self) -> tuple[int, int]:
        start, _ = self.audio_head_range
        return start, self.boa_token_id

    @cached_property
    def audio_generation_allowed_ids(self) -> tuple[int, ...]:
        start, end = self.codec_audio_range
        return (*range(start, end), self.eoa_token_id)

    @cached_property
    def text_generation_allowed_ids(self) -> tuple[int, ...]:
        start, end = self.layout.blocks[Modality.TEXT.value]
        blocked = {self.pad_token_id, self.bos_token_id}
        return tuple(token_id for token_id in range(start, end) if token_id not in blocked)

    def generation_allowed_ids(self, modality: Modality) -> tuple[int, ...]:
        if modality is Modality.AUDIO:
            return self.audio_generation_allowed_ids
        if modality is Modality.TEXT:
            return self.text_generation_allowed_ids
        raise ValueError(f"unsupported generation modality: {modality.value}")

    def is_codec_audio_id(self, token_id: int) -> bool:
        start, end = self.codec_audio_range
        return start <= token_id < end


def audio_tokenizer(path: str | Path) -> AudioTokenizer:
    from zhuyin.tokenizers.codec_bpe import codec_bpe

    tokenizer = codec_bpe(Path(path).expanduser())
    return cast(AudioTokenizer, cast(object, TorchCodecBPE.wrap(tokenizer)))


def dtype(value: str) -> torch.dtype:
    try:
        result = getattr(torch, value)
    except AttributeError as error:
        raise ValueError(f"unknown torch dtype: {value}") from error
    if not isinstance(result, torch.dtype):
        raise ValueError(f"unknown torch dtype: {value}")
    return result
