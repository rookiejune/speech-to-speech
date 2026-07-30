from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, cast

import torch
from anytrain.codec import AcousticLayout
from anydataset.types import AudioView, Modality
from anytrain.module.idspace import Layout
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .audio_tokenizer import (
    BiCodecAudioTokenizer,
    FlattenedAudioTokenizer,
    NativeAudioTokenizer,
    TorchCodecBPE,
)
from .codec import load_codec
from .types import (
    AudioTokenizer,
    Backbone,
    CodecBackend,
    SemanticCodec,
    TextTokenizer,
    codec_frame_rate as validated_codec_frame_rate,
    frame_codec,
    structured_codec,
    supports_acoustic,
    supports_structured,
)
from .._compat import StrEnum, auto
from ..audio_route import (
    BICODEC_GENERATE_GLOBAL,
    BICODEC_REUSE_PROMPT_GLOBAL,
    FULL_OUTPUT,
    SEMANTIC_GENERATOR,
    Config as AudioRouteConfig,
    StreamSource,
)

if TYPE_CHECKING:
    from anytrain.codec import SemanticAcousticCodec
    from anytrain.framework.flow_matching import ContinuousFlowRuntime

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


class BackboneInitialization(StrEnum):
    PRETRAINED = auto()
    RANDOM = auto()


@dataclass(frozen=True)
class Config:
    codec: str = "longcat"
    backbone: str = "Qwen/Qwen3-0.6B"
    backbone_initialization: BackboneInitialization = BackboneInitialization.PRETRAINED
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
        if not isinstance(self.backbone_initialization, BackboneInitialization):
            raise TypeError("backbone_initialization must be a BackboneInitialization.")
        if not isinstance(self.audio_representation, AudioRepresentation):
            raise TypeError("audio_representation must be an AudioRepresentation.")
        if (
            self.audio_representation is AudioRepresentation.FULL_CODEC_SEQUENCE
            and self.audio_tokenizer is not None
        ):
            raise ValueError(
                "full codec sequence representation cannot use a BPE audio tokenizer."
            )
        if self.semantic_codec_artifact is not None:
            if not self.semantic_codec_artifact:
                raise ValueError("semantic_codec_artifact must not be empty.")
            if self.audio_representation is AudioRepresentation.FULL_CODEC_SEQUENCE:
                raise ValueError(
                    "semantic codec artifacts require the decoupled audio representation."
                )
            if self.audio_view not in {AudioView.LONGCAT, AudioView.BICODEC}:
                raise ValueError(
                    "semantic codec artifacts currently require LongCat or BiCodec."
                )
        if self.audio_view is AudioView.BICODEC and self.semantic_codec_artifact is None:
            if self.audio_representation is not AudioRepresentation.FULL_CODEC_SEQUENCE:
                raise ValueError(
                    "BiCodec is a fixed-length structured codec and requires "
                    "semantic_codec_artifact for semantic-only decoding or "
                    "full_codec_sequence for token-only decoding."
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
        if self.codec == "stable_codec":
            return AudioView.STABLE
        try:
            return AudioView(self.codec)
        except ValueError as error:
            raise ValueError(f"unsupported codec: {self.codec}") from error


@dataclass(frozen=True)
class Runtime:
    config: Config
    audio_route: AudioRouteConfig | None = None

    def __post_init__(self) -> None:
        if self.audio_route is not None and not isinstance(
            self.audio_route,
            AudioRouteConfig,
        ):
            raise TypeError("runtime audio_route must be an audio route Config.")
        validate_audio_route(self.config, self.audio_route)

    @property
    def codec_name(self) -> str:
        return self.config.codec

    @property
    def audio_view(self) -> AudioView:
        return self.config.audio_view

    @property
    def codec_frame_rate(self) -> float:
        return validated_codec_frame_rate(self.codec)

    @property
    def audio_representation(self) -> AudioRepresentation:
        return self.config.audio_representation

    @property
    def semantic_codec_artifact(self) -> str | None:
        return self.config.semantic_codec_artifact

    @property
    def acoustic_side_channel(self) -> bool:
        return (
            self.audio_representation is AudioRepresentation.DECOUPLED
            and self.config.semantic_codec_artifact is None
            and supports_acoustic(self.codec)
        )

    @property
    def structured_full_sequence(self) -> bool:
        if self.audio_representation is not AudioRepresentation.FULL_CODEC_SEQUENCE:
            return False
        if not supports_structured(self.codec):
            return False
        return structured_codec(self.codec).acoustic_layout is AcousticLayout.FIXED_LENGTH

    @property
    def acoustic_layout(self) -> AcousticLayout:
        if supports_structured(self.codec):
            return AcousticLayout(structured_codec(self.codec).acoustic_layout)
        return AcousticLayout.FRAME_ALIGNED

    @property
    def acoustic_unit_length(self) -> int | None:
        if supports_structured(self.codec):
            return structured_codec(self.codec).acoustic_unit_length
        return None

    @cached_property
    def text_tokenizer(self) -> TextTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.config.backbone)
        bind_chat_bos(tokenizer)
        return cast(TextTokenizer, cast(object, tokenizer))

    @cached_property
    def backbone(self) -> Backbone:
        kwargs = {}
        if self.config.dtype is not None:
            kwargs["dtype"] = dtype(self.config.dtype)
        if self.config.attn_implementation is not None:
            kwargs["attn_implementation"] = self.config.attn_implementation
        if self.config.backbone_initialization is BackboneInitialization.PRETRAINED:
            backbone = AutoModelForCausalLM.from_pretrained(
                self.config.backbone,
                **kwargs,
            )
        else:
            config = AutoConfig.from_pretrained(self.config.backbone)
            backbone = AutoModelForCausalLM.from_config(config, **kwargs)
        if self.config.device is not None:
            backbone = cast(nn.Module, cast(object, backbone)).to(self.config.device)
        return cast(Backbone, cast(object, backbone))

    @cached_property
    def codec(self) -> CodecBackend:
        return load_codec(self.config.codec, self.config.device)

    @property
    def semantic_codebook_sizes(self) -> tuple[int, ...]:
        if supports_structured(self.codec):
            return tuple(structured_codec(self.codec).semantic_codebook_sizes)
        return tuple(frame_codec(self.codec).codebook_sizes)

    @cached_property
    def semantic_codec(self) -> SemanticCodec:
        artifact = self.config.semantic_codec_artifact
        if artifact is None:
            raise RuntimeError(
                "semantic-only waveform decoding requires runtime.semantic_codec_artifact; "
                "use full_codec_sequence for FrameCodec token generation."
            )
        from semantic_acoustic_codec.runtime import SemanticCodecRuntime, load_artifact

        support = load_artifact(
            Path(artifact).expanduser(),
            device=self.config.device,
        )
        runtime = SemanticCodecRuntime(
            support,
            cast("SemanticAcousticCodec", cast(object, self.codec)),
        )
        return cast(SemanticCodec, cast(object, runtime))

    @cached_property
    def audio_tokenizer(self) -> AudioTokenizer:
        if self.config.audio_representation is AudioRepresentation.FULL_CODEC_SEQUENCE:
            if self.structured_full_sequence:
                codec = structured_codec(self.codec)
                return BiCodecAudioTokenizer(
                    semantic_vocab_size=self.semantic_codebook_sizes[0],
                    acoustic_codebook_sizes=codec.acoustic_codebook_sizes,
                    acoustic_unit_length=codec.acoustic_unit_length,
                )
            return FlattenedAudioTokenizer(
                codebook_sizes=frame_codec(self.codec).codebook_sizes,
                codec_name=self.codec_name,
            )
        if self.config.audio_tokenizer is None:
            return NativeAudioTokenizer(vocab_size=int(self.semantic_codebook_sizes[0]))
        return audio_tokenizer(self.config.audio_tokenizer)

    @cached_property
    def layout(self) -> Layout:
        text_vocab_size = len(self.text_tokenizer)
        audio_vocab_size = self.audio_tokenizer.vocab_size + 3
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

    @cached_property
    def pad_token_id(self) -> int:
        return text_special_id(self.text_tokenizer, "pad_token_id")

    @cached_property
    def bos_token_id(self) -> int:
        return text_special_id(self.text_tokenizer, "bos_token_id")

    @cached_property
    def eos_token_id(self) -> int:
        return text_special_id(self.text_tokenizer, "eos_token_id")

    @property
    def boa_token_id(self) -> int:
        return len(self.text_tokenizer) + self.audio_tokenizer.vocab_size

    @property
    def eoa_token_id(self) -> int:
        return self.boa_token_id + 1

    @property
    def mask_token_id(self) -> int:
        return self.boa_token_id + 2

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


def validate_audio_route(
    config: Config,
    route: AudioRouteConfig | None,
) -> None:
    """Validate that an audio route is executable by the runtime representation."""
    if route is None:
        return
    if not route.output.streams:
        raise ValueError("runtime audio route must declare at least one output stream.")

    representation = config.audio_representation
    if representation is AudioRepresentation.DECOUPLED:
        if route != SEMANTIC_GENERATOR:
            raise ValueError(
                "decoupled audio representation requires audio_route=semantic_generator."
            )
        return

    if StreamSource.GENERATOR in {
        route.decode.semantic,
        route.decode.acoustic,
    }:
        raise ValueError(
            "full codec sequence routes cannot use generator-owned decode streams."
        )
    if config.audio_view is AudioView.BICODEC:
        if route not in (
            BICODEC_GENERATE_GLOBAL,
            BICODEC_REUSE_PROMPT_GLOBAL,
        ):
            raise ValueError(
                "BiCodec full codec sequence requires a supported global route."
            )
        return
    if route != FULL_OUTPUT:
        raise ValueError(
            "frame codec full sequence representation requires audio_route=full_output."
        )


def audio_tokenizer(path: str | Path) -> AudioTokenizer:
    from zhuyin.tokenizers.codec_bpe import codec_bpe

    tokenizer = codec_bpe(Path(path).expanduser())
    return cast(AudioTokenizer, cast(object, TorchCodecBPE.wrap(tokenizer)))


# Qwen chat turn start; stock HF leaves bos_token unset while keeping this in vocab.
_CHAT_BOS_TOKEN = "<|im_start|>"


def bind_chat_bos(tokenizer: object) -> None:
    """Expose chat turn-start as bos when the tokenizer leaves bos unset."""
    if getattr(tokenizer, "bos_token_id", None) is not None:
        return
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return
    token_id = convert(_CHAT_BOS_TOKEN)
    if isinstance(token_id, bool) or not isinstance(token_id, Integral):
        return
    token_id = int(token_id)
    if token_id < 0:
        return
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and token_id == int(unk):
        return
    setattr(tokenizer, "bos_token", _CHAT_BOS_TOKEN)


def text_special_id(tokenizer: TextTokenizer, name: str) -> int:
    """Resolve a required text special-token id from HF tokenizer attributes."""
    if name not in {"pad_token_id", "bos_token_id", "eos_token_id"}:
        raise ValueError(f"unsupported text special token attribute: {name}.")
    token_id = getattr(tokenizer, name)
    if token_id is not None:
        return _token_id(token_id, name)
    map_key = name[: -len("_id")]
    token = tokenizer.special_tokens_map.get(map_key)
    if token is None:
        raise ValueError(f"text tokenizer is missing {name}.")
    if not isinstance(token, str):
        raise TypeError(f"text tokenizer {map_key} must be a string.")
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"text token {token!r} must map to one id.")
    return _token_id(ids[0], name)


def _token_id(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"text tokenizer {name} must be an integer.")
    token_id = int(value)
    if token_id < 0:
        raise ValueError(f"text tokenizer {name} must be non-negative.")
    return token_id


def dtype(value: str) -> torch.dtype:
    try:
        result = getattr(torch, value)
    except AttributeError as error:
        raise ValueError(f"unknown torch dtype: {value}") from error
    if not isinstance(result, torch.dtype):
        raise ValueError(f"unknown torch dtype: {value}")
    return result
