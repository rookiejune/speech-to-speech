from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, cast

from anytrain.codec import AcousticLayout
from anydataset.types import AudioView, Modality
from anytrain.module.idspace import Layout

from .audio_tokenizer import (
    BiCodecAudioTokenizer,
    FlattenedAudioTokenizer,
    NativeAudioTokenizer,
    TorchCodecBPE,
)
from .backbone import (
    AdapterConfig as BackboneAdapterConfig,
    BackboneAdapter,
    BackboneInitialization,
    BackboneType,
    bind_chat_bos as bind_chat_bos,
    create as create_backbone_adapter,
    dtype as dtype,
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
    validate_backbone_readout,
)
from .._compat import StrEnum, auto

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


class AudioSequenceLayout(StrEnum):
    FLATTENED = auto()
    SEMANTIC = auto()


@dataclass(frozen=True)
class Config:
    codec: str = "longcat"
    backbone_type: BackboneType = BackboneType.HF_CAUSAL_LM
    backbone: str = "Qwen/Qwen3-0.6B"
    backbone_initialization: BackboneInitialization = BackboneInitialization.PRETRAINED
    backbone_trust_remote_code: bool = False
    backbone_chat_template: Optional[str] = None
    backbone_readout: str = "last_hidden_state"
    backbone_readouts: dict[str, str] = field(default_factory=dict)
    backbone_supports_cache_position: bool = True
    backbone_module: str = ""
    backbone_body: str = "base_model"
    audio_tokenizer: Optional[Union[str, Path]] = None
    semantic_codec_artifact: Optional[str] = None
    device: Optional[str] = None
    dtype: Optional[str] = None
    attn_implementation: Optional[str] = None
    gradient_checkpointing: bool = False
    flow_method: str = "midpoint"
    flow_nfe: int = 20
    flow_num_steps: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.backbone_type, BackboneType):
            raise TypeError("backbone_type must be a BackboneType.")
        if not isinstance(self.backbone_initialization, BackboneInitialization):
            raise TypeError("backbone_initialization must be a BackboneInitialization.")
        if not isinstance(self.backbone_trust_remote_code, bool):
            raise TypeError("backbone_trust_remote_code must be a bool.")
        _validate_optional_nonempty_string(
            self.backbone_chat_template,
            "backbone_chat_template",
        )
        validate_backbone_readout(self.backbone_readout)
        _validate_backbone_readouts(self.backbone_readouts)
        if not isinstance(self.backbone_supports_cache_position, bool):
            raise TypeError("backbone_supports_cache_position must be a bool.")
        _validate_path(self.backbone_module, "backbone_module")
        _validate_path(self.backbone_body, "backbone_body", allow_empty=False)
        if self.semantic_codec_artifact is not None:
            if not self.semantic_codec_artifact:
                raise ValueError("semantic_codec_artifact must not be empty.")
            if self.audio_view not in {AudioView.LONGCAT, AudioView.BICODEC}:
                raise ValueError(
                    "semantic codec artifacts currently require LongCat or BiCodec."
                )
        if not isinstance(self.gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a bool.")
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
    audio_sequence_layout: AudioSequenceLayout = AudioSequenceLayout.SEMANTIC

    def __post_init__(self) -> None:
        if not isinstance(self.audio_sequence_layout, AudioSequenceLayout):
            raise TypeError("audio_sequence_layout must be an AudioSequenceLayout.")
        _validate_sequence_layout_config(self.config, self.audio_sequence_layout)

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
    def semantic_codec_artifact(self) -> str | None:
        return self.config.semantic_codec_artifact

    @property
    def acoustic_side_channel(self) -> bool:
        return (
            self.audio_sequence_layout is AudioSequenceLayout.SEMANTIC
            and self.config.semantic_codec_artifact is None
            and supports_acoustic(self.codec)
        )

    @property
    def structured_full_sequence(self) -> bool:
        if self.audio_sequence_layout is not AudioSequenceLayout.FLATTENED:
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

    @property
    def backbone_trust_remote_code(self) -> bool:
        return self.config.backbone_trust_remote_code

    @property
    def backbone_readout(self) -> str:
        return self.config.backbone_readout

    @property
    def backbone_chat_template(self) -> str | None:
        return self.config.backbone_chat_template

    @property
    def backbone_readouts(self) -> Mapping[str, str]:
        return self.config.backbone_readouts

    @property
    def backbone_supports_cache_position(self) -> bool:
        return self.config.backbone_supports_cache_position

    @property
    def gradient_checkpointing(self) -> bool:
        return self.config.gradient_checkpointing

    @property
    def backbone_module(self) -> str:
        return self.config.backbone_module

    @property
    def backbone_body(self) -> str:
        return self.config.backbone_body

    @property
    def backbone_adapter_config(self) -> BackboneAdapterConfig:
        return BackboneAdapterConfig(
            type=self.config.backbone_type,
            path=self.config.backbone,
            initialization=self.config.backbone_initialization,
            trust_remote_code=self.config.backbone_trust_remote_code,
            chat_template=self.config.backbone_chat_template,
            readout=self.config.backbone_readout,
            readouts=self.config.backbone_readouts,
            supports_cache_position=self.config.backbone_supports_cache_position,
            module=self.config.backbone_module,
            body=self.config.backbone_body,
            device=self.config.device,
            dtype=self.config.dtype,
            attn_implementation=self.config.attn_implementation,
            gradient_checkpointing=self.config.gradient_checkpointing,
        )

    @cached_property
    def backbone_adapter(self) -> BackboneAdapter:
        return create_backbone_adapter(self.backbone_adapter_config)

    @cached_property
    def text_tokenizer(self) -> TextTokenizer:
        return self.backbone_adapter.text_tokenizer

    @cached_property
    def backbone(self) -> Backbone:
        return self.backbone_adapter.model

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
                "use audio_sequence_layout=flattened for token-only generation."
            )
        from semantic_acoustic_codec.runtime import SemanticCodecRuntime
        from semantic_acoustic_codec.runtime.artifact import load_artifact

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
        if self.audio_sequence_layout is AudioSequenceLayout.FLATTENED:
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
        if self.audio_view is AudioView.BICODEC:
            codec = structured_codec(self.codec)
            return BiCodecAudioTokenizer(
                semantic_vocab_size=self.semantic_codebook_sizes[0],
                acoustic_codebook_sizes=codec.acoustic_codebook_sizes,
                acoustic_unit_length=codec.acoustic_unit_length,
            )
        if self.config.audio_tokenizer is None:
            return NativeAudioTokenizer(vocab_size=int(self.semantic_codebook_sizes[0]))
        return audio_tokenizer(self.config.audio_tokenizer)

    @cached_property
    def layout(self) -> Layout:
        text_vocab_size = text_tokenizer_vocab_size(self.text_tokenizer)
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
        return text_tokenizer_vocab_size(self.text_tokenizer) + self.audio_tokenizer.vocab_size

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


def runtime_for_sequence_layout(config: Config, layout: AudioSequenceLayout) -> Runtime:
    return Runtime(
        config,
        audio_sequence_layout=layout,
    )


def _validate_sequence_layout_config(config: Config, layout: AudioSequenceLayout) -> None:
    if not isinstance(layout, AudioSequenceLayout):
        raise TypeError("audio_sequence_layout must be an AudioSequenceLayout.")
    if layout is AudioSequenceLayout.FLATTENED:
        if config.audio_tokenizer is not None:
            raise ValueError(
                "audio_sequence_layout=flattened cannot use a BPE audio tokenizer."
            )
        if config.semantic_codec_artifact is not None:
            raise ValueError(
                "runtime.semantic_codec_artifact requires audio_sequence_layout=semantic."
            )
    if layout is AudioSequenceLayout.SEMANTIC and config.audio_view is AudioView.BICODEC:
        if config.semantic_codec_artifact is None:
            raise ValueError(
                "BiCodec semantic audio_sequence_layout requires "
                "runtime.semantic_codec_artifact."
            )
        if config.audio_tokenizer is not None:
            raise ValueError(
                "BiCodec semantic audio_sequence_layout uses structured audio tokens "
                "and cannot use a BPE audio tokenizer."
            )


def audio_tokenizer(path: str | Path) -> AudioTokenizer:
    from zhuyin.tokenizers.codec_bpe import codec_bpe

    tokenizer = codec_bpe(Path(path).expanduser())
    return cast(AudioTokenizer, cast(object, TorchCodecBPE.wrap(tokenizer)))


def text_tokenizer_vocab_size(tokenizer: object) -> int:
    try:
        return _positive_integral(len(cast(TextTokenizer, tokenizer)), "text tokenizer length")
    except (AttributeError, NotImplementedError, TypeError):
        pass
    for attribute in ("vocab_size", "total_vocab_size"):
        try:
            value = getattr(tokenizer, attribute)
        except AttributeError:
            continue
        if value is None:
            continue
        return _positive_integral(value, f"text tokenizer {attribute}")
    raise AttributeError("text tokenizer does not expose a positive vocabulary size.")


def _positive_integral(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


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


def _validate_path(value: object, name: str, *, allow_empty: bool = True) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        if allow_empty:
            return
        raise ValueError(f"{name} must not be empty.")
    if value.startswith(".") or value.endswith(".") or ".." in value:
        raise ValueError(f"{name} must be a dotted attribute path.")
    for part in value.split("."):
        if not part.isidentifier():
            raise ValueError(f"{name} must contain identifier path components.")


def _validate_backbone_readouts(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("backbone_readouts must be a mapping.")
    for modality, readout in value.items():
        if not isinstance(modality, str):
            raise TypeError("backbone_readouts keys must be strings.")
        if modality not in {Modality.TEXT.value, Modality.AUDIO.value}:
            raise ValueError("backbone_readouts keys must be 'text' or 'audio'.")
        validate_backbone_readout(readout)


def _validate_optional_nonempty_string(value: object, name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} must be a string or None.")
    if value == "":
        raise ValueError(f"{name} must not be empty.")
