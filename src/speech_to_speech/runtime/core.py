from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, cast

from anydataset.types import AudioView, Modality
from anytrain.codec import AcousticLayout
from anytrain.module.idspace import Layout

from ._artifact import content_sha256
from .audio_tokenizer import (
    BiCodecAudioTokenizer,
    FlattenedAudioTokenizer,
    NativeAudioTokenizer,
)
from .backbone import (
    AdapterConfig as BackboneAdapterConfig,
    Backbone,
    BackboneAdapter,
    create as create_backbone_adapter,
)
from .codec import load_codec
from .codec_contract import (
    CodecBackend,
    SemanticCodec,
    codec_frame_rate as validated_codec_frame_rate,
    frame_codec,
    structured_codec,
    supports_acoustic,
    supports_structured,
)
from .config import AudioSequenceLayout, Config, validate_sequence_layout_config
from .tokenizer import AudioTokenizer, TextTokenizer
from .tokenizer_factory import (
    audio_tokenizer,
    text_special_id,
    text_tokenizer_vocab_size,
)

if TYPE_CHECKING:
    from anytrain.codec import SemanticAcousticCodec
    from anytrain.framework.flow_matching import ContinuousFlowRuntime


@dataclass(frozen=True)
class Runtime:
    config: Config
    audio_sequence_layout: AudioSequenceLayout = AudioSequenceLayout.SEMANTIC

    def __post_init__(self) -> None:
        if not isinstance(self.audio_sequence_layout, AudioSequenceLayout):
            raise TypeError("audio_sequence_layout must be an AudioSequenceLayout.")
        validate_sequence_layout_config(self.config, self.audio_sequence_layout)

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
    def acoustic_generator_artifact(self) -> str | None:
        return self.config.acoustic_generator_artifact

    @cached_property
    def acoustic_generator_artifact_sha256(self) -> str | None:
        artifact = self.config.acoustic_generator_artifact
        if artifact is None:
            return None
        return content_sha256(Path(artifact).expanduser())

    @property
    def acoustic_side_channel(self) -> bool:
        return (
            self.audio_sequence_layout is AudioSequenceLayout.SEMANTIC
            and self.config.acoustic_generator_artifact is None
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
        artifact = self.config.acoustic_generator_artifact
        if artifact is None:
            raise RuntimeError(
                "semantic-only waveform decoding requires runtime.acoustic_generator_artifact; "
                "use audio_sequence_layout=flattened for token-only generation."
            )
        if self.acoustic_generator_artifact_sha256 is None:
            raise RuntimeError("acoustic generator artifact identity was not resolved.")
        from semantic_acoustic_generator.runtime import GeneratorRuntime
        from semantic_acoustic_generator.runtime.artifact import load_artifact

        support = load_artifact(
            Path(artifact).expanduser(),
            device=self.config.device,
        )
        runtime = GeneratorRuntime(
            support,
            cast("SemanticAcousticCodec", cast(object, self.codec)),
        )
        return cast(SemanticCodec, cast(object, runtime))

    @cached_property
    def audio_tokenizer(self) -> AudioTokenizer:
        if self.audio_view is AudioView.BICODEC:
            codec = structured_codec(self.codec)
            semantic_tokenizer = (
                None
                if self.config.audio_tokenizer is None
                else audio_tokenizer(self.config.audio_tokenizer)
            )
            global_unit_length = codec.acoustic_unit_length
            if global_unit_length is None:
                raise ValueError("BiCodec requires a fixed global unit length.")
            return BiCodecAudioTokenizer(
                semantic_codebook_size=self.semantic_codebook_sizes[0],
                global_codebook_sizes=codec.acoustic_codebook_sizes,
                global_unit_length=global_unit_length,
                semantic_tokenizer=semantic_tokenizer,
            )
        if self.audio_sequence_layout is AudioSequenceLayout.FLATTENED:
            return FlattenedAudioTokenizer(
                codebook_sizes=frame_codec(self.codec).codebook_sizes,
                codec_name=self.codec_name,
            )
        if self.config.audio_tokenizer is None:
            return NativeAudioTokenizer(vocab_size=int(self.semantic_codebook_sizes[0]))
        return cast(
            AudioTokenizer,
            cast(object, audio_tokenizer(self.config.audio_tokenizer)),
        )

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

__all__ = ["Runtime"]
