from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, cast

from anydataset.types import AudioView, Modality
from anytrain.codec import AudioBackendIdentity, AudioCodeSchema, AudioCodeSpec
from anytrain.module.idspace import Layout

from ..task import ControlToken
from ._artifact import content_sha256
from .audio_tokenizer import (
    AudioTokenizer,
    BiCodecAudioTokenizer,
    FlattenedAudioTokenizer,
    NativeAudioTokenizer,
)
from .audio_schema import AudioTokenRegistry, AudioTokenSpec
from .backbone import (
    AdapterConfig as BackboneAdapterConfig,
    Backbone,
    BackboneAdapter,
    TextTokenizer,
    create as create_backbone_adapter,
)
from .codec import (
    audio_backend_identity,
    audio_code_spec,
    load_audio_detokenizer_backend,
    load_audio_tokenizer_backend,
)
from .codec_contract import (
    CodecBackend,
    SemanticCodec,
    acoustic_codec,
)
from .config import (
    AudioSequenceLayout,
    Config,
    validate_sequence_layout_config,
)
from .tokenizer_factory import (
    audio_tokenizer,
    text_special_id,
    text_tokenizer_vocab_size,
)

if TYPE_CHECKING:
    from anytrain.codec import SemanticAcousticCodec
    from anytrain.framework.flow_matching import ContinuousFlowRuntime


def _audio_tokens(
    *,
    name: str,
    view: AudioView,
    spec: AudioCodeSpec,
    bpe: str | Path | None,
    flattened: bool,
) -> AudioTokenizer:
    if view is AudioView.BICODEC:
        if spec.schema is not AudioCodeSchema.SEMANTIC_GLOBAL:
            raise ValueError("BiCodec audio tokens require a semantic-global code spec.")
        if spec.global_unit_length is None:
            raise ValueError("BiCodec audio code spec requires global_unit_length.")
        semantic = None if bpe is None else audio_tokenizer(bpe)
        return BiCodecAudioTokenizer(
            semantic_codebook_size=spec.semantic_codebook_sizes[0],
            global_codebook_sizes=spec.global_codebook_sizes,
            global_unit_length=spec.global_unit_length,
            semantic_tokenizer=semantic,
        )
    if flattened:
        if not spec.frame_codebook_sizes:
            raise ValueError(
                f"flattened audio tokenizer {name!r} requires frame codebook metadata."
            )
        return FlattenedAudioTokenizer(
            codebook_sizes=spec.frame_codebook_sizes,
            codec_name=name,
        )
    if bpe is not None:
        return cast(AudioTokenizer, cast(object, audio_tokenizer(bpe)))
    return NativeAudioTokenizer(vocab_size=int(spec.primary_codebook_sizes[0]))


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
        return self.config.audio_output.tokenizer

    @property
    def output_audio_tokenizer_name(self) -> str:
        return self.config.audio_output.tokenizer

    @property
    def output_audio_detokenizer_name(self) -> str | None:
        return self.config.audio_output.detokenizer

    @property
    def output_codec_name(self) -> str:
        return self.codec_name

    @property
    def input_audio_decoupled(self) -> bool:
        """Legacy name for whether input/output use distinct model token spaces."""

        return not self.input_audio_token_space_shared

    @property
    def input_audio_backend_shared(self) -> bool:
        return self.input_audio_backend_identity == self.output_audio_backend_identity

    @property
    def input_audio_backend_identity(self) -> AudioBackendIdentity:
        return audio_backend_identity(self.input_audio_tokenizer_name)

    @property
    def output_audio_backend_identity(self) -> AudioBackendIdentity:
        return audio_backend_identity(self.output_audio_tokenizer_name)

    @property
    def output_audio_detokenizer_identity(self) -> AudioBackendIdentity | None:
        name = self.output_audio_detokenizer_name
        return None if name is None else audio_backend_identity(name)

    @property
    def output_audio_detokenizer_backend_shared(self) -> bool:
        identity = self.output_audio_detokenizer_identity
        return identity is not None and identity in {
            self.input_audio_backend_identity,
            self.output_audio_backend_identity,
        }

    @property
    def input_audio_token_space_shared(self) -> bool:
        config = self.config.audio_input
        return (
            config is None
            or config.token_space_identity == self.config.audio_output.token_space_identity
        )

    @property
    def input_codec_name(self) -> str:
        config = self.config.audio_input
        if config is None or config.tokenizer is None:
            return self.codec_name
        return config.tokenizer

    @property
    def input_audio_tokenizer_name(self) -> str:
        return self.input_codec_name

    @property
    def audio_view(self) -> AudioView:
        return AudioView(self.output_audio_code_spec.view)

    @property
    def output_audio_view(self) -> AudioView:
        return self.audio_view

    @property
    def input_audio_view(self) -> AudioView:
        config = self.config.audio_input
        if config is None or config.tokenizer is None:
            return self.audio_view
        return AudioView(self.input_audio_code_spec.view)

    @cached_property
    def output_audio_code_spec(self) -> AudioCodeSpec:
        return audio_code_spec(self.output_audio_tokenizer_name)

    @cached_property
    def input_audio_code_spec(self) -> AudioCodeSpec:
        if self.input_audio_backend_shared:
            return self.output_audio_code_spec
        return audio_code_spec(self.input_audio_tokenizer_name)

    @property
    def codec_frame_rate(self) -> float:
        return self.output_audio_code_spec.frame_rate

    @property
    def output_codec_frame_rate(self) -> float:
        return self.codec_frame_rate

    @property
    def input_codec_frame_rate(self) -> float:
        input_config = self.config.audio_input
        configured = None if input_config is None else input_config.frame_rate
        backend_rate = self.input_audio_code_spec.frame_rate
        if configured is not None:
            if not math.isclose(float(configured), backend_rate):
                raise ValueError(
                    "runtime.audio_input frame_rate does not match the audio tokenizer preset."
                )
        return backend_rate

    @property
    def acoustic_generator_artifact(self) -> str | None:
        return self.config.audio_output.acoustic_generator_artifact

    @cached_property
    def acoustic_generator_artifact_sha256(self) -> str | None:
        artifact = self.config.audio_output.acoustic_generator_artifact
        if artifact is None:
            return None
        return content_sha256(Path(artifact).expanduser())

    @property
    def acoustic_side_channel(self) -> bool:
        return (
            self.audio_sequence_layout is AudioSequenceLayout.SEMANTIC
            and self.config.audio_output.acoustic_generator_artifact is None
            and self.output_audio_code_spec.schema is AudioCodeSchema.SEMANTIC_ACOUSTIC
        )

    @property
    def structured_full_sequence(self) -> bool:
        return (
            self.audio_sequence_layout is AudioSequenceLayout.FLATTENED
            and self.output_audio_code_spec.schema is AudioCodeSchema.SEMANTIC_GLOBAL
        )

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
    def output_audio_tokenizer_backend(self) -> CodecBackend:
        return load_audio_tokenizer_backend(
            self.output_audio_tokenizer_name,
            self.config.device,
        )

    @cached_property
    def codec(self) -> CodecBackend:
        """Deprecated output-tokenizer backend alias."""

        return self.output_audio_tokenizer_backend

    @cached_property
    def output_codec(self) -> CodecBackend:
        """Deprecated output-tokenizer backend alias."""

        return self.output_audio_tokenizer_backend

    @cached_property
    def input_audio_tokenizer_backend(self) -> CodecBackend:
        if self.input_audio_backend_shared:
            return self.output_audio_tokenizer_backend
        return load_audio_tokenizer_backend(
            self.input_audio_tokenizer_name,
            self.config.device,
        )

    @cached_property
    def input_codec(self) -> CodecBackend:
        """Deprecated input-tokenizer backend alias."""

        return self.input_audio_tokenizer_backend

    @cached_property
    def output_audio_detokenizer(self) -> CodecBackend | None:
        name = self.output_audio_detokenizer_name
        if name is None:
            return None
        identity = cast(AudioBackendIdentity, self.output_audio_detokenizer_identity)
        if identity == self.output_audio_backend_identity:
            return self.output_audio_tokenizer_backend
        if identity == self.input_audio_backend_identity:
            return self.input_audio_tokenizer_backend
        return load_audio_detokenizer_backend(name, self.config.device)

    @property
    def semantic_codebook_sizes(self) -> tuple[int, ...]:
        return self.output_audio_code_spec.primary_codebook_sizes

    @property
    def input_semantic_codebook_sizes(self) -> tuple[int, ...]:
        return self.input_audio_code_spec.primary_codebook_sizes

    @cached_property
    def semantic_codec(self) -> SemanticCodec:
        artifact = self.config.audio_output.acoustic_generator_artifact
        if artifact is None:
            raise RuntimeError(
                "semantic-only waveform decoding requires "
                "runtime.audio_output.acoustic_generator_artifact; "
                "use audio_sequence_layout=flattened for token-only generation."
            )
        if self.acoustic_generator_artifact_sha256 is None:
            raise RuntimeError("acoustic generator artifact identity was not resolved.")
        from semantic_acoustic_generator.runtime import GeneratorRuntime
        from semantic_acoustic_generator.runtime.artifact import load_artifact

        backend = self.output_audio_detokenizer
        if backend is None:
            raise RuntimeError(
                "semantic-only waveform decoding requires runtime.audio_output.detokenizer."
            )
        support = load_artifact(
            Path(artifact).expanduser(),
            device=self.config.device,
        )
        codec = acoustic_codec(backend)
        runtime = GeneratorRuntime(
            support,
            cast("SemanticAcousticCodec", cast(object, codec)),
        )
        return cast(SemanticCodec, cast(object, runtime))

    @cached_property
    def audio_tokenizer(self) -> AudioTokenizer:
        return _audio_tokens(
            name=self.output_audio_tokenizer_name,
            view=self.output_audio_view,
            spec=self.output_audio_code_spec,
            bpe=self.config.audio_output.bpe,
            flattened=self.audio_sequence_layout is AudioSequenceLayout.FLATTENED,
        )

    @cached_property
    def output_audio_tokenizer(self) -> AudioTokenizer:
        return self.audio_tokenizer

    @cached_property
    def output_audio_token_spec(self) -> AudioTokenSpec:
        return AudioTokenSpec.create(
            codec_name=self.output_codec_name,
            sequence_layout=self.audio_sequence_layout.value,
            tokenizer=self.output_audio_tokenizer,
        )

    @cached_property
    def output_audio_token_registry(self) -> AudioTokenRegistry:
        spec = self.output_audio_token_spec
        return AudioTokenRegistry(
            specs=(spec,),
            default_schema_id=spec.schema_id,
        )

    @cached_property
    def input_audio_tokenizer(self) -> AudioTokenizer:
        config = self.config.audio_input
        if config is None or self.input_audio_token_space_shared:
            if (
                config is not None
                and config.vocab_size is not None
                and config.vocab_size != self.audio_tokenizer.vocab_size
            ):
                raise ValueError(
                    "shared runtime.audio_input vocab_size does not match the "
                    "output audio token vocabulary."
                )
            return self.audio_tokenizer
        tokenizer = _audio_tokens(
            name=self.input_audio_tokenizer_name,
            view=self.input_audio_view,
            spec=self.input_audio_code_spec,
            bpe=config.bpe,
            flattened=(
                config.bpe is None
                and self.audio_sequence_layout is AudioSequenceLayout.FLATTENED
                and len(self.input_audio_code_spec.frame_codebook_sizes) > 1
            ),
        )
        if config.vocab_size is not None and tokenizer.vocab_size != config.vocab_size:
            raise ValueError("runtime.audio_input token vocabulary does not match vocab_size.")
        return tokenizer

    @cached_property
    def input_audio_token_spec(self) -> AudioTokenSpec:
        if self.input_audio_token_space_shared:
            return self.output_audio_token_spec
        return AudioTokenSpec.create(
            codec_name=self.input_codec_name,
            sequence_layout=self.audio_sequence_layout.value,
            tokenizer=self.input_audio_tokenizer,
        )

    @cached_property
    def input_audio_token_registry(self) -> AudioTokenRegistry:
        if self.input_audio_token_space_shared:
            return self.output_audio_token_registry
        spec = self.input_audio_token_spec
        return AudioTokenRegistry(
            specs=(spec,),
            default_schema_id=spec.schema_id,
        )

    @cached_property
    def layout(self) -> Layout:
        text_vocab_size = self.lexical_text_vocab_size + len(ControlToken)
        audio_vocab_size = self.audio_tokenizer.vocab_size + 4
        if self.input_audio_decoupled:
            input_audio_vocab_size = self.input_audio_tokenizer.vocab_size + 3
            input_end = text_vocab_size + input_audio_vocab_size
            return Layout(
                text=(0, text_vocab_size),
                audio_input=(text_vocab_size, input_end),
                audio=(input_end, input_end + audio_vocab_size),
            )
        return Layout(
            text=(0, text_vocab_size),
            audio=(text_vocab_size, text_vocab_size + audio_vocab_size),
        )

    @cached_property
    def lexical_text_vocab_size(self) -> int:
        return text_tokenizer_vocab_size(self.text_tokenizer)

    @cached_property
    def control_token_ids(self) -> tuple[int, ...]:
        start, _ = self.layout.blocks[Modality.TEXT.value]
        first = start + self.lexical_text_vocab_size
        return tuple(range(first, first + len(ControlToken)))

    def control_token_id(self, token: ControlToken) -> int:
        if not isinstance(token, ControlToken):
            raise TypeError("control token lookup requires a ControlToken.")
        return self.control_token_ids[list(ControlToken).index(token)]

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
        start, _ = self.layout.blocks[Modality.AUDIO.value]
        return start + self.audio_tokenizer.vocab_size

    @property
    def eoa_token_id(self) -> int:
        return self.boa_token_id + 1

    @property
    def mask_token_id(self) -> int:
        return self.boa_token_id + 2

    @property
    def audio_schema_token_id(self) -> int:
        return self.boa_token_id + 3

    @property
    def output_audio_schema_token_id(self) -> int:
        return self.audio_schema_token_id

    @property
    def output_audio_schema_id(self) -> str:
        return self.output_audio_token_spec.schema_id

    @property
    def output_audio_block_name(self) -> str:
        return Modality.AUDIO.value

    @property
    def output_boa_token_id(self) -> int:
        return self.boa_token_id

    @property
    def output_eoa_token_id(self) -> int:
        return self.eoa_token_id

    @property
    def output_mask_token_id(self) -> int:
        return self.mask_token_id

    @property
    def input_audio_block_name(self) -> str:
        return "audio_input" if self.input_audio_decoupled else Modality.AUDIO.value

    @property
    def input_boa_token_id(self) -> int:
        if not self.input_audio_decoupled:
            return self.boa_token_id
        start, _ = self.layout.blocks[self.input_audio_block_name]
        return start + self.input_audio_tokenizer.vocab_size

    @property
    def input_eoa_token_id(self) -> int:
        if not self.input_audio_decoupled:
            return self.eoa_token_id
        return self.input_boa_token_id + 1

    @property
    def input_audio_schema_token_id(self) -> int:
        if not self.input_audio_decoupled:
            return self.audio_schema_token_id
        return self.input_boa_token_id + 2

    @property
    def input_audio_schema_id(self) -> str:
        return self.input_audio_token_spec.schema_id

    @property
    def audio_head_range(self) -> tuple[int, int]:
        return self.layout.blocks[Modality.AUDIO.value]

    @property
    def output_audio_head_range(self) -> tuple[int, int]:
        return self.audio_head_range

    @property
    def codec_audio_range(self) -> tuple[int, int]:
        start, _ = self.audio_head_range
        return start, self.boa_token_id

    @property
    def output_codec_audio_range(self) -> tuple[int, int]:
        return self.codec_audio_range

    @property
    def input_codec_audio_range(self) -> tuple[int, int]:
        if not self.input_audio_decoupled:
            return self.codec_audio_range
        start, _ = self.layout.blocks[self.input_audio_block_name]
        return start, self.input_boa_token_id

    @cached_property
    def audio_generation_allowed_ids(self) -> tuple[int, ...]:
        start, end = self.codec_audio_range
        return (
            self.boa_token_id,
            self.audio_schema_token_id,
            *range(start, end),
            self.eoa_token_id,
        )

    @cached_property
    def output_audio_generation_allowed_ids(self) -> tuple[int, ...]:
        return self.audio_generation_allowed_ids

    @cached_property
    def text_generation_allowed_ids(self) -> tuple[int, ...]:
        start, _ = self.layout.blocks[Modality.TEXT.value]
        end = start + self.lexical_text_vocab_size
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


def runtime_for_sequence_layout(
    config: Config,
    layout: AudioSequenceLayout,
) -> Runtime:
    return Runtime(config, audio_sequence_layout=layout)


__all__ = ["Runtime", "runtime_for_sequence_layout"]
