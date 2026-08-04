from __future__ import annotations

from functools import cached_property
from typing import Protocol

from anydataset.types import AudioView, Modality
from anytrain.module.idspace import Layout

from .config import AudioSequenceLayout
from .audio_tokenizer.contract import AudioTokenizer
from .backbone import BackboneAdapter
from .backbone.contract import Backbone, TextTokenizer
from .codec_contract import (
    CodecBackend,
    SemanticCodec,
)


class DataRuntime(Protocol):
    @property
    def codec_name(self) -> str: ...

    @property
    def audio_view(self) -> AudioView: ...

    @property
    def codec_frame_rate(self) -> float: ...

    @property
    def audio_sequence_layout(self) -> AudioSequenceLayout: ...

    @property
    def acoustic_generator_artifact(self) -> str | None: ...

    @cached_property
    def text_tokenizer(self) -> TextTokenizer: ...

    @cached_property
    def audio_tokenizer(self) -> AudioTokenizer: ...

    @cached_property
    def layout(self) -> Layout: ...

    @cached_property
    def pad_token_id(self) -> int: ...

    @cached_property
    def eos_token_id(self) -> int: ...

    @property
    def boa_token_id(self) -> int: ...

    @property
    def eoa_token_id(self) -> int: ...

    @property
    def mask_token_id(self) -> int: ...


class GenerationRuntime(DataRuntime, Protocol):
    @cached_property
    def codec(self) -> CodecBackend: ...

    @cached_property
    def semantic_codec(self) -> SemanticCodec: ...

    @property
    def acoustic_side_channel(self) -> bool: ...

    @property
    def structured_full_sequence(self) -> bool: ...

    @cached_property
    def bos_token_id(self) -> int: ...

    @property
    def codec_audio_range(self) -> tuple[int, int]: ...

    @property
    def audio_head_range(self) -> tuple[int, int]: ...

    @cached_property
    def audio_generation_allowed_ids(self) -> tuple[int, ...]: ...

    def generation_allowed_ids(self, modality: Modality) -> tuple[int, ...]: ...

    def is_codec_audio_id(self, token_id: int) -> bool: ...


class TokenModelRuntime(GenerationRuntime, Protocol):
    @cached_property
    def acoustic_generator_artifact_sha256(self) -> str | None: ...

    @property
    def semantic_codebook_sizes(self) -> tuple[int, ...]: ...

    @cached_property
    def backbone_adapter(self) -> BackboneAdapter: ...

    @property
    def backbone_trust_remote_code(self) -> bool: ...

    @property
    def backbone_chat_template(self) -> str | None: ...

    @property
    def backbone_readout(self) -> str: ...

    @property
    def backbone_supports_cache_position(self) -> bool: ...

    @property
    def gradient_checkpointing(self) -> bool: ...

    @cached_property
    def backbone(self) -> Backbone: ...
