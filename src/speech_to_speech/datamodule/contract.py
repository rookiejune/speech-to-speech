from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, TypedDict

from anydataset.types import AudioView
from anytrain.module.idspace import Layout
from torch import Tensor

from ..runtime import AudioSequenceLayout
from ..runtime.audio_tokenizer.contract import AudioTokenizer
from ..runtime.backbone.contract import TextTokenizer
from ..runtime.protocol import DataRuntime
from ..runtime.codec_contract import CodecBackend


ACOUSTIC_PAD_ID = -1
CTC_PAD_ID = -1


class AcousticTarget(TypedDict):
    semantic_codes: Tensor
    codes: Tensor
    token_positions: Tensor


class CTCTarget(TypedDict):
    """Transcript supervision attached to one audio span."""

    token_positions: Tensor
    text_token_ids: Tensor


@dataclass(frozen=True)
class Labels:
    """Training-only supervision for the response side of a sample."""

    response_ids: Tensor
    token_labels: Tensor
    acoustic_target: AcousticTarget | None = None
    source_ctc: CTCTarget | None = None
    target_ctc: CTCTarget | None = None
    audio_seconds: float = 0.0


class TextRuntime(Protocol):
    @cached_property
    def text_tokenizer(self) -> TextTokenizer: ...

    @cached_property
    def layout(self) -> Layout: ...

    @cached_property
    def pad_token_id(self) -> int: ...

    @cached_property
    def eos_token_id(self) -> int: ...


class DatasetRuntime(DataRuntime, Protocol):
    @cached_property
    def codec(self) -> CodecBackend: ...


@dataclass(frozen=True)
class TextRuntimeSnapshot:
    """Pickleable worker view for text-only dataloaders."""

    text_tokenizer: TextTokenizer
    layout_blocks: tuple[tuple[str, tuple[int, int]], ...]
    pad_token_id: int
    eos_token_id: int

    @classmethod
    def from_runtime(cls, runtime: TextRuntime) -> TextRuntimeSnapshot:
        return cls(
            text_tokenizer=runtime.text_tokenizer,
            layout_blocks=tuple(runtime.layout.blocks.items()),
            pad_token_id=runtime.pad_token_id,
            eos_token_id=runtime.eos_token_id,
        )

    @cached_property
    def layout(self) -> Layout:
        return Layout(**dict(self.layout_blocks))


@dataclass(frozen=True)
class DataRuntimeSnapshot:
    """Pickleable worker view of the data-only runtime capabilities."""

    input_audio_decoupled: bool
    input_codec_name: str
    input_audio_view: AudioView
    input_codec_frame_rate: float
    codec_name: str
    audio_view: AudioView
    codec_frame_rate: float
    audio_sequence_layout: AudioSequenceLayout
    acoustic_generator_artifact: str | None
    text_tokenizer: TextTokenizer
    input_audio_tokenizer: AudioTokenizer
    audio_tokenizer: AudioTokenizer
    layout_blocks: tuple[tuple[str, tuple[int, int]], ...]
    pad_token_id: int
    eos_token_id: int
    boa_token_id: int
    eoa_token_id: int
    mask_token_id: int
    input_audio_block_name: str
    input_boa_token_id: int
    input_eoa_token_id: int
    input_codec_audio_range: tuple[int, int]

    @classmethod
    def from_runtime(cls, runtime: DataRuntime) -> DataRuntimeSnapshot:
        return cls(
            input_audio_decoupled=runtime.input_audio_decoupled,
            input_codec_name=runtime.input_codec_name,
            input_audio_view=runtime.input_audio_view,
            input_codec_frame_rate=runtime.input_codec_frame_rate,
            codec_name=runtime.codec_name,
            audio_view=runtime.audio_view,
            codec_frame_rate=runtime.codec_frame_rate,
            audio_sequence_layout=runtime.audio_sequence_layout,
            acoustic_generator_artifact=runtime.acoustic_generator_artifact,
            text_tokenizer=runtime.text_tokenizer,
            input_audio_tokenizer=runtime.input_audio_tokenizer,
            audio_tokenizer=runtime.audio_tokenizer,
            layout_blocks=tuple(runtime.layout.blocks.items()),
            pad_token_id=runtime.pad_token_id,
            eos_token_id=runtime.eos_token_id,
            boa_token_id=runtime.boa_token_id,
            eoa_token_id=runtime.eoa_token_id,
            mask_token_id=runtime.mask_token_id,
            input_audio_block_name=runtime.input_audio_block_name,
            input_boa_token_id=runtime.input_boa_token_id,
            input_eoa_token_id=runtime.input_eoa_token_id,
            input_codec_audio_range=runtime.input_codec_audio_range,
        )

    @cached_property
    def layout(self) -> Layout:
        # Layout stores a mappingproxy and cannot cross spawn workers directly.
        return Layout(**dict(self.layout_blocks))


__all__ = [
    "ACOUSTIC_PAD_ID",
    "AcousticTarget",
    "CTC_PAD_ID",
    "CTCTarget",
    "DataRuntime",
    "DataRuntimeSnapshot",
    "DatasetRuntime",
    "Labels",
    "TextRuntime",
    "TextRuntimeSnapshot",
]
