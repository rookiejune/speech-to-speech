from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, TypedDict

from anydataset.types import AudioView
from anytrain.module.idspace import Layout
from torch import Tensor

from ..audio import AudioStream
from ..runtime import AudioSequenceLayout
from ..runtime.audio_tokenizer.contract import AudioTokenizer
from ..runtime.audio_schema import AudioTokenRegistry, AudioTokenSpec
from ..runtime.backbone.contract import TextTokenizer
from ..runtime.protocol import DataRuntime
from ..runtime.codec_contract import CodecBackend
from ..task import ControlToken


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
    def lexical_text_vocab_size(self) -> int: ...

    @cached_property
    def control_token_ids(self) -> tuple[int, ...]: ...

    def control_token_id(self, token: ControlToken) -> int: ...

    @cached_property
    def pad_token_id(self) -> int: ...

    @cached_property
    def bos_token_id(self) -> int: ...

    @cached_property
    def eos_token_id(self) -> int: ...


class DatasetRuntime(DataRuntime, Protocol):
    @cached_property
    def codec(self) -> CodecBackend: ...

    @cached_property
    def input_codec(self) -> CodecBackend: ...


@dataclass(frozen=True)
class TextRuntimeSnapshot:
    """Pickleable worker view for text-only dataloaders."""

    text_tokenizer: TextTokenizer
    layout_blocks: tuple[tuple[str, tuple[int, int]], ...]
    lexical_text_vocab_size: int
    control_token_ids: tuple[int, ...]
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int

    @classmethod
    def from_runtime(cls, runtime: TextRuntime) -> TextRuntimeSnapshot:
        return cls(
            text_tokenizer=runtime.text_tokenizer,
            layout_blocks=tuple(runtime.layout.blocks.items()),
            lexical_text_vocab_size=runtime.lexical_text_vocab_size,
            control_token_ids=runtime.control_token_ids,
            pad_token_id=runtime.pad_token_id,
            bos_token_id=runtime.bos_token_id,
            eos_token_id=runtime.eos_token_id,
        )

    @cached_property
    def layout(self) -> Layout:
        return Layout(**dict(self.layout_blocks))

    def control_token_id(self, token: ControlToken) -> int:
        return _control_token_id(self.control_token_ids, token)


@dataclass(frozen=True)
class DataRuntimeSnapshot:
    """Pickleable worker view of the data-only runtime capabilities."""

    input_audio_decoupled: bool
    input_codec_name: str
    input_audio_view: AudioView
    input_audio_stream_views: tuple[tuple[AudioStream, AudioView], ...]
    input_codec_frame_rate: float
    codec_name: str
    audio_view: AudioView
    codec_frame_rate: float
    audio_sequence_layout: AudioSequenceLayout
    acoustic_generator_artifact: str | None
    text_tokenizer: TextTokenizer
    input_audio_tokenizer: AudioTokenizer
    audio_tokenizer: AudioTokenizer
    input_audio_token_spec: AudioTokenSpec
    audio_token_spec: AudioTokenSpec
    layout_blocks: tuple[tuple[str, tuple[int, int]], ...]
    lexical_text_vocab_size: int
    control_token_ids: tuple[int, ...]
    pad_token_id: int
    eos_token_id: int
    boa_token_id: int
    eoa_token_id: int
    mask_token_id: int
    audio_schema_token_id: int
    input_audio_block_name: str
    input_boa_token_id: int
    input_eoa_token_id: int
    input_audio_schema_token_id: int
    input_codec_audio_range: tuple[int, int]

    @property
    def output_codec_name(self) -> str:
        return self.codec_name

    @property
    def output_audio_view(self) -> AudioView:
        return self.audio_view

    @property
    def output_codec_frame_rate(self) -> float:
        return self.codec_frame_rate

    @property
    def output_audio_tokenizer(self) -> AudioTokenizer:
        return self.audio_tokenizer

    @property
    def output_audio_token_spec(self) -> AudioTokenSpec:
        return self.audio_token_spec

    @cached_property
    def output_audio_token_registry(self) -> AudioTokenRegistry:
        spec = self.output_audio_token_spec
        return AudioTokenRegistry((spec,), spec.schema_id)

    @cached_property
    def input_audio_token_registry(self) -> AudioTokenRegistry:
        spec = self.input_audio_token_spec
        return AudioTokenRegistry((spec,), spec.schema_id)

    @property
    def output_audio_schema_id(self) -> str:
        return self.output_audio_token_spec.schema_id

    @property
    def input_audio_schema_id(self) -> str:
        return self.input_audio_token_spec.schema_id

    @property
    def output_audio_block_name(self) -> str:
        return "audio"

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
    def output_audio_schema_token_id(self) -> int:
        return self.audio_schema_token_id

    @classmethod
    def from_runtime(cls, runtime: DataRuntime) -> DataRuntimeSnapshot:
        return cls(
            input_audio_decoupled=runtime.input_audio_decoupled,
            input_codec_name=runtime.input_codec_name,
            input_audio_view=runtime.input_audio_view,
            input_audio_stream_views=getattr(
                runtime,
                "input_audio_stream_views",
                ((AudioStream.SEMANTIC, runtime.input_audio_view),),
            ),
            input_codec_frame_rate=runtime.input_codec_frame_rate,
            codec_name=runtime.codec_name,
            audio_view=runtime.audio_view,
            codec_frame_rate=runtime.codec_frame_rate,
            audio_sequence_layout=runtime.audio_sequence_layout,
            acoustic_generator_artifact=runtime.acoustic_generator_artifact,
            text_tokenizer=runtime.text_tokenizer,
            input_audio_tokenizer=runtime.input_audio_tokenizer,
            audio_tokenizer=runtime.audio_tokenizer,
            input_audio_token_spec=runtime.input_audio_token_spec,
            audio_token_spec=runtime.output_audio_token_spec,
            layout_blocks=tuple(runtime.layout.blocks.items()),
            lexical_text_vocab_size=runtime.lexical_text_vocab_size,
            control_token_ids=runtime.control_token_ids,
            pad_token_id=runtime.pad_token_id,
            eos_token_id=runtime.eos_token_id,
            boa_token_id=runtime.boa_token_id,
            eoa_token_id=runtime.eoa_token_id,
            mask_token_id=runtime.mask_token_id,
            audio_schema_token_id=runtime.audio_schema_token_id,
            input_audio_block_name=runtime.input_audio_block_name,
            input_boa_token_id=runtime.input_boa_token_id,
            input_eoa_token_id=runtime.input_eoa_token_id,
            input_audio_schema_token_id=runtime.input_audio_schema_token_id,
            input_codec_audio_range=runtime.input_codec_audio_range,
        )

    @cached_property
    def layout(self) -> Layout:
        # Layout stores a mappingproxy and cannot cross spawn workers directly.
        return Layout(**dict(self.layout_blocks))

    def control_token_id(self, token: ControlToken) -> int:
        return _control_token_id(self.control_token_ids, token)


def _control_token_id(ids: tuple[int, ...], token: ControlToken) -> int:
    if not isinstance(token, ControlToken):
        raise TypeError("control token lookup requires a ControlToken.")
    if len(ids) != len(ControlToken):
        raise ValueError("runtime control token ids do not match the control vocabulary.")
    return ids[list(ControlToken).index(token)]


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
