from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

from anydataset.types import AudioView
from anytrain.idspace import Layout

from ..runtime.protocol import DataRuntime
from ..runtime.types import AudioTokenizer, Codec, TextTokenizer


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
    def codec(self) -> Codec: ...


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

    codec_name: str
    audio_view: AudioView
    text_tokenizer: TextTokenizer
    audio_tokenizer: AudioTokenizer
    layout_blocks: tuple[tuple[str, tuple[int, int]], ...]
    pad_token_id: int
    eos_token_id: int
    boa_token_id: int
    eoa_token_id: int

    @classmethod
    def from_runtime(cls, runtime: DataRuntime) -> DataRuntimeSnapshot:
        return cls(
            codec_name=runtime.codec_name,
            audio_view=runtime.audio_view,
            text_tokenizer=runtime.text_tokenizer,
            audio_tokenizer=runtime.audio_tokenizer,
            layout_blocks=tuple(runtime.layout.blocks.items()),
            pad_token_id=runtime.pad_token_id,
            eos_token_id=runtime.eos_token_id,
            boa_token_id=runtime.boa_token_id,
            eoa_token_id=runtime.eoa_token_id,
        )

    @cached_property
    def layout(self) -> Layout:
        # Layout stores a mappingproxy and cannot cross spawn workers directly.
        return Layout(**dict(self.layout_blocks))


__all__ = [
    "DataRuntime",
    "DataRuntimeSnapshot",
    "DatasetRuntime",
    "TextRuntime",
    "TextRuntimeSnapshot",
]
