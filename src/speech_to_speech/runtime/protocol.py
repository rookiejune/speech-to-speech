from __future__ import annotations

from functools import cached_property
from typing import Protocol

from anydataset.types import AudioView, Modality
from anytrain.idspace import Layout

from .types import AudioTokenizer, Backbone, Codec, TextTokenizer


class DataRuntime(Protocol):
    @property
    def codec_name(self) -> str: ...

    @property
    def audio_view(self) -> AudioView: ...

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


class GenerationRuntime(DataRuntime, Protocol):
    @cached_property
    def codec(self) -> Codec: ...

    @cached_property
    def bos_token_id(self) -> int: ...

    @property
    def codec_audio_range(self) -> tuple[int, int]: ...

    @cached_property
    def audio_generation_allowed_ids(self) -> tuple[int, ...]: ...

    def generation_allowed_ids(self, modality: Modality) -> tuple[int, ...]: ...

    def is_codec_audio_id(self, token_id: int) -> bool: ...


class TokenModelRuntime(GenerationRuntime, Protocol):
    @cached_property
    def backbone(self) -> Backbone: ...
