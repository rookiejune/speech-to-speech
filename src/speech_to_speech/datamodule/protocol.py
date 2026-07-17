from functools import cached_property
from typing import Protocol

from anydataset.types import AudioView
from anytrain.idspace import Layout

from ..runtime.types import AudioTokenizer, TextTokenizer


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
