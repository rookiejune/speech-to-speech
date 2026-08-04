from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from torch import Tensor


class AudioTokenizer(Protocol):
    embedding_initialization: str

    @property
    def vocab_size(self) -> int: ...

    def encode(
        self, frames: Sequence[Sequence[int]] | Tensor
    ) -> list[int] | Tensor: ...

    def decode(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[tuple[int, ...]] | Tensor: ...

    def frame_spans(
        self,
        token_ids: Sequence[int] | Tensor,
    ) -> list[int] | Tensor: ...

    def contract_state(self) -> Mapping[str, object]: ...


class TextTokenizer(Protocol):
    special_tokens_map: Mapping[str, str | Sequence[str]]
    pad_token_id: int | None
    eos_token_id: int | None
    bos_token_id: int | None

    def __len__(self) -> int: ...

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str: ...

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
        enable_thinking: bool = ...,
        return_dict: bool = ...,
    ) -> str | list[int]: ...

__all__ = ["AudioTokenizer", "TextTokenizer"]
