from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from torch import Tensor


class AudioTokenizer(Protocol):
    embedding_initialization: str

    @property
    def vocab_size(self) -> int: ...

    def encode(
        self,
        frames: Sequence[Sequence[int]] | Tensor,
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


__all__ = ["AudioTokenizer"]
