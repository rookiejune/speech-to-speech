from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, overload

from torch import Tensor, nn


class Codec(Protocol):
    @property
    def acoustic_feature_dim(self) -> int: ...

    @property
    def semantic_codebook(self) -> Tensor: ...

    def encode(self, audio: Tensor, sample_rate: int) -> tuple[Tensor, Tensor]: ...

    def decode(self, semantic_codes: Tensor, acoustic_codes: Tensor) -> Tensor: ...

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor: ...

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor: ...


class AudioTokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, ids: Sequence[Sequence[int]] | Tensor) -> list[int] | Tensor: ...

    def expand(
        self,
        ids: Sequence[int] | Tensor,
        *,
        strict: bool | None = None,
    ) -> list[tuple[int, ...]] | Tensor:
        """Expand audio BPE tokens to ``[frames, semantic_codebooks]`` units."""
        ...

    def expand_with_counts(
        self,
        ids: Sequence[int] | Tensor,
        *,
        strict: bool | None = None,
    ) -> tuple[list[tuple[int, ...]] | Tensor, list[int] | Tensor]: ...

    def repeat_interleave(
        self,
        x: Tensor,
        spans: Tensor,
        mask: Tensor | None = None,
        *,
        dim: int = -2,
        strict: bool | None = None,
    ) -> tuple[Tensor, Tensor]: ...


class TextTokenizer(Protocol):
    special_tokens_map: Mapping[str, str | Sequence[str]]

    def __len__(self) -> int: ...

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]: ...

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
        enable_thinking: bool = ...,
        return_dict: bool = ...,
    ) -> list[int]: ...


class Backbone(Protocol):
    @property
    def config(self) -> Any: ...

    @property
    def embed_tokens(self) -> nn.Embedding: ...

    def get_output_embeddings(self) -> nn.Module: ...

    def __call__(self, **kwargs: Any) -> Any: ...
