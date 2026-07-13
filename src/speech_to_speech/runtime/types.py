from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from torch import Tensor, nn


class Codec(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_rate(self) -> float: ...

    @property
    def acoustic_feature_dim(self) -> int: ...

    @property
    def semantic_codebook(self) -> Tensor: ...

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]: ...

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor: ...

    def decode(self, codes: Tensor) -> Tensor: ...

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor: ...

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor: ...


class AudioTokenizer(Protocol):
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


class TextTokenizer(Protocol):
    special_tokens_map: Mapping[str, str | Sequence[str]]

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


class BackboneConfig(Protocol):
    hidden_size: int


class BackboneOutput(Protocol):
    last_hidden_state: Tensor
    past_key_values: Any
    hidden_states: tuple[Tensor, ...] | None
    attentions: tuple[Tensor, ...] | None


class BackboneBody(Protocol):
    def __call__(self, **kwargs: Any) -> BackboneOutput: ...


class Backbone(Protocol):
    @property
    def config(self) -> BackboneConfig: ...

    def get_input_embeddings(self) -> nn.Embedding: ...

    def get_output_embeddings(self) -> nn.Linear: ...

    @property
    def base_model(self) -> BackboneBody: ...

    def __call__(self, **kwargs: Any) -> Any: ...
