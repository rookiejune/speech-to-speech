from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, cast, runtime_checkable

from torch import Tensor, nn
from transformers.cache_utils import Cache


class SemanticCodec(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_rate(self) -> float: ...

    def decode(self, semantic_codes: Tensor) -> Tensor: ...


class Codec(SemanticCodec, Protocol):
    @property
    def semantic_feature_dim(self) -> int: ...

    @property
    def codebook_sizes(self) -> tuple[int, ...]: ...

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor: ...


class CodebookCodec(Codec, Protocol):
    @property
    def semantic_codebook(self) -> Tensor: ...


class AcousticCodec(CodebookCodec, Protocol):
    @property
    def acoustic_feature_dim(self) -> int: ...

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]: ...

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor: ...

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor: ...


@runtime_checkable
class _CodebookCapability(Protocol):
    @property
    def semantic_codebook(self) -> Tensor: ...


@runtime_checkable
class _AcousticCapability(_CodebookCapability, Protocol):
    @property
    def acoustic_feature_dim(self) -> int: ...

    @property
    def acoustic_codebook_sizes(self) -> tuple[int, ...]: ...

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor: ...

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor: ...


def codebook_codec(codec: Codec) -> CodebookCodec:
    if not isinstance(codec, _CodebookCapability):
        raise TypeError("codec-initialized audio embeddings require a semantic codebook.")
    return cast(CodebookCodec, codec)


def acoustic_codec(codec: Codec) -> AcousticCodec:
    if not isinstance(codec, _AcousticCapability):
        raise TypeError("acoustic decoding requires an acoustic codec capability.")
    sizes = tuple(codec.acoustic_codebook_sizes)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("acoustic codec codebook sizes must be positive and non-empty.")
    if codec.acoustic_feature_dim <= 0:
        raise ValueError("acoustic codec feature dimension must be positive.")
    return cast(AcousticCodec, codec)


def supports_acoustic(codec: Codec) -> bool:
    if not isinstance(codec, _AcousticCapability):
        return False
    acoustic_codec(codec)
    return True


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
    past_key_values: Cache | None
    hidden_states: tuple[Tensor, ...] | None
    attentions: tuple[Tensor, ...] | None


class BackboneBody(Protocol):
    def __call__(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None,
        output_hidden_states: bool,
        past_key_values: Cache | None,
        use_cache: bool,
        position_ids: Tensor | None,
        cache_position: Tensor | None,
    ) -> BackboneOutput: ...


class Backbone(Protocol):
    @property
    def config(self) -> BackboneConfig: ...

    def get_input_embeddings(self) -> nn.Embedding: ...

    def get_output_embeddings(self) -> nn.Linear: ...

    @property
    def base_model(self) -> BackboneBody: ...
