from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from anytrain.idspace import IdSpaceEmbedding, Modality
from torch import Tensor, nn

from .types import AudioBoundary


def audio_special_embeddings(
    hidden_size: int,
    *,
    like: Tensor,
    std: float,
) -> nn.ParameterDict:
    embeddings = nn.ParameterDict()
    for name in (AudioBoundary.BOA.value, AudioBoundary.EOA.value):
        parameter = nn.Parameter(torch.empty(hidden_size, device=like.device, dtype=like.dtype))
        nn.init.normal_(parameter, std=std)
        embeddings[name] = parameter
    return embeddings


class AudioLMHead(Protocol):
    @property
    def vocab_size(self) -> int: ...

    @property
    def global_ids(self) -> tuple[int, ...]: ...

    def __call__(self, hidden_states: Tensor) -> Tensor: ...

    def to_head_ids(self, ids: Sequence[int] | Tensor) -> list[int] | Tensor: ...

    def to_global_ids(self, ids: Sequence[int] | Tensor) -> list[int] | Tensor: ...


def audio_lm_head(
    embedding: IdSpaceEmbedding,
    *,
    special_tokens: tuple[str, ...],
) -> AudioLMHead:
    return embedding.head_view(
        special_tokens=special_tokens,
        modalities=(Modality.AUDIO,),
    )


def text_embedding(model: nn.Module) -> nn.Embedding | IdSpaceEmbedding:
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embed = get_input_embeddings()
        if isinstance(embed, nn.Embedding | IdSpaceEmbedding):
            return embed

    embed = getattr(model, "embed_tokens", None)
    if isinstance(embed, nn.Embedding | IdSpaceEmbedding):
        return embed

    child = getattr(model, "model", None)
    if isinstance(child, nn.Module):
        embed = getattr(child, "embed_tokens", None)
        if isinstance(embed, nn.Embedding | IdSpaceEmbedding):
            return embed

    raise AttributeError("qwen3 model must expose embed_tokens.")


def hidden_size(module: nn.Module | None) -> int | None:
    if module is None:
        return None
    config = getattr(module, "config", None)
    value = getattr(config, "hidden_size", None)
    if isinstance(value, int):
        return value
    return None


def set_text_embedding(model: nn.Module, embedding: IdSpaceEmbedding) -> None:
    set_input_embeddings = getattr(model, "set_input_embeddings", None)
    if callable(set_input_embeddings):
        set_input_embeddings(embedding)
        return

    if hasattr(model, "embed_tokens"):
        model.embed_tokens = embedding
        return

    child = getattr(model, "model", None)
    if isinstance(child, nn.Module) and hasattr(child, "embed_tokens"):
        child.embed_tokens = embedding
        return

    raise AttributeError("qwen3 model must expose embed_tokens.")
