"""Lookup-table audio BPE embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def lookup_audio_embedding(
    *,
    vocab_size: int,
    hidden_size: int,
    like: Tensor,
    std: float,
) -> nn.Embedding:
    embedding = nn.Embedding(
        vocab_size,
        hidden_size,
        device=like.device,
        dtype=like.dtype,
    )
    nn.init.normal_(embedding.weight, std=std)
    return embedding


__all__ = ["lookup_audio_embedding"]
