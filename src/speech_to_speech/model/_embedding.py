from __future__ import annotations

import torch
from torch import nn


class EmbeddingView(nn.Module):
    """Expose an owned embedding to HF backbone APIs without re-parenting it."""

    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        object.__setattr__(self, "_embedding", embedding)

    @property
    def weight(self) -> torch.Tensor:
        return self._embedding.weight

    @property
    def num_embeddings(self) -> int:
        return self._embedding.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding.embedding_dim

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding(input_ids)


class CastOutput(nn.Module):
    """Cast adapter outputs to the backbone embedding dtype at the idspace boundary."""

    def __init__(self, module: nn.Module, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.module = module
        self._dtype = dtype

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.module(values).to(dtype=self._dtype)
