from __future__ import annotations

from math import prod
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FsqAffineEmbedding(nn.Module):
    """Rank-1 affine embedding for packed FSQ product indices.

    Each residual stage keeps FSQ intrinsic dimension 1. A product id unpacks to
    per-dimension levels ``q_j``; the embedding is
    ``sum_j (b_j + qtilde_j w_j)`` with ``qtilde`` on the FSQ ``[-1, 1]`` grid.
    Marker / special rows stay free embeddings. ``.weight`` materializes the full
    tied table for vocabulary logits.
    """

    def __init__(
        self,
        *,
        codebook_sizes: tuple[int, ...],
        fsq_levels: tuple[tuple[int, ...], ...],
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("FSQ embedding dimension must be positive.")
        if len(codebook_sizes) != len(fsq_levels):
            raise ValueError("fsq_levels must align with codebook_sizes.")
        if not codebook_sizes:
            raise ValueError("FSQ embedding requires at least one codebook.")
        for size, stage in zip(codebook_sizes, fsq_levels):
            if size < 1:
                raise ValueError("codebook sizes must be positive.")
            if not stage or any(level < 2 for level in stage):
                raise ValueError("FSQ levels must be at least 2.")
            if prod(stage) != size:
                raise ValueError(
                    f"FSQ levels {stage} must multiply to codebook size {size}."
                )

        code_vocab = sum(codebook_sizes)
        free = num_embeddings - code_vocab
        if free < 0:
            raise ValueError(
                "num_embeddings must cover codebook rows plus free marker rows."
            )

        self.codebook_sizes = codebook_sizes
        self.fsq_levels = fsq_levels
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim

        self.biases = nn.ParameterList()
        self.slopes = nn.ParameterList()
        scalars: list[Tensor] = []
        for stage, levels in enumerate(fsq_levels):
            width = len(levels)
            bias = nn.Parameter(torch.empty(width, embedding_dim))
            slope = nn.Parameter(torch.empty(width, embedding_dim))
            nn.init.normal_(bias, std=embedding_dim**-0.5)
            nn.init.normal_(slope, std=embedding_dim**-0.5)
            self.biases.append(bias)
            self.slopes.append(slope)
            level_ids = _product_level_indices(codebook_sizes[stage], levels)
            scalars.append(_level_scalars(level_ids, levels))
        for stage, values in enumerate(scalars):
            self.register_buffer(f"scalars_{stage}", values, persistent=False)

        self.free = nn.Embedding(free, embedding_dim)

    @property
    def num_embeddings(self) -> int:
        return self._num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def weight(self) -> Tensor:
        return self._materialize()

    def forward(self, input_ids: Tensor) -> Tensor:
        return F.embedding(input_ids, self.weight)

    def code_embedding(self, product_ids: Tensor, *, stage: int) -> Tensor:
        """Embed raw product indices for one residual stage."""
        if stage < 0 or stage >= len(self.codebook_sizes):
            raise ValueError(f"FSQ stage {stage} is out of range.")
        size = self.codebook_sizes[stage]
        if bool((product_ids < 0).any()) or bool((product_ids >= size).any()):
            raise ValueError("product id is outside the FSQ codebook.")
        scalars = self._scalars(stage).index_select(0, product_ids.reshape(-1))
        values = self._affine(stage, scalars)
        return values.view(*product_ids.shape, self._embedding_dim)

    def _materialize(self) -> Tensor:
        rows = [
            self._affine(stage, self._scalars(stage))
            for stage in range(len(self.codebook_sizes))
        ]
        code = torch.cat(rows, dim=0)
        if self.free.num_embeddings == 0:
            return code
        return torch.cat([code, self.free.weight], dim=0)

    def _affine(self, stage: int, scalars: Tensor) -> Tensor:
        # scalars: [N, D], bias/slope: [D, d] -> sum_j (b_j + q_j w_j)
        return scalars @ self.slopes[stage] + self.biases[stage].sum(dim=0)

    def _scalars(self, stage: int) -> Tensor:
        return cast(Tensor, getattr(self, f"scalars_{stage}"))


def _product_level_indices(vocab: int, levels: tuple[int, ...]) -> Tensor:
    level_tensor = torch.tensor(levels, dtype=torch.int64)
    basis = torch.ones(len(levels), dtype=torch.int64)
    if len(levels) > 1:
        basis[1:] = torch.cumprod(level_tensor[:-1], dim=0)
    indices = torch.arange(vocab, dtype=torch.int64)
    return (indices[:, None] // basis) % level_tensor


def _level_scalars(level_indices: Tensor, levels: tuple[int, ...]) -> Tensor:
    level_tensor = torch.tensor(levels, dtype=torch.float32)
    half = 2.0 / (level_tensor - 1.0)
    return level_indices.to(dtype=torch.float32) * half - 1.0
