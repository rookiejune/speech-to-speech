from __future__ import annotations

from math import prod
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .._helper import register


class FsqAffineEmbedding(nn.Module):
    """Rank-1 affine embedding for packed FSQ product indices.

    Each residual stage keeps FSQ intrinsic dimension 1. A product id unpacks to
    per-dimension levels ``q_j``; the embedding is
    ``b + sum_j qtilde_j w_j`` with ``qtilde`` on the FSQ ``[-1, 1]`` grid.
    Marker / special rows stay free embeddings. Normal lookup and logits remain
    factorized; ``.weight`` is a compatibility path that materializes the table.
    """

    topology: Tensor

    def __init__(
        self,
        *,
        codebook_sizes: tuple[int, ...],
        fsq_levels: tuple[tuple[int, ...], ...],
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        free = _validate_embedding_shape(
            codebook_sizes,
            fsq_levels,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )

        self.codebook_sizes = codebook_sizes
        self.fsq_levels = fsq_levels
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        register(self, "topology", _topology_state(codebook_sizes, fsq_levels))

        self.offsets = nn.ParameterList()
        self.slopes = nn.ParameterList()
        scalars: list[Tensor] = []
        for stage, levels in enumerate(fsq_levels):
            width = len(levels)
            offset = nn.Parameter(torch.empty(embedding_dim))
            slope = nn.Parameter(torch.empty(width, embedding_dim))
            nn.init.normal_(offset, std=embedding_dim**-0.5)
            nn.init.normal_(slope, std=embedding_dim**-0.5)
            self.offsets.append(offset)
            self.slopes.append(slope)
            level_ids = _product_level_indices(codebook_sizes[stage], levels)
            scalars.append(_level_scalars(level_ids, levels))
        for stage, values in enumerate(scalars):
            self.register_buffer(f"scalars_{stage}", values, persistent=False)

        self.free = nn.Embedding(free, embedding_dim)
        nn.init.normal_(self.free.weight, std=embedding_dim**-0.5)

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
        return self.rows(input_ids)

    def rows(self, input_ids: Tensor) -> Tensor:
        """Return vocabulary rows without materializing the complete table."""
        self._validate_input_ids(input_ids)
        flat_ids = input_ids.reshape(-1)
        output = self.slopes[0].new_empty(flat_ids.numel(), self._embedding_dim)

        start = 0
        for stage, size in enumerate(self.codebook_sizes):
            mask = flat_ids.ge(start) & flat_ids.lt(start + size)
            positions = mask.nonzero(as_tuple=False).flatten()
            product_ids = flat_ids.index_select(0, positions) - start
            output = output.index_copy(
                0,
                positions,
                self._code_rows(product_ids, stage=stage),
            )
            start += size

        if self.free.num_embeddings > 0:
            mask = flat_ids.ge(start)
            positions = mask.nonzero(as_tuple=False).flatten()
            free_ids = flat_ids.index_select(0, positions) - start
            output = output.index_copy(0, positions, self.free(free_ids))
        return output.view(*input_ids.shape, self._embedding_dim)

    def logits(self, hidden: Tensor, local_ids: Tensor | None = None) -> Tensor:
        """Compute tied logits directly in the unadapted embedding space."""
        if local_ids is not None:
            if local_ids.dim() != 1:
                raise ValueError("local_ids must be a one-dimensional tensor.")
            weight = self.rows(local_ids)
            return F.linear(hidden.to(dtype=weight.dtype), weight)

        values = hidden.to(dtype=self.slopes[0].dtype)
        logits: list[Tensor] = []
        for stage in range(len(self.codebook_sizes)):
            factors = F.linear(values, self.slopes[stage])
            offset = F.linear(values, self.offsets[stage].unsqueeze(0))
            logits.append(factors @ self._scalars(stage).T + offset)
        if self.free.num_embeddings > 0:
            logits.append(F.linear(values, self.free.weight))
        return torch.cat(logits, dim=-1)

    def code_embedding(self, product_ids: Tensor, *, stage: int) -> Tensor:
        """Embed raw product indices for one residual stage."""
        if stage < 0 or stage >= len(self.codebook_sizes):
            raise ValueError(f"FSQ stage {stage} is out of range.")
        size = self.codebook_sizes[stage]
        if bool((product_ids < 0).any()) or bool((product_ids >= size).any()):
            raise ValueError("product id is outside the FSQ codebook.")
        return self._code_rows(product_ids, stage=stage)

    def _code_rows(self, product_ids: Tensor, *, stage: int) -> Tensor:
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
        return scalars @ self.slopes[stage] + self.offsets[stage]

    def _scalars(self, stage: int) -> Tensor:
        return cast(Tensor, getattr(self, f"scalars_{stage}"))

    def _validate_input_ids(self, input_ids: Tensor) -> None:
        # A one-column expanded view preserves nn.Embedding's index contract
        # without allocating a vocabulary-sized parameter table.
        validation_weight = self.slopes[0].detach()[:1, :1].expand(
            self._num_embeddings, 1
        )
        F.embedding(input_ids, validation_weight)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        key = f"{prefix}topology"
        checkpoint = state_dict.get(key)
        expected = _topology_state(self.codebook_sizes, self.fsq_levels)
        if checkpoint is not None and not (
            checkpoint.dtype is torch.int64
            and checkpoint.shape == expected.shape
            and torch.equal(checkpoint.detach().cpu(), expected)
        ):
            error_msgs.append(
                f'FSQ topology at "{key}" does not match the configured '
                f"codebook_sizes={self.codebook_sizes!r}, "
                f"fsq_levels={self.fsq_levels!r}."
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def _topology_state(
    codebook_sizes: tuple[int, ...],
    fsq_levels: tuple[tuple[int, ...], ...],
) -> Tensor:
    values = [len(codebook_sizes)]
    for size, levels in zip(codebook_sizes, fsq_levels):
        values.extend((size, len(levels), *levels))
    return torch.tensor(values, dtype=torch.int64)


def _validate_embedding_shape(
    codebook_sizes: tuple[int, ...],
    fsq_levels: tuple[tuple[int, ...], ...],
    *,
    num_embeddings: int,
    embedding_dim: int,
) -> int:
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

    free = num_embeddings - sum(codebook_sizes)
    if free < 0:
        raise ValueError(
            "num_embeddings must cover codebook rows plus free marker rows."
        )
    return free


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
