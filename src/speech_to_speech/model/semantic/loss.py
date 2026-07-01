"""Semantic LM supervised-position and row-loss helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from ...types.datamodule import CausalLMBatch, IGNORE_INDEX


def loss_positions(batch: CausalLMBatch) -> Tensor:
    if isinstance(batch.logits_to_keep, Tensor):
        positions = batch.logits_to_keep.to(device=batch.labels.device, dtype=torch.long)
        if positions.dim() != 2 or positions.size(-1) != 2:
            raise ValueError("logits_to_keep tensor must have shape (n, 2).")
        return positions
    if batch.logits_to_keep <= 0:
        raise ValueError("logits_to_keep must be positive.")
    mask = batch.labels.ne(IGNORE_INDEX)
    if not bool(mask.any()):
        raise ValueError("labels must contain at least one supervised token.")
    if batch.logits_to_keep >= mask.size(1):
        return mask.nonzero(as_tuple=False)
    keep = mask.cumsum(dim=1) > (mask.sum(dim=1, keepdim=True) - batch.logits_to_keep).clamp_min(0)
    return (mask & keep).nonzero(as_tuple=False)


def loss_weights(batch: CausalLMBatch, positions: Tensor, *, dtype: torch.dtype) -> Tensor:
    if batch.loss_weights is None:
        return torch.ones(positions.size(0), dtype=dtype, device=batch.labels.device)
    weights = batch.loss_weights.to(device=batch.labels.device, dtype=dtype)
    if weights.shape != batch.labels.shape:
        raise ValueError("loss_weights must have the same shape as labels.")
    selected = weights[positions[:, 0], positions[:, 1]]
    if not bool(selected.gt(0).all()):
        raise ValueError("supervised loss weights must be positive.")
    return selected


def batch_loss(
    token_loss: Tensor,
    batch_index: Tensor,
    batch_size: int,
    weights: Tensor,
) -> Tensor:
    if token_loss.shape != weights.shape:
        raise ValueError("token_loss and weights must have the same shape.")
    loss = token_loss.new_zeros(batch_size)
    counts = token_loss.new_zeros(batch_size)
    loss.scatter_add_(0, batch_index, token_loss * weights)
    counts.scatter_add_(0, batch_index, weights)
    if not bool(counts.gt(0).all()):
        raise ValueError("each batch row must contain at least one supervised token.")
    return loss / counts
