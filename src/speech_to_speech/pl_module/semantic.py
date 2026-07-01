"""Semantic loss helpers used by the Lightning module."""

from __future__ import annotations

import torch
from anytrain.idspace import IdSpace
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..types.datamodule import CausalLMBatch, IGNORE_INDEX
from ..model.semantic import loss_positions, loss_weights
from ..types.model import AudioBoundary


def semantic_batch(
    batch: CausalLMBatch,
    *,
    idspace: IdSpace,
    stop_loss_weight: float,
) -> CausalLMBatch:
    if stop_loss_weight == 1.0:
        return batch
    eoa_id = idspace.special_token_id(AudioBoundary.EOA)
    weights = base_loss_weights(batch)
    stop_mask = batch.labels.eq(eoa_id)
    weights = torch.where(
        stop_mask,
        weights * stop_loss_weight,
        weights,
    )
    return with_loss_weights(batch, weights)


def semantic_row_loss(
    batch: CausalLMBatch,
    output: CausalLMOutputWithPast,
) -> Tensor:
    loss = output.loss
    if loss is None:
        raise RuntimeError("model output must include loss.")
    if loss.dim() == 0:
        return loss.unsqueeze(0).expand(batch.input_ids.size(0))
    if loss.dim() != 1 or loss.size(0) != batch.input_ids.size(0):
        raise RuntimeError("model loss must be scalar or one value per batch row.")
    return loss


def loss_token_counts(batch: CausalLMBatch, *, dtype: torch.dtype) -> Tensor:
    positions = loss_positions(batch)
    counts = torch.zeros(batch.input_ids.size(0), device=batch.labels.device)
    weights = loss_weights(batch, positions, dtype=counts.dtype)
    counts.scatter_add_(
        0,
        positions[:, 0],
        weights,
    )
    return counts.to(dtype=dtype)


def base_loss_weights(batch: CausalLMBatch) -> Tensor:
    if batch.loss_weights is None:
        return batch.labels.ne(IGNORE_INDEX).to(dtype=torch.float)
    if batch.loss_weights.shape != batch.labels.shape:
        raise ValueError("loss_weights must have the same shape as labels.")
    return batch.loss_weights.clone()


def with_loss_weights(batch: CausalLMBatch, loss_weights: Tensor) -> CausalLMBatch:
    return CausalLMBatch(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        labels=batch.labels,
        logits_to_keep=batch.logits_to_keep,
        loss_weights=loss_weights,
        source_audio=batch.source_audio,
        target_audio=batch.target_audio,
        task_family=batch.task_family,
    )
