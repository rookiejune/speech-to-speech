"""Task-family metric logging for Lightning training."""

from __future__ import annotations

import torch
from torch import Tensor

from ..datamodule.types import CausalLMBatch, TaskFamily
from .metrics import MetricLogger, log_reduced_mean, log_reduced_sum, reduced_weighted_mean

TASK_FAMILY_GROUPS = {
    "semantic_ar": (
        TaskFamily.SOURCE_AR,
        TaskFamily.TARGET_AR,
    ),
    "translation": (
        TaskFamily.SOURCE_TO_TARGET,
        TaskFamily.TARGET_TO_SOURCE,
    ),
}


def log_task_losses(
    logger: MetricLogger,
    batch: CausalLMBatch,
    row_loss: Tensor,
    token_counts: Tensor,
    *,
    name_prefix: str = "",
    on_step: bool,
) -> None:
    if batch.task_family is None:
        return
    row_loss = row_loss.detach()
    for family in TaskFamily:
        mask = batch.task_family.eq(family.id)
        mean = reduced_weighted_mean(row_loss[mask], token_counts[mask])
        tokens = token_counts[mask].sum()
        log_reduced_mean(
            logger,
            f"{name_prefix}loss/{family.value}",
            mean,
            on_step=on_step,
            on_epoch=True,
        )
        log_reduced_sum(
            logger,
            f"{name_prefix}tokens/{family.value}",
            tokens,
            on_step=on_step,
            on_epoch=False,
            skip_zero=True,
        )


def log_task_group_losses(
    logger: MetricLogger,
    batch: CausalLMBatch,
    row_loss: Tensor,
    token_counts: Tensor,
    *,
    name_prefix: str = "",
    on_step: bool,
) -> None:
    if batch.task_family is None:
        return
    row_loss = row_loss.detach()
    for name, families in TASK_FAMILY_GROUPS.items():
        mask = torch.zeros_like(batch.task_family, dtype=torch.bool)
        for family in families:
            mask |= batch.task_family.eq(family.id)
        log_reduced_mean(
            logger,
            f"{name_prefix}loss/{name}",
            reduced_weighted_mean(row_loss[mask], token_counts[mask]),
            on_step=on_step,
            on_epoch=True,
        )
