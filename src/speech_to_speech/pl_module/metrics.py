"""Distributed-safe metric reductions for the Lightning training adapter.

The helpers in this module consume losses and counts already computed by
`SpeechToSpeechModule`, reduce sum/count pairs across the active process group,
and return values that can be logged with Lightning `sync_dist=False`. They do
not inspect raw dataset samples or model internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch.distributed as dist
from torch import Tensor


class MetricLogger(Protocol):
    def log(self, name: str, value: Tensor, **kwargs: object) -> None: ...


@dataclass(frozen=True)
class ReducedMean:
    value: Tensor
    weight: Tensor
    local_sum: Tensor


def reduced_weighted_mean(values: Tensor, weights: Tensor) -> ReducedMean:
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape.")
    weights = weights.to(device=values.device, dtype=values.dtype)
    local_sum = (values * weights).sum()
    local_weight = weights.sum()
    global_sum = reduced_sum(local_sum)
    global_weight = reduced_sum(local_weight)
    value = global_sum.new_zeros(())
    if has_positive_weight(global_weight):
        value = global_sum / global_weight.to(device=global_sum.device, dtype=global_sum.dtype)
    return ReducedMean(value=value, weight=global_weight, local_sum=local_sum)


def scaled_loss(mean: ReducedMean) -> Tensor:
    if not has_positive_weight(mean.weight):
        raise ValueError("weighted mean requires a positive weight sum.")
    weight = mean.weight.to(device=mean.local_sum.device, dtype=mean.local_sum.dtype)
    return mean.local_sum * float(distributed_world_size()) / weight


def reduced_sum(value: Tensor) -> Tensor:
    result = value.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def has_positive_weight(weight: Tensor) -> bool:
    return bool(weight.detach().gt(0).cpu().item())


def log_reduced_mean(
    logger: MetricLogger,
    name: str,
    mean: ReducedMean,
    *,
    on_step: bool,
    on_epoch: bool,
    prog_bar: bool = False,
) -> None:
    log_reduced_value(
        logger,
        name,
        mean.value,
        mean.weight,
        on_step=on_step,
        on_epoch=on_epoch,
        prog_bar=prog_bar,
    )


def log_reduced_sum(
    logger: MetricLogger,
    name: str,
    value: Tensor,
    *,
    on_step: bool,
    on_epoch: bool,
    prog_bar: bool = False,
    skip_zero: bool = False,
) -> None:
    value = reduced_sum(value)
    if skip_zero and not has_positive_weight(value):
        return
    logger.log(
        name,
        value,
        batch_size=1,
        on_step=on_step,
        on_epoch=on_epoch,
        prog_bar=prog_bar,
        sync_dist=False,
    )


def log_reduced_value(
    logger: MetricLogger,
    name: str,
    value: Tensor,
    weight: Tensor,
    *,
    on_step: bool,
    on_epoch: bool,
    prog_bar: bool = False,
) -> None:
    if not has_positive_weight(weight):
        return
    logger.log(
        name,
        value.detach(),
        batch_size=_batch_size(weight),
        on_step=on_step,
        on_epoch=on_epoch,
        prog_bar=prog_bar,
        sync_dist=False,
    )


def _batch_size(weight: Tensor) -> int:
    value = float(weight.detach().cpu().item())
    if value <= 0.0:
        return 0
    return max(1, int(value))
