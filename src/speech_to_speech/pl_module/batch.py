"""Batch movement helpers for Lightning training."""

from __future__ import annotations

from torch import Tensor
from torch import device as TorchDevice

from ..datamodule.types import CausalLMBatch, LongCatBatchSide


def batch_to_device(batch: CausalLMBatch, device: TorchDevice) -> CausalLMBatch:
    logits_to_keep = batch.logits_to_keep
    if isinstance(logits_to_keep, Tensor):
        logits_to_keep = logits_to_keep.to(device=device)
    task_family = batch.task_family
    if task_family is not None:
        task_family = task_family.to(device=device)
    return CausalLMBatch(
        input_ids=batch.input_ids.to(device=device),
        attention_mask=batch.attention_mask.to(device=device),
        labels=batch.labels.to(device=device),
        logits_to_keep=logits_to_keep,
        loss_weights=tensor_to_device(batch.loss_weights, device),
        source_audio=side_to_device(batch.source_audio, device),
        target_audio=side_to_device(batch.target_audio, device),
        task_family=task_family,
    )


def tensor_to_device(tensor: Tensor | None, device: TorchDevice) -> Tensor | None:
    if tensor is None:
        return None
    return tensor.to(device=device)


def side_to_device(
    side: LongCatBatchSide | None,
    device: TorchDevice,
) -> LongCatBatchSide | None:
    if side is None:
        return None
    return LongCatBatchSide(
        semantic_ids=side.semantic_ids.to(device=device),
        semantic_mask=side.semantic_mask.to(device=device),
        acoustic_ids=side.acoustic_ids.to(device=device),
        acoustic_mask=side.acoustic_mask.to(device=device),
    )
