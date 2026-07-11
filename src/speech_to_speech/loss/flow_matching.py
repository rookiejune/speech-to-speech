from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .types import LossItem


class FlowRuntime(Protocol):
    def training_sample(self, x_1: Tensor, *, x_0: Tensor | None = None): ...


class AcousticFlowLoss(nn.Module):
    """Frame-masked velocity objective for acoustic latent prediction."""

    def forward(
        self,
        decoder: nn.Module,
        condition: Tensor,
        target: Tensor,
        mask: Tensor,
        runtime: FlowRuntime,
    ) -> LossItem:
        if condition.dim() != 3 or target.dim() != 3 or mask.dim() != 2:
            raise ValueError(
                "condition, target, and mask must have shapes [B, F, H], [B, F, D], and [B, F]."
            )
        if condition.shape[:2] != target.shape[:2] or mask.shape != target.shape[:2]:
            raise ValueError("flow condition, target, and mask must align on [batch, frame].")
        if mask.dtype != torch.bool:
            raise TypeError("flow mask must be boolean.")

        sample = runtime.training_sample(target)
        prediction = decoder(sample.x_t, sample.t, condition=condition)
        if prediction.shape != sample.velocity.shape:
            raise ValueError("flow decoder output must match target latent shape.")

        frame_loss = F.mse_loss(prediction, sample.velocity, reduction="none").mean(dim=-1)
        weights = mask.to(dtype=frame_loss.dtype)
        frame_count = weights.sum(dim=1)
        loss = (frame_loss * weights).sum(dim=1) / frame_count.clamp_min(1)
        return LossItem(
            loss=loss,
            details={
                "frames": frame_count.to(dtype=target.dtype),
                "t": sample.t.to(dtype=target.dtype),
            },
        )
