from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .types import LossItem


class TrainingSample(Protocol):
    x_t: Tensor
    t: Tensor
    velocity: Tensor


class FlowRuntime(Protocol):
    def training_sample(
        self,
        x_1: Tensor,
        *,
        x_0: Tensor | None = None,
    ) -> TrainingSample: ...


class FeatureDecoder(Protocol):
    def __call__(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor,
    ) -> Tensor: ...

    def forward_with_features(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]: ...


class AcousticFlowLoss(nn.Module):
    """Frame-masked velocity objective for acoustic latent prediction."""

    def forward(
        self,
        decoder: FeatureDecoder,
        condition: Tensor,
        target: Tensor,
        mask: Tensor,
        runtime: FlowRuntime,
    ) -> LossItem:
        self._validate_inputs(condition, target, mask)
        sample = runtime.training_sample(target)
        prediction = decoder(sample.x_t, sample.t, condition=condition, mask=mask)
        return self._loss(prediction, sample, target, mask)

    def forward_with_features(
        self,
        decoder: FeatureDecoder,
        condition: Tensor,
        target: Tensor,
        mask: Tensor,
        runtime: FlowRuntime,
    ) -> tuple[LossItem, Tensor]:
        self._validate_inputs(condition, target, mask)
        sample = runtime.training_sample(target)
        prediction, representation = decoder.forward_with_features(
            sample.x_t,
            sample.t,
            condition=condition,
            mask=mask,
        )
        return self._loss(prediction, sample, target, mask), representation

    def _validate_inputs(self, condition: Tensor, target: Tensor, mask: Tensor) -> None:
        if condition.dim() != 3 or target.dim() != 3 or mask.dim() != 2:
            raise ValueError(
                "condition, target, and mask must have shapes [B, F, H], [B, F, D], and [B, F]."
            )
        if condition.shape[:2] != target.shape[:2] or mask.shape != target.shape[:2]:
            raise ValueError(
                "flow condition, target, and mask must align on [batch, frame]."
            )
        if mask.dtype != torch.bool:
            raise TypeError("flow mask must be boolean.")

    def _loss(
        self,
        prediction: Tensor,
        sample: TrainingSample,
        target: Tensor,
        mask: Tensor,
    ) -> LossItem:
        if prediction.shape != sample.velocity.shape:
            raise ValueError("flow decoder output must match target latent shape.")

        frame_mask = mask[..., None]
        safe_prediction = prediction.masked_fill(~frame_mask, 0)
        safe_velocity = sample.velocity.masked_fill(~frame_mask, 0)
        frame_loss = F.mse_loss(
            safe_prediction,
            safe_velocity,
            reduction="none",
        ).mean(dim=-1)
        weights = mask.to(dtype=frame_loss.dtype)
        frame_count = weights.sum(dim=1)
        loss = frame_loss.sum(dim=1) / frame_count.clamp_min(1)
        return LossItem(
            loss=loss,
            details={
                "frames": frame_count.to(dtype=target.dtype),
                "t": sample.t.to(dtype=target.dtype),
            },
        )
