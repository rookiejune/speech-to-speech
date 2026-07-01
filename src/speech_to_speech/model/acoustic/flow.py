from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .condition import acoustic_velocity


@dataclass(frozen=True)
class AcousticFlowSample:
    final: Tensor
    time_grid: Tensor
    forward_count: int


class _AcousticFlowModel(nn.Module):
    def __init__(self, dit: nn.Module, *, guidance_scale: float) -> None:
        super().__init__()
        self.dit = dit
        self.guidance_scale = guidance_scale
        self.forward_count = 0

    def forward(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        *,
        last_hidden_state: Tensor,
        acoustic_condition: Tensor,
        mask: Tensor,
    ) -> Tensor:
        self.forward_count += 1
        return acoustic_velocity(
            self.dit,
            x_t=x_t,
            timesteps=timesteps,
            last_hidden_state=last_hidden_state,
            acoustic_condition=acoustic_condition,
            mask=mask,
            guidance_scale=self.guidance_scale,
        )


def acoustic_flow_source_sample_like(target_features: Tensor) -> Tensor:
    from anytrain.framework.flow_matching import ContinuousFlowMatcher

    return ContinuousFlowMatcher().source.sample_like(target_features)


@torch.no_grad()
def full_sequence_acoustic_flow_sample(
    dit: nn.Module,
    x_0: Tensor,
    *,
    last_hidden_state: Tensor,
    acoustic_condition: Tensor,
    mask: Tensor,
    time_grid: Tensor,
    guidance_scale: float = 1.0,
) -> AcousticFlowSample:
    from anytrain.framework.flow_matching import ContinuousFlowMatcher, ODESampler

    model = _AcousticFlowModel(dit, guidance_scale=guidance_scale)
    matcher = ContinuousFlowMatcher(
        sampler=ODESampler(
            method="euler",
            nfe=time_grid.numel() - 1,
            num_steps=time_grid.numel(),
            return_intermediates=False,
        )
    )
    sample = matcher.sample(
        model,
        x_0,
        time_grid=time_grid,
        last_hidden_state=last_hidden_state,
        acoustic_condition=acoustic_condition,
        mask=mask,
    )
    return AcousticFlowSample(
        final=torch.where(mask.unsqueeze(-1), sample.final, x_0),
        time_grid=time_grid,
        forward_count=model.forward_count,
    )


__all__ = [
    "AcousticFlowSample",
    "acoustic_flow_source_sample_like",
    "full_sequence_acoustic_flow_sample",
]
