from __future__ import annotations

import torch
from torch import Tensor, nn


class HiddenConditionAdapter(nn.Module):
    """Map causal token-model states into the acoustic generator condition space."""

    def __init__(self, hidden_dim: int, condition_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0 or condition_dim <= 0:
            raise ValueError("hidden and acoustic condition dimensions must be positive.")
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, condition_dim)
        if hidden_dim == condition_dim:
            with torch.no_grad():
                self.projection.weight.copy_(torch.eye(hidden_dim))
                self.projection.bias.zero_()

    def forward(self, hidden_state: Tensor) -> Tensor:
        if hidden_state.dim() != 3 or hidden_state.size(-1) != self.hidden_dim:
            raise ValueError("hidden state must have shape [batch, frame, hidden_dim].")
        parameter = self.projection.weight
        value = hidden_state.to(device=parameter.device, dtype=parameter.dtype)
        return self.projection(self.norm(value))


__all__ = ["HiddenConditionAdapter"]
