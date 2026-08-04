from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any, TypedDict, cast

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer

from .._compat import register


class TowerFields(TypedDict):
    layers: int
    heads: int
    ffn_ratio: float
    dropout: float


def enable_gradient_checkpointing(module: nn.Module) -> int:
    """Enable HF-style activation checkpointing on custom model layers."""
    checkpointing = partial(checkpoint, use_reentrant=False)
    count = 0
    for layer in module.modules():
        if not isinstance(layer, GradientCheckpointingLayer):
            continue
        checkpoint_layer = cast(Any, layer)
        checkpoint_layer._gradient_checkpointing_func = checkpointing
        checkpoint_layer.gradient_checkpointing = True
        count += 1
    return count


def validate_tower_fields(
    name: str,
    *,
    layers: int,
    heads: int,
    ffn_ratio: float,
    dropout: float,
) -> None:
    if isinstance(layers, bool) or layers <= 0:
        raise ValueError(f"{name} layers must be positive.")
    if isinstance(heads, bool) or heads <= 0:
        raise ValueError(f"{name} heads must be positive.")
    if isinstance(ffn_ratio, bool) or ffn_ratio <= 0:
        raise ValueError(f"{name} ffn_ratio must be positive.")
    if isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise ValueError(f"{name} dropout must be in [0, 1).")


def tower_fields(config: Mapping[str, object]) -> TowerFields:
    return {
        "layers": cast(int, config.get("layers", 2)),
        "heads": cast(int, config.get("heads", 8)),
        "ffn_ratio": cast(float, config.get("ffn_ratio", 4.0)),
        "dropout": cast(float, config.get("dropout", 0.0)),
    }


def valid_mask(
    features: Tensor,
    mask: Tensor | None,
    *,
    name: str,
) -> Tensor:
    if mask is None:
        return torch.ones(
            features.shape[:2],
            device=features.device,
            dtype=torch.bool,
        )
    if mask.dim() != 2 or mask.shape != features.shape[:2]:
        raise ValueError(f"{name} mask must have shape [batch, frames].")
    return mask.to(device=features.device, dtype=torch.bool)


def safe_transformer_mask(valid: Tensor) -> Tensor:
    """Keep all-padding rows finite while their final outputs remain zero."""
    safe = valid.clone()
    empty = ~valid.any(dim=1)
    safe[:, 0] |= empty
    return safe


class CastOutput(GradientCheckpointingLayer):
    """Cast adapter outputs to the current backbone dtype at the idspace boundary."""

    def __init__(self, module: nn.Module, *, reference: Tensor) -> None:
        super().__init__()
        self.module = module
        parameter = next(module.parameters(), None)
        compute_reference = (
            reference.detach().new_empty(0, dtype=torch.float32)
            if parameter is None
            else parameter.detach().new_empty(0)
        )
        self._compute_reference: Tensor
        self._output_reference: Tensor
        register(self, "_compute_reference", compute_reference, persistent=False)
        register(
            self,
            "_output_reference",
            reference.detach().new_empty(0),
            persistent=False,
        )

    def forward(
        self,
        values: Tensor,
        *,
        cast_output: bool = True,
    ) -> Tensor:
        output = self.module(values.to(dtype=self._compute_reference.dtype))
        if cast_output:
            output = output.to(dtype=self._output_reference.dtype)
        return output


__all__ = [
    "CastOutput",
    "GradientCheckpointingLayer",
    "enable_gradient_checkpointing",
    "safe_transformer_mask",
    "tower_fields",
    "valid_mask",
    "validate_tower_fields",
]
