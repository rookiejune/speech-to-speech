from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from torch import Tensor


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


def tower_fields(config: Mapping[str, object]) -> dict[str, int | float]:
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
    if bool(valid.any(dim=1).all()):
        return valid
    safe = valid.clone()
    empty = ~valid.any(dim=1)
    safe[empty, 0] = True
    return safe
