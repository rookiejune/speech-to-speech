from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict, cast

import torch
from torch import Tensor, nn

from .._compat import StrEnum, auto
from ._checkpointing import GradientCheckpointingLayer


class AdapterType(StrEnum):
    LINEAR = auto()
    MLP = auto()


class TowerFields(TypedDict):
    layers: int
    heads: int
    ffn_ratio: float
    dropout: float


class MLPAdapter(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        intermediate_size = int(round((8.0 / 3.0) * in_features))

        self.gate_proj = nn.Linear(in_features, intermediate_size, bias=False)
        self.up_proj = nn.Linear(in_features, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, out_features, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def create_adapter(
    adapter_type: AdapterType | None, in_features: int, out_features: int
) -> nn.Module:
    if adapter_type is None:
        if in_features != out_features:
            raise ValueError("identity adapter requires matching feature dimensions.")
        return nn.Identity()
    if adapter_type is AdapterType.LINEAR:
        return nn.Linear(in_features=in_features, out_features=out_features)
    if adapter_type is AdapterType.MLP:
        return MLPAdapter(in_features=in_features, out_features=out_features)
    raise AssertionError(f"unsupported adapter type: {adapter_type}")


def register(
    module: nn.Module,
    name: str,
    tensor: torch.Tensor,
    *,
    persistent: bool = True,
) -> None:
    """Register a buffer in both the Stable Codec Torch 2.4 env and newer Torch."""
    buffer_type = getattr(nn, "Buffer", None)
    if buffer_type is None:
        module.register_buffer(name, tensor, persistent=persistent)
        return
    setattr(module, name, buffer_type(tensor, persistent=persistent))


def top_p_filter(logits: Tensor, top_p: float) -> Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probabilities = sorted_logits.softmax(dim=-1)
    remove = probabilities.cumsum(dim=-1) - probabilities >= top_p
    remove[..., 0] = False
    filtered = logits.new_full(logits.shape, float("-inf"))
    filtered.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits.masked_fill(remove, float("-inf")),
    )
    return filtered


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


def require_embedding(value: object, name: str) -> nn.Embedding:
    if not isinstance(value, nn.Embedding):
        raise TypeError(f"{name} must be a torch.nn.Embedding.")
    return value


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
        values: torch.Tensor,
        *,
        cast_output: bool = True,
    ) -> torch.Tensor:
        output = self.module(values.to(dtype=self._compute_reference.dtype))
        if cast_output:
            output = output.to(dtype=self._output_reference.dtype)
        return output
