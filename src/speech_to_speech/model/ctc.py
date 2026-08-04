from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional, cast

import torch
from torch import Tensor, nn

from .._compat import StrEnum, auto
from ..runtime.backbone import BackboneOutputView
from ..runtime.types import BackboneOutput, BackboneReadout
from ._checkpointing import GradientCheckpointingLayer
from ._helper import register, safe_transformer_mask, validate_tower_fields


class CTCRoute(StrEnum):
    SOURCE = auto()
    TARGET = auto()


class CTCDecoderType(StrEnum):
    """Architecture between an audio-slot backbone readout and the text head."""

    IDENTITY = auto()
    LINEAR = auto()
    TRANSFORMER = auto()


@dataclass(frozen=True)
class CTCDecoderConfig:
    """One route-local CTC decoder.

    ``backbone_readout=None`` reuses the prediction-modality readout selected for
    token CE. An explicit path may select another final branch or a hidden layer
    exposed by the backbone output. The vocabulary projection is never
    configurable: all variants still terminate at the frozen tied text head.
    """

    type: CTCDecoderType = CTCDecoderType.IDENTITY
    backbone_readout: Optional[str] = None
    pool_factor: int = 1
    layers: int = 2
    heads: int = 8
    ffn_ratio: float = 4.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.type, CTCDecoderType):
            raise TypeError("CTC decoder type must be a CTCDecoderType.")
        if self.backbone_readout is not None:
            BackboneReadout(self.backbone_readout)
        if isinstance(self.pool_factor, bool) or not isinstance(
            self.pool_factor, int
        ):
            raise TypeError("CTC decoder pool_factor must be an integer.")
        if self.pool_factor <= 0:
            raise ValueError("CTC decoder pool_factor must be positive.")
        validate_tower_fields(
            "CTC decoder",
            layers=self.layers,
            heads=self.heads,
            ffn_ratio=self.ffn_ratio,
            dropout=self.dropout,
        )

    @property
    def requires_hidden_states(self) -> bool:
        return (
            self.backbone_readout is not None
            and BackboneReadout(self.backbone_readout).requires_hidden_states
        )


@dataclass(frozen=True)
class CTCRouteConfig:
    weight: float = 0.0
    decoder: CTCDecoderConfig = field(default_factory=CTCDecoderConfig)

    def __post_init__(self) -> None:
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise TypeError("CTC route weight must be a number.")
        if not math.isfinite(float(self.weight)) or self.weight < 0:
            raise ValueError("CTC route weight must be finite and non-negative.")
        if not isinstance(self.decoder, CTCDecoderConfig):
            raise TypeError("CTC route decoder must be a CTCDecoderConfig.")

    @property
    def enabled(self) -> bool:
        return self.weight > 0


@dataclass(frozen=True)
class CTCConfig:
    source: CTCRouteConfig = field(default_factory=CTCRouteConfig)
    target: CTCRouteConfig = field(default_factory=CTCRouteConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.source, CTCRouteConfig):
            raise TypeError("CTC source route must be a CTCRouteConfig.")
        if not isinstance(self.target, CTCRouteConfig):
            raise TypeError("CTC target route must be a CTCRouteConfig.")

    @property
    def enabled(self) -> bool:
        return self.source.enabled or self.target.enabled

    @property
    def active_routes(self) -> frozenset[CTCRoute]:
        return frozenset(
            route
            for route, config in (
                (CTCRoute.SOURCE, self.source),
                (CTCRoute.TARGET, self.target),
            )
            if config.enabled
        )

    def route(self, route: CTCRoute) -> CTCRouteConfig:
        if route is CTCRoute.SOURCE:
            return self.source
        if route is CTCRoute.TARGET:
            return self.target
        raise AssertionError(f"unsupported CTC route: {route}")


@dataclass(frozen=True)
class ObjectiveHiddenOutput:
    token: Tensor
    source_ctc: Tensor | None = None
    target_ctc: Tensor | None = None

    def route(self, route: CTCRoute) -> Tensor | None:
        if route is CTCRoute.SOURCE:
            return self.source_ctc
        if route is CTCRoute.TARGET:
            return self.target_ctc
        raise AssertionError(f"unsupported CTC route: {route}")


class CTCDecoder(GradientCheckpointingLayer):
    """Pool and decode one audio route while preserving the frozen text head."""

    def __init__(
        self,
        config: CTCDecoderConfig,
        hidden_size: int,
        *,
        causal: bool,
    ) -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
            raise TypeError("CTC decoder hidden_size must be an integer.")
        if hidden_size <= 0:
            raise ValueError("CTC decoder hidden_size must be positive.")
        if not isinstance(causal, bool):
            raise TypeError("CTC decoder causal must be a bool.")
        if config.type is CTCDecoderType.TRANSFORMER and hidden_size % config.heads:
            raise ValueError(
                "CTC transformer hidden_size must be divisible by decoder heads."
            )

        self.config = config
        self.hidden_size = hidden_size
        self.causal = causal
        self._dtype_reference: Tensor
        register(
            self,
            "_dtype_reference",
            torch.empty(0, dtype=torch.float32),
            persistent=False,
        )
        if config.type is CTCDecoderType.IDENTITY:
            self.decoder: nn.Module = nn.Identity()
        elif config.type is CTCDecoderType.LINEAR:
            projection = nn.Linear(hidden_size, hidden_size, bias=False)
            nn.init.eye_(projection.weight)
            self.decoder = projection
        elif config.type is CTCDecoderType.TRANSFORMER:
            intermediate = max(1, int(round(config.ffn_ratio * hidden_size)))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=config.heads,
                dim_feedforward=intermediate,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.decoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        else:
            raise AssertionError(f"unsupported CTC decoder type: {config.type}")
        self.to(dtype=torch.float32)

    def forward(self, hidden_states: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if hidden_states.dim() != 3 or not hidden_states.is_floating_point():
            raise ValueError("CTC decoder hidden_states must be floating [B, A, H].")
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError("CTC decoder hidden dimension does not match its config.")
        if mask.dtype is not torch.bool or mask.shape != hidden_states.shape[:2]:
            raise ValueError("CTC decoder mask must be boolean [B, A].")
        if mask.device != hidden_states.device:
            raise ValueError("CTC decoder mask must share the hidden-state device.")

        values, pooled_mask = _masked_mean_pool(
            hidden_states,
            mask,
            self.config.pool_factor,
        )
        if self.config.type is CTCDecoderType.IDENTITY:
            return values.masked_fill(~pooled_mask[..., None], 0), pooled_mask

        values = values.to(dtype=self._dtype_reference.dtype)
        values = values.masked_fill(~pooled_mask[..., None], 0)
        if self.config.type is CTCDecoderType.LINEAR:
            return self.decoder(values).masked_fill(
                ~pooled_mask[..., None], 0
            ), pooled_mask

        safe = safe_transformer_mask(pooled_mask)
        causal_mask = (
            torch.ones(
                values.size(1),
                values.size(1),
                device=values.device,
                dtype=torch.bool,
            ).triu(1)
            if self.causal
            else None
        )
        decoded = self.decoder(
            values,
            mask=causal_mask,
            src_key_padding_mask=~safe,
            is_causal=self.causal,
        )
        return decoded.masked_fill(~pooled_mask[..., None], 0), pooled_mask

    def contract_state(self) -> Mapping[str, object]:
        return {
            "grammar": "ctc-decoder-v1",
            "type": self.config.type.value,
            "causal": self.causal,
            "backbone_readout": self.config.backbone_readout,
            "pool_factor": self.config.pool_factor,
            "hidden_size": self.hidden_size,
            "layers": self.config.layers,
            "heads": self.config.heads,
            "ffn_ratio": self.config.ffn_ratio,
            "dropout": self.config.dropout,
        }


class CTCDecoderRoutes(nn.Module):
    def __init__(self, config: CTCConfig, hidden_size: int) -> None:
        super().__init__()
        if not isinstance(config, CTCConfig):
            raise TypeError("CTC decoder routes require a CTCConfig.")
        self.config = config
        self.source = CTCDecoder(config.source.decoder, hidden_size, causal=False)
        self.target = CTCDecoder(config.target.decoder, hidden_size, causal=True)

    def decoder(self, route: CTCRoute) -> CTCDecoder:
        if route is CTCRoute.SOURCE:
            return self.source
        if route is CTCRoute.TARGET:
            return self.target
        raise AssertionError(f"unsupported CTC route: {route}")

    def requires_hidden_states(self, routes: frozenset[CTCRoute]) -> bool:
        return any(self.decoder(route).config.requires_hidden_states for route in routes)

    def hidden_states(
        self,
        output: BackboneOutputView,
        route: CTCRoute,
    ) -> Tensor:
        readout = self.decoder(route).config.backbone_readout
        value = (
            output.last_hidden_state
            if readout is None
            else BackboneReadout(readout).select(cast(BackboneOutput, output.output))
        )
        if value.dim() != 3 or not value.is_floating_point():
            raise ValueError("CTC backbone readout must resolve to floating [B, T, H].")
        if value.size(-1) != self.decoder(route).hidden_size:
            raise ValueError("CTC backbone readout hidden size does not match decoder.")
        return value

    def forward(
        self,
        route: CTCRoute,
        hidden_states: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.decoder(route)(hidden_states, mask)


def _masked_mean_pool(
    hidden_states: Tensor,
    mask: Tensor,
    factor: int,
) -> tuple[Tensor, Tensor]:
    if factor == 1:
        return hidden_states, mask
    steps = hidden_states.size(1)
    pooled_steps = math.ceil(steps / factor)
    padding = pooled_steps * factor - steps
    if padding:
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, padding))
        mask = torch.nn.functional.pad(mask, (0, padding), value=False)
    grouped = hidden_states.reshape(
        hidden_states.size(0),
        pooled_steps,
        factor,
        hidden_states.size(-1),
    )
    grouped_mask = mask.reshape(mask.size(0), pooled_steps, factor)
    counts = grouped_mask.sum(dim=2)
    pooled = (
        grouped.masked_fill(~grouped_mask[..., None], 0).sum(dim=2)
        / counts.clamp_min(1)[..., None].to(dtype=hidden_states.dtype)
    )
    pooled_mask = counts.gt(0)
    return pooled.masked_fill(~pooled_mask[..., None], 0), pooled_mask


__all__ = [
    "CTCConfig",
    "CTCDecoder",
    "CTCDecoderConfig",
    "CTCDecoderRoutes",
    "CTCDecoderType",
    "CTCRoute",
    "CTCRouteConfig",
    "ObjectiveHiddenOutput",
]
