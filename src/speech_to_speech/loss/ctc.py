from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from anytrain.loss import LossItem
from torch import Tensor, nn
from torch.nn import functional as F

from ..datamodule.contract import (
    CTC_PAD_ID,
    CTCTarget,
)
from ..model.ctc import CTCRoute


CTCDecode = Callable[[CTCRoute, Tensor, Tensor], tuple[Tensor, Tensor]]


@dataclass(frozen=True)
class CTCRouteConfig:
    weight: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise TypeError("CTC route weight must be a number.")
        if not math.isfinite(float(self.weight)) or self.weight < 0:
            raise ValueError("CTC route weight must be finite and non-negative.")

    @property
    def enabled(self) -> bool:
        return self.weight > 0


@dataclass(frozen=True)
class CTCConfig:
    source: CTCRouteConfig = field(default_factory=CTCRouteConfig)
    target: CTCRouteConfig = field(default_factory=CTCRouteConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.source, CTCRouteConfig):
            raise TypeError("CTC source loss must be a CTCRouteConfig.")
        if not isinstance(self.target, CTCRouteConfig):
            raise TypeError("CTC target loss must be a CTCRouteConfig.")

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


class CTCAlignmentLoss(nn.Module):
    """Project audio-slot hidden states through the frozen text readout."""

    def __init__(self, blank_token_id: int, config: CTCConfig) -> None:
        super().__init__()
        if isinstance(blank_token_id, bool) or not isinstance(blank_token_id, int):
            raise TypeError("CTC blank_token_id must be an integer.")
        if blank_token_id < 0:
            raise ValueError("CTC blank_token_id must be non-negative.")
        if not isinstance(config, CTCConfig):
            raise TypeError("CTC alignment config must be a CTCConfig.")
        self.blank_token_id = blank_token_id
        self.config = config

    def forward(
        self,
        reference_states: Tensor,
        *,
        source_hidden_states: Tensor | None,
        target_hidden_states: Tensor | None,
        source: CTCTarget | None,
        target: CTCTarget | None,
        decode: CTCDecode,
    ) -> LossItem:
        if reference_states.dim() != 3 or not reference_states.is_floating_point():
            raise ValueError("CTC hidden_states must be floating-point [B, T, H].")
        if not callable(decode):
            raise TypeError("CTC decode must be callable.")
        source_loss, source_tokens, source_steps = self._route(
            reference_states,
            source_hidden_states,
            source if self.config.source.enabled else None,
            decode,
            route=CTCRoute.SOURCE,
            name="source",
        )
        target_loss, target_tokens, target_steps = self._route(
            reference_states,
            target_hidden_states,
            target if self.config.target.enabled else None,
            decode,
            route=CTCRoute.TARGET,
            name="target",
        )
        sequences = (source_tokens.gt(0) | target_tokens.gt(0)).to(
            dtype=reference_states.dtype
        )
        loss = (
            float(self.config.source.weight) * source_loss
            + float(self.config.target.weight) * target_loss
        )
        return LossItem(
            loss=loss,
            details={
                "source_loss": source_loss,
                "target_loss": target_loss,
                "source_tokens": source_tokens,
                "target_tokens": target_tokens,
                "source_steps": source_steps,
                "target_steps": target_steps,
                "tokens": source_tokens + target_tokens,
                "sequences": sequences,
            },
        )

    def _route(
        self,
        reference_states: Tensor,
        hidden_states: Tensor | None,
        value: CTCTarget | None,
        decode: CTCDecode,
        *,
        route: CTCRoute,
        name: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = reference_states.size(0)
        zero = reference_states.new_zeros(batch_size)
        if value is None:
            return zero, zero, zero
        if hidden_states is None:
            raise ValueError(f"{name} CTC target requires route hidden states.")
        if hidden_states.dim() != 3 or not hidden_states.is_floating_point():
            raise ValueError(f"{name} CTC hidden states must be floating [B, T, H].")
        if hidden_states.size(0) != batch_size:
            raise ValueError(f"{name} CTC hidden states must align with batch rows.")
        positions = value["token_positions"]
        labels = value["text_token_ids"]
        if positions.dim() != 2 or labels.dim() != 2:
            raise ValueError(f"{name} CTC fields must have shapes [B, A] and [B, U].")
        if positions.size(0) != batch_size or labels.size(0) != batch_size:
            raise ValueError(f"{name} CTC fields must align with hidden-state rows.")
        if positions.device != hidden_states.device or labels.device != hidden_states.device:
            raise ValueError(f"{name} CTC fields must share the hidden-state device.")

        position_mask = positions.ne(CTC_PAD_ID)
        label_mask = labels.ne(CTC_PAD_ID)
        target_lengths = label_mask.sum(dim=1).to(dtype=torch.long)
        active = target_lengths.gt(0)
        if not bool(active.any()):
            return zero, zero, zero

        selected_positions = positions - int(route is CTCRoute.TARGET)
        safe_positions = selected_positions.clamp_min(0)
        if bool((safe_positions[position_mask] >= hidden_states.size(1)).any()):
            raise ValueError(f"{name} CTC position lies outside hidden states.")
        states = hidden_states.gather(
            1,
            safe_positions[..., None].expand(-1, -1, hidden_states.size(-1)),
        )
        logits, decoded_mask = decode(route, states, position_mask)
        if not isinstance(logits, Tensor) or logits.dim() != 3:
            raise ValueError("CTC decoder must return logits with shape [B, P, V].")
        if (
            not isinstance(decoded_mask, Tensor)
            or decoded_mask.dtype is not torch.bool
            or decoded_mask.shape != logits.shape[:2]
        ):
            raise ValueError("CTC decoder mask must be boolean and align with logits.")
        if logits.size(0) != batch_size or logits.size(-1) < 1:
            raise ValueError("CTC text logits must align with batch rows.")
        if logits.device != hidden_states.device or decoded_mask.device != logits.device:
            raise ValueError("CTC decoder outputs must share the hidden-state device.")
        input_lengths = decoded_mask.sum(dim=1).to(dtype=torch.long)
        _validate_pooled_lengths(
            input_lengths[active],
            labels[active],
            label_mask[active],
            name=name,
        )
        if self.blank_token_id >= logits.size(-1):
            raise ValueError("CTC blank token lies outside the text vocabulary.")
        active_labels = labels[active][label_mask[active]].to(dtype=torch.long)
        if bool(active_labels.eq(self.blank_token_id).any()):
            raise ValueError("CTC targets must not contain the blank token.")
        if bool((active_labels < 0).any()) or bool(
            (active_labels >= logits.size(-1)).any()
        ):
            raise ValueError("CTC targets contain an id outside the text vocabulary.")

        route_loss = F.ctc_loss(
            F.log_softmax(logits[active].float(), dim=-1).transpose(0, 1),
            active_labels,
            input_lengths[active],
            target_lengths[active],
            blank=self.blank_token_id,
            reduction="none",
            zero_infinity=False,
        )
        normalized = route_loss / target_lengths[active].to(dtype=route_loss.dtype)
        loss = zero.to(dtype=normalized.dtype)
        loss[active] = normalized
        return (
            loss.to(dtype=hidden_states.dtype),
            target_lengths.to(dtype=hidden_states.dtype),
            input_lengths.to(dtype=hidden_states.dtype),
        )


def _validate_pooled_lengths(
    input_lengths: Tensor,
    labels: Tensor,
    label_mask: Tensor,
    *,
    name: str,
) -> None:
    repeats = (
        label_mask[:, 1:]
        & label_mask[:, :-1]
        & labels[:, 1:].eq(labels[:, :-1])
    ).sum(dim=1)
    minimum = label_mask.sum(dim=1) + repeats
    if bool((input_lengths < minimum).any()):
        raise ValueError(
            f"{name} CTC pooling leaves too few steps for its transcript labels."
        )


__all__ = [
    "CTCAlignmentLoss",
    "CTCConfig",
    "CTCDecode",
    "CTCRouteConfig",
]
