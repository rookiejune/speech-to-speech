from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from anytrain.loss import LossItem
from torch import Tensor, nn
from torch.nn import functional as F

from ..datamodule.types import CTC_PAD_ID, CTCTarget


TextReadout = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class CTCConfig:
    """Weights for transcript-latent source and target audio spans."""

    source_weight: float = 0.0
    target_weight: float = 0.0

    def __post_init__(self) -> None:
        _weight(self.source_weight, name="source_weight")
        _weight(self.target_weight, name="target_weight")

    @property
    def enabled(self) -> bool:
        return self.source_weight > 0 or self.target_weight > 0


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
        hidden_states: Tensor,
        *,
        source: CTCTarget | None,
        target: CTCTarget | None,
        text_readout: TextReadout,
    ) -> LossItem:
        if hidden_states.dim() != 3 or not hidden_states.is_floating_point():
            raise ValueError("CTC hidden_states must be floating-point [B, T, H].")
        if not callable(text_readout):
            raise TypeError("CTC text_readout must be callable.")
        source_loss, source_tokens, source_steps = self._route(
            hidden_states,
            source if self.config.source_weight > 0 else None,
            text_readout,
            causal=False,
            name="source",
        )
        target_loss, target_tokens, target_steps = self._route(
            hidden_states,
            target if self.config.target_weight > 0 else None,
            text_readout,
            causal=True,
            name="target",
        )
        sequences = (source_tokens.gt(0) | target_tokens.gt(0)).to(
            dtype=hidden_states.dtype
        )
        loss = (
            float(self.config.source_weight) * source_loss
            + float(self.config.target_weight) * target_loss
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
        hidden_states: Tensor,
        value: CTCTarget | None,
        text_readout: TextReadout,
        *,
        causal: bool,
        name: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = hidden_states.size(0)
        zero = hidden_states.new_zeros(batch_size)
        if value is None:
            return zero, zero, zero
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
        input_lengths = position_mask.sum(dim=1).to(dtype=torch.long)
        target_lengths = label_mask.sum(dim=1).to(dtype=torch.long)
        active = target_lengths.gt(0)
        if not bool(active.any()):
            return zero, zero, zero

        selected_positions = positions - int(causal)
        safe_positions = selected_positions.clamp_min(0)
        states = hidden_states.gather(
            1,
            safe_positions[..., None].expand(-1, -1, hidden_states.size(-1)),
        )
        logits = text_readout(states)
        if not isinstance(logits, Tensor) or logits.dim() != 3:
            raise ValueError("CTC text readout must return logits with shape [B, A, V].")
        if logits.shape[:2] != positions.shape or logits.size(-1) < 1:
            raise ValueError("CTC text logits must align with audio positions.")
        if self.blank_token_id >= logits.size(-1):
            raise ValueError("CTC blank token lies outside the text vocabulary.")
        active_labels = labels[active][label_mask[active]].to(dtype=torch.long)
        if bool(active_labels.eq(self.blank_token_id).any()):
            raise ValueError("CTC targets must not contain the blank token.")
        if bool((active_labels < 0).any()) or bool(
            (active_labels >= logits.size(-1)).any()
        ):
            raise ValueError("CTC targets contain an id outside the text vocabulary.")

        route = F.ctc_loss(
            F.log_softmax(logits[active].float(), dim=-1).transpose(0, 1),
            active_labels,
            input_lengths[active],
            target_lengths[active],
            blank=self.blank_token_id,
            reduction="none",
            zero_infinity=False,
        )
        normalized = route / target_lengths[active].to(dtype=route.dtype)
        loss = zero.to(dtype=normalized.dtype)
        loss[active] = normalized
        return (
            loss.to(dtype=hidden_states.dtype),
            target_lengths.to(dtype=hidden_states.dtype),
            input_lengths.to(dtype=hidden_states.dtype),
        )


def _weight(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"CTC {name} must be a number.")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"CTC {name} must be finite and non-negative.")


__all__ = ["CTCAlignmentLoss", "CTCConfig", "TextReadout"]
