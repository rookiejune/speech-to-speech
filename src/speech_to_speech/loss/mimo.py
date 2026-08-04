"""Dual-stream causal language-model objective.

This objective is intentionally independent from the repository's serialized
single-stream :class:`TokenLoss`.  Text and semantic-audio logits are shifted
and normalized separately, then combined with explicit route weights.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from anytrain.loss import LossItem
from torch import Tensor, nn
from torch.nn import functional as F

from .._tensor import is_signed_integer_dtype
from ..mimo import MIMO_IGNORE_INDEX, MimoBatch

TensorReadout = Callable[[Tensor], Tensor]


class MimoObjective(nn.Module):
    """Compute aligned text/audio causal cross entropy.

    Parameters are local vocabulary logits, so text and audio heads may have
    different vocabulary sizes.  For each route, token losses are averaged per
    batch row using that route's mask before the two route losses are combined.
    A route may be entirely unsupervised for a row (for example, an audio-only
    task); its contribution is then zero rather than an artificial loss.
    """

    def __init__(
        self,
        *,
        text_weight: float = 1.0,
        audio_weight: float = 1.0,
        ignore_index: int = MIMO_IGNORE_INDEX,
    ) -> None:
        super().__init__()
        _validate_weight(text_weight, name="text_weight")
        _validate_weight(audio_weight, name="audio_weight")
        if text_weight == 0 and audio_weight == 0:
            raise ValueError("at least one MIMO route weight must be positive.")
        if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
            raise TypeError("ignore_index must be an integer.")
        self.text_weight = float(text_weight)
        self.audio_weight = float(audio_weight)
        self.ignore_index = ignore_index

    def forward(
        self,
        text_logits: Tensor,
        audio_logits: Tensor,
        text_labels: Tensor,
        audio_labels: Tensor,
        *,
        text_loss_mask: Tensor | None = None,
        audio_loss_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        validate: bool = True,
    ) -> LossItem:
        """Return one loss row per input example.

        ``*_logits`` and ``*_labels`` have shape ``[B, T, V]`` and ``[B, T]``.
        Causal alignment is handled here: logits at ``t`` supervise labels at
        ``t + 1``.  Explicit loss masks use the unshifted ``[B, T]`` layout,
        matching the batch contract.
        """

        if not isinstance(validate, bool):
            raise TypeError("validate must be a boolean.")
        _validate_logits(text_logits, text_labels, name="text")
        _validate_logits(audio_logits, audio_labels, name="audio")
        if text_logits.shape[:2] != audio_logits.shape[:2]:
            raise ValueError("text and audio logits must share [B, T].")
        if text_labels.shape != audio_labels.shape:
            raise ValueError("text and audio labels must share [B, T].")
        _validate_devices(
            text_logits,
            audio_logits,
            text_labels,
            audio_labels,
        )

        batch_size, sequence_length = text_labels.shape
        effective_attention = _effective_mask(
            attention_mask,
            (batch_size, sequence_length),
            device=text_labels.device,
            name="attention_mask",
        )
        text_mask = _effective_loss_mask(
            text_loss_mask,
            text_labels,
            effective_attention,
            name="text_loss_mask",
            ignore_index=self.ignore_index,
            validate=validate,
        )
        audio_mask = _effective_loss_mask(
            audio_loss_mask,
            audio_labels,
            effective_attention,
            name="audio_loss_mask",
            ignore_index=self.ignore_index,
            validate=validate,
        )
        target_attention = effective_attention[:, 1:] & effective_attention[:, :-1]
        text_target_mask = text_mask[:, 1:] & target_attention
        audio_target_mask = audio_mask[:, 1:] & target_attention

        if validate and not bool((text_target_mask | audio_target_mask).any(dim=1).all()):
            raise ValueError("each MIMO loss row must contain a supervised target.")

        text_token_loss = _causal_cross_entropy(
            text_logits,
            text_labels,
            text_target_mask,
            ignore_index=self.ignore_index,
            validate=validate,
            name="text",
        )
        audio_token_loss = _causal_cross_entropy(
            audio_logits,
            audio_labels,
            audio_target_mask,
            ignore_index=self.ignore_index,
            validate=validate,
            name="audio",
        )
        text_count = text_target_mask.sum(dim=1)
        audio_count = audio_target_mask.sum(dim=1)
        text_loss = _masked_row_mean(text_token_loss, text_target_mask, text_count)
        audio_loss = _masked_row_mean(audio_token_loss, audio_target_mask, audio_count)
        total_loss = self.text_weight * text_loss + self.audio_weight * audio_loss
        total_count = text_count + audio_count
        return LossItem(
            loss=total_loss,
            details={
                "text_loss": text_loss,
                "audio_loss": audio_loss,
                "text_tokens": text_count.to(dtype=total_loss.dtype),
                "audio_tokens": audio_count.to(dtype=total_loss.dtype),
                "tokens": total_count.to(dtype=total_loss.dtype),
            },
        )

    def from_hidden_states(
        self,
        text_hidden_states: Tensor,
        audio_hidden_states: Tensor,
        text_labels: Tensor,
        audio_labels: Tensor,
        *,
        text_readout: TensorReadout,
        audio_readout: TensorReadout,
        text_loss_mask: Tensor | None = None,
        audio_loss_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        validate: bool = True,
    ) -> LossItem:
        """Apply independent readouts, then evaluate :meth:`forward`.

        This helper keeps the objective agnostic to a concrete HF model while
        making the intended dual-hidden-state integration explicit.
        """

        _validate_hidden(text_hidden_states, text_labels, name="text")
        _validate_hidden(audio_hidden_states, audio_labels, name="audio")
        if text_hidden_states.shape[:2] != audio_hidden_states.shape[:2]:
            raise ValueError("text and audio hidden states must share [B, T].")
        if not callable(text_readout) or not callable(audio_readout):
            raise TypeError("text_readout and audio_readout must be callable.")
        text_logits = text_readout(text_hidden_states)
        audio_logits = audio_readout(audio_hidden_states)
        return self(
            text_logits,
            audio_logits,
            text_labels,
            audio_labels,
            text_loss_mask=text_loss_mask,
            audio_loss_mask=audio_loss_mask,
            attention_mask=attention_mask,
            validate=validate,
        )

    def mean(self, item: LossItem, *, distributed: bool = False) -> Tensor:
        """Reduce rows with optional global token normalization for DDP."""

        if not isinstance(distributed, bool):
            raise TypeError("distributed must be a boolean.")
        if not isinstance(item, LossItem):
            raise TypeError("MimoObjective.mean expects a LossItem.")
        details = item.details
        required = {"text_loss", "audio_loss", "text_tokens", "audio_tokens"}
        if details is None or not required.issubset(details):
            raise ValueError("MIMO loss details are missing route losses or counts.")
        text, audio = self.route_means(item, distributed=distributed)
        return self.text_weight * text + self.audio_weight * audio

    def route_means(
        self,
        item: LossItem,
        *,
        distributed: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Return text/audio route means with the requested reduction scope."""

        if not isinstance(distributed, bool):
            raise TypeError("distributed must be a boolean.")
        if not isinstance(item, LossItem):
            raise TypeError("MimoObjective.route_means expects a LossItem.")
        details = item.details
        required = {"text_loss", "audio_loss", "text_tokens", "audio_tokens"}
        if details is None or not required.issubset(details):
            raise ValueError("MIMO loss details are missing route losses or counts.")
        return (
            _distributed_route_mean(
                details["text_loss"], details["text_tokens"], distributed=distributed
            ),
            _distributed_route_mean(
                details["audio_loss"], details["audio_tokens"], distributed=distributed
            ),
        )

    def from_batch(
        self,
        batch: MimoBatch,
        model: Any,
        *,
        validate: bool = True,
    ) -> LossItem:
        """Evaluate a model exposing the dual-stream protocol.

        ``model.dual_hidden_states(batch)`` may return a
        :class:`DualStreamHiddenStates` value or a ``(text, audio)`` tuple.
        The model must then expose either ``dual_logits(hidden)`` returning a
        pair or separate ``text_logits``/``audio_logits`` callables.
        """

        if not isinstance(batch, MimoBatch):
            raise TypeError("MimoObjective.from_batch expects a MimoBatch.")
        if batch.ignore_index != self.ignore_index:
            raise ValueError("MimoBatch and MimoObjective must share ignore_index.")
        hidden = _call_dual_hidden(model, batch)
        text_hidden, audio_hidden = _unpack_hidden(hidden)
        text_logits, audio_logits = _call_dual_logits(model, hidden, text_hidden, audio_hidden)
        return self(
            text_logits,
            audio_logits,
            batch.text_labels,
            batch.audio_labels,
            text_loss_mask=batch.text_loss_mask,
            audio_loss_mask=batch.audio_loss_mask,
            attention_mask=batch.attention_mask,
            validate=validate,
        )


class MimoLoss(MimoObjective):
    """Descriptive alias for callers that distinguish losses from wrappers."""


def _call_dual_hidden(model: Any, batch: MimoBatch) -> Any:
    method = getattr(model, "dual_hidden_states", None)
    if not callable(method):
        method = getattr(model, "mimo_hidden_states", None)
    if not callable(method):
        raise TypeError(
            "MIMO model must expose dual_hidden_states(batch) or mimo_hidden_states(batch)."
        )
    return method(batch)


def _unpack_hidden(hidden: Any) -> tuple[Tensor, Tensor]:
    if hasattr(hidden, "text") and hasattr(hidden, "audio"):
        text_hidden = hidden.text
        audio_hidden = hidden.audio
    elif isinstance(hidden, (tuple, list)) and len(hidden) == 2:
        text_hidden, audio_hidden = hidden
    else:
        raise TypeError("dual hidden states must expose text/audio tensors.")
    if not isinstance(text_hidden, Tensor) or not isinstance(audio_hidden, Tensor):
        raise TypeError("dual hidden states text/audio values must be tensors.")
    return text_hidden, audio_hidden


def _call_dual_logits(
    model: Any,
    hidden: Any,
    text_hidden: Tensor,
    audio_hidden: Tensor,
) -> tuple[Tensor, Tensor]:
    method = getattr(model, "dual_logits", None)
    if callable(method):
        result: Any = method(hidden)
        if hasattr(result, "text") and hasattr(result, "audio"):
            text_logits = result.text
            audio_logits = result.audio
        elif isinstance(result, (tuple, list)) and len(result) == 2:
            text_logits, audio_logits = result
        else:
            raise TypeError("dual_logits must return (text_logits, audio_logits).")
    else:
        text_method = getattr(model, "text_logits", None)
        audio_method = getattr(model, "audio_logits", None)
        if not callable(text_method) or not callable(audio_method):
            raise TypeError(
                "MIMO model must expose dual_logits(hidden) or text_logits/audio_logits."
            )
        text_logits = text_method(text_hidden)
        audio_logits = audio_method(audio_hidden)
    if not isinstance(text_logits, Tensor) or not isinstance(audio_logits, Tensor):
        raise TypeError("dual logits must be tensors.")
    return text_logits, audio_logits


def _validate_logits(logits: Tensor, labels: Tensor, *, name: str) -> None:
    if not isinstance(logits, Tensor) or logits.dim() != 3:
        raise ValueError(f"{name}_logits must have shape [B, T, V].")
    if logits.size(-1) < 1:
        raise ValueError(f"{name}_logits must have a non-empty vocabulary.")
    if not logits.is_floating_point():
        raise TypeError(f"{name}_logits must use a floating-point dtype.")
    if not isinstance(labels, Tensor) or labels.dim() != 2:
        raise ValueError(f"{name}_labels must have shape [B, T].")
    if labels.shape != logits.shape[:2]:
        raise ValueError(f"{name}_logits and {name}_labels must align on [B, T].")
    if not is_signed_integer_dtype(labels.dtype):
        raise TypeError(f"{name}_labels must use a signed integer dtype.")


def _validate_hidden(hidden: Tensor, labels: Tensor, *, name: str) -> None:
    if not isinstance(hidden, Tensor) or hidden.dim() != 3:
        raise ValueError(f"{name}_hidden_states must have shape [B, T, H].")
    if not isinstance(labels, Tensor) or labels.dim() != 2:
        raise ValueError(f"{name}_labels must have shape [B, T].")
    if hidden.shape[:2] != labels.shape:
        raise ValueError(f"{name}_hidden_states and labels must align on [B, T].")


def _effective_mask(
    value: Tensor | None,
    shape: tuple[int, int],
    *,
    device: torch.device,
    name: str,
) -> Tensor:
    if value is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must use boolean dtype.")
    if value.device != device:
        raise ValueError(f"{name} must share the labels device.")
    return value


def _effective_loss_mask(
    value: Tensor | None,
    labels: Tensor,
    attention_mask: Tensor,
    *,
    name: str,
    ignore_index: int,
    validate: bool,
) -> Tensor:
    if value is None:
        mask = labels.ne(ignore_index)
    else:
        if value.shape != labels.shape:
            raise ValueError(f"{name} must align with labels.")
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must use boolean dtype.")
        if value.device != labels.device:
            raise ValueError(f"{name} must share the labels device.")
        if validate and bool((value & labels.eq(ignore_index)).any()):
            raise ValueError(f"{name} cannot select ignore-index labels.")
        mask = value
    if validate and bool((mask & ~attention_mask).any()):
        raise ValueError(f"{name} cannot select masked attention positions.")
    return mask


def _causal_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
    *,
    ignore_index: int,
    validate: bool,
    name: str,
) -> Tensor:
    target = labels[:, 1:]
    valid_target = mask
    if validate:
        invalid = valid_target & (target.lt(0) | target.ge(logits.size(-1)))
        if bool(invalid.any()):
            raise ValueError(f"{name}_labels contain an id outside the local vocabulary.")
    safe_target = torch.where(
        valid_target,
        target,
        torch.full_like(target, ignore_index),
    ).to(dtype=torch.long)
    return F.cross_entropy(
        logits[:, :-1].transpose(1, 2),
        safe_target,
        reduction="none",
        ignore_index=ignore_index,
    )


def _masked_row_mean(values: Tensor, mask: Tensor, count: Tensor) -> Tensor:
    return (values * mask.to(dtype=values.dtype)).sum(dim=1) / count.clamp_min(1).to(
        dtype=values.dtype
    )


def _route_mean(loss: Tensor, count: Tensor) -> Tensor:
    if loss.shape != count.shape or loss.dim() != 1:
        raise ValueError("MIMO route losses and counts must align on batch rows.")
    count = count.to(device=loss.device, dtype=loss.dtype)
    total = count.sum()
    value = (loss.masked_fill(count == 0, 0) * count).sum() / total.clamp_min(1)
    return torch.where(total > 0, value, loss.new_zeros(()))


def _distributed_route_mean(
    loss: Tensor,
    count: Tensor,
    *,
    distributed: bool,
) -> Tensor:
    """Reduce a route by its global token count when DDP is active."""

    local = _route_mean(loss, count)
    if not distributed:
        return local
    dist = torch.distributed
    if not dist.is_available() or not dist.is_initialized():
        return local
    weights = count.to(device=loss.device, dtype=loss.dtype)
    local_count = weights.sum()
    global_count = local_count.detach().clone()
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    if float(global_count.item()) <= 0:
        return loss.new_zeros(())
    numerator = (loss * weights).sum()
    world_size = dist.get_world_size()
    if numerator.requires_grad:
        # Lightning/DDP averages rank gradients; compensate with world size
        # while using the globally reduced denominator.
        return numerator * float(world_size) / global_count
    global_numerator = numerator.detach().clone()
    dist.all_reduce(global_numerator, op=dist.ReduceOp.SUM)
    return global_numerator / global_count


def _validate_weight(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a number.")
    if not torch.isfinite(torch.tensor(float(value))) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")


def _validate_devices(*values: Tensor) -> None:
    if len({value.device for value in values}) != 1:
        raise ValueError("MIMO logits and labels must share a device.")


__all__ = ["MimoLoss", "MimoObjective", "TensorReadout"]
