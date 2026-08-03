from __future__ import annotations

import torch
from anydataset.types import Modality
from torch import Tensor
from torch.nn import functional as F

from .._tensor import is_signed_integer_dtype
from ..model.base import Model


def target_token_logps(
    model: Model,
    hidden_states: Tensor,
    target_ids: Tensor,
    mask: Tensor,
    modalities: frozenset[Modality],
    *,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Score selected next-token targets against their modality-local heads."""
    if hidden_states.dim() != 3 or target_ids.dim() != 2:
        raise ValueError(
            "token hidden states and targets must have shapes [B, T, H] and [B, T-1]."
        )
    if hidden_states.shape[:2] != (target_ids.size(0), target_ids.size(1) + 1):
        raise ValueError("token hidden states and next-token targets must align.")
    if not is_signed_integer_dtype(target_ids.dtype):
        raise TypeError("target token ids must use a signed integer dtype.")
    if mask.dtype != torch.bool:
        raise TypeError("target token mask must be boolean.")
    if mask.shape != target_ids.shape:
        raise ValueError("target token mask must align with target ids.")
    if target_ids.device != hidden_states.device or mask.device != target_ids.device:
        raise ValueError("token hidden states, targets, and mask must share a device.")
    if attention_mask is not None and attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention mask must align with token hidden states.")
    if not modalities:
        raise ValueError("target token log-probs require at least one modality.")

    prediction_states = hidden_states[:, :-1]
    groups: list[tuple[Tensor, Tensor]] = []
    for modality in sorted(modalities, key=lambda value: value.value):
        start, end = model.layout.blocks[modality.value]
        modality_mask = mask & target_ids.ge(start) & target_ids.lt(end)
        local_ids = (target_ids[modality_mask] - start).to(dtype=torch.long)
        if modality is Modality.AUDIO:
            selection_mask = modality_mask.new_zeros(hidden_states.shape[:2])
            selection_mask[:, :-1] = modality_mask
            audio_hidden, _ = model.project_audio_hidden(
                hidden_states,
                attention_mask=attention_mask,
                selection_mask=selection_mask,
            )
            logits = model.token_logits(
                prediction_states[modality_mask],
                modality,
                audio_hidden_state=audio_hidden,
            )
        else:
            logits = model.token_logits(prediction_states[modality_mask], modality)
        expected_shape = (local_ids.numel(), end - start)
        if logits.shape != expected_shape:
            raise ValueError(
                "token logits do not match selected targets and modality vocabulary."
            )
        logps = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        groups.append(
            (modality_mask, logps.gather(-1, local_ids[:, None]).squeeze(-1))
        )

    dtype = groups[0][1].dtype
    for _, values in groups[1:]:
        dtype = torch.promote_types(dtype, values.dtype)
    output = torch.zeros(target_ids.shape, device=target_ids.device, dtype=dtype)
    for group_mask, values in groups:
        output[group_mask] = values.to(dtype=dtype)
    return output


__all__ = ["target_token_logps"]
