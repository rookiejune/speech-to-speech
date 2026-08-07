from __future__ import annotations

from typing import Any

import torch
from anydataset.types import Modality
from anytrain.framework.rl import DPOLoss, GRPOLoss, sequence_logps
from torch import Tensor
from torch.nn import functional as F

from .._tensor import is_signed_integer_dtype
from ..datamodule.batch import ModelBatch
from ..model.base import Model
from ..rl.types import GRPOBatch, PreferenceBatch
from .contract import Outputs
from .supervised import Objective


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


class DPOObjective(Objective[Model]):
    def __init__(
        self,
        *,
        beta: float = 0.1,
        reference_free: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.loss = DPOLoss(
            beta=beta,
            reference_free=reference_free,
            label_smoothing=label_smoothing,
        )

    def forward(self, batch: object, model: Model) -> Outputs:
        if not isinstance(batch, PreferenceBatch):
            raise TypeError("DPOObjective requires a PreferenceBatch.")
        policy_chosen_logps, policy_rejected_logps = _preference_logps(batch, model)
        loss = self.loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=batch.ref_chosen_logps,
            ref_rejected_logps=batch.ref_rejected_logps,
            validate=False,
        )
        return {"loss": loss, "dpo": loss}


def _preference_logps(
    batch: PreferenceBatch,
    model: Model,
) -> tuple[Tensor, Tensor]:
    chosen = batch.chosen
    rejected = batch.rejected
    prediction = chosen.prediction_modality
    if rejected.prediction_modality is not prediction:
        raise ValueError("chosen and rejected batches must use the same prediction modality.")
    input_ids = torch.cat((chosen.input_ids, rejected.input_ids))
    attention_mask = torch.cat((chosen.attention_mask, rejected.attention_mask))
    response_mask = torch.cat(
        (
            chosen.token_labels[:, 1:].ne(-100),
            rejected.token_labels[:, 1:].ne(-100),
        )
    )
    hidden_states = model.token_hidden_states(
        input_ids,
        attention_mask=attention_mask,
        audio_input_positions=_audio_input_positions(chosen, rejected),
        prediction=prediction,
        **_preference_input_kwargs(chosen, rejected),
    )
    token_logps = target_token_logps(
        model,
        hidden_states,
        input_ids[:, 1:],
        response_mask,
        prediction.supervised_modalities(),
        attention_mask=attention_mask,
    )
    logps = sequence_logps(token_logps, response_mask)
    chosen_logps, rejected_logps = logps.chunk(2)
    return chosen_logps, rejected_logps


def _audio_input_positions(
    chosen: ModelBatch,
    rejected: ModelBatch,
) -> Tensor | None:
    first = chosen.audio_input_positions
    second = rejected.audio_input_positions
    if first is None and second is None:
        return None
    width = max(
        0 if first is None else first.size(1),
        0 if second is None else second.size(1),
    )
    reference = first if first is not None else second
    if reference is None:
        raise AssertionError("audio input position reference disappeared.")

    def padded(value: Tensor | None, rows: int) -> Tensor:
        if value is None:
            return reference.new_full((rows, width), -1)
        return F.pad(value, (0, width - value.size(1)), value=-1)

    return torch.cat(
        (
            padded(first, chosen.input_ids.size(0)),
            padded(second, rejected.input_ids.size(0)),
        )
    )


def _preference_input_kwargs(
    chosen: ModelBatch,
    rejected: ModelBatch,
) -> dict[str, Any]:
    first = chosen.input_modalities
    second = rejected.input_modalities
    if first is None or second is None:
        return {}
    return {
        "input_modalities": first | second,
        "validate_input": False,
        "validate_audio_input_positions": not (
            chosen.audio_input_positions_validated
            and rejected.audio_input_positions_validated
        ),
    }


class GRPOObjective(Objective[Model]):
    def __init__(
        self,
        *,
        clip_range: float = 0.2,
        kl_beta: float = 0.0,
    ) -> None:
        super().__init__()
        self.loss = GRPOLoss(clip_range=clip_range, kl_beta=kl_beta)

    def forward(self, batch: object, model: Model) -> Outputs:
        if not isinstance(batch, GRPOBatch):
            raise TypeError("GRPOObjective requires a GRPOBatch.")
        policy_token_logps = _rollout_token_logps(batch.sequences, model).view_as(
            batch.old_token_logps
        )
        response_mask = _response_mask(batch.sequences).view_as(batch.old_token_logps)
        loss = self.loss(
            policy_token_logps=policy_token_logps,
            old_token_logps=batch.old_token_logps,
            rewards=batch.rewards,
            response_mask=response_mask,
            ref_token_logps=batch.ref_token_logps,
            group_mask=batch.group_mask,
            validate=False,
        )
        return {"loss": loss, "grpo": loss}


def _rollout_token_logps(batch: ModelBatch, model: Model) -> Tensor:
    response_mask = _response_mask(batch)
    hidden_states = model.token_hidden_states(
        batch.input_ids,
        attention_mask=batch.attention_mask,
        audio_input_positions=batch.audio_input_positions,
        prediction=batch.prediction_modality,
        **_rollout_input_kwargs(batch),
    )
    return target_token_logps(
        model,
        hidden_states,
        batch.input_ids[:, 1:],
        response_mask,
        batch.prediction_modality.supervised_modalities(),
        attention_mask=batch.attention_mask,
    )


def _response_mask(batch: ModelBatch) -> Tensor:
    return batch.token_labels[:, 1:].ne(-100)


def _rollout_input_kwargs(batch: ModelBatch) -> dict[str, Any]:
    if batch.input_modalities is None:
        return {}
    return {
        "input_modalities": batch.input_modalities,
        "validate_input": False,
        "validate_audio_input_positions": not batch.audio_input_positions_validated,
    }


__all__ = ["DPOObjective", "GRPOObjective", "target_token_logps"]
