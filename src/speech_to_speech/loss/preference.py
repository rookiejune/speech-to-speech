from __future__ import annotations

import torch
import torch.nn.functional as F
from anytrain.framework.rl import DPOLoss, sequence_logps
from anytrain.loss import LossItem
from typing import Any

from torch import Tensor

from ..datamodule.preference import PreferenceBatch
from ..datamodule.types import ModelBatch
from ..model.base import Model
from .logprob import target_token_logps
from .module import Objective
from .types import Outputs


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
        loss, details = self.loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=batch.ref_chosen_logps,
            ref_rejected_logps=batch.ref_rejected_logps,
            validate=False,
        )
        item_details = {"preferences": loss.detach().new_ones(())}
        item_details.update(details)
        return {"loss": loss, "dpo": LossItem(loss, item_details)}


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
        **_input_kwargs(chosen, rejected),
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


def _input_kwargs(
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
