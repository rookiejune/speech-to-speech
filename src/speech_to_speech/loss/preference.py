from __future__ import annotations

from anytrain.framework.rl import DPOLoss, gather_token_logps, sequence_logps
from anytrain.loss import LossItem
from torch import Tensor

from ..datamodule.preference import PreferenceBatch
from ..datamodule.types import ModelBatch
from ..model.base import Model
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
        policy_chosen_logps = _sequence_logps(batch.chosen, model)
        policy_rejected_logps = _sequence_logps(batch.rejected, model)
        loss, details = self.loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=batch.ref_chosen_logps,
            ref_rejected_logps=batch.ref_rejected_logps,
        )
        item_details = {"preferences": loss.detach().new_ones(())}
        item_details.update(details)
        return {"loss": loss, "dpo": LossItem(loss, item_details)}


def _sequence_logps(batch: ModelBatch, model: Model) -> Tensor:
    hidden_states = model.token_hidden_states(
        batch.input_ids,
        attention_mask=batch.attention_mask,
        audio_input_positions=batch.audio_input_positions,
    )
    logits = model.token_logits(hidden_states)
    target_ids = batch.input_ids[:, 1:]
    target_logits = logits[:, :-1]
    mask = batch.token_labels[:, 1:].ne(-100)
    if not bool(mask.any(dim=1).all()):
        raise ValueError("each preference response must contain at least one token.")
    token_logps = gather_token_logps(target_logits, target_ids)
    return sequence_logps(token_logps, mask)
