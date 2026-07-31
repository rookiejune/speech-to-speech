from __future__ import annotations

from anytrain.framework.rl import GRPOLoss, gather_token_logps
from anytrain.loss import LossItem
from torch import Tensor

from ..datamodule.rollout import GRPOBatch
from ..datamodule.types import ModelBatch
from ..model.base import Model
from .module import Objective
from .types import Outputs


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
        policy_token_logps = _token_logps(batch.sequences, model).view_as(
            batch.old_token_logps
        )
        response_mask = _response_mask(batch.sequences).view_as(batch.old_token_logps)
        loss, details = self.loss(
            policy_token_logps=policy_token_logps,
            old_token_logps=batch.old_token_logps,
            rewards=batch.rewards,
            response_mask=response_mask,
            ref_token_logps=batch.ref_token_logps,
            group_mask=batch.group_mask,
        )
        item_details = {"preferences": loss.detach().new_ones(())}
        item_details.update(details)
        return {"loss": loss, "grpo": LossItem(loss, item_details)}


def _token_logps(batch: ModelBatch, model: Model) -> Tensor:
    hidden_states = model.token_hidden_states(
        batch.input_ids,
        attention_mask=batch.attention_mask,
        audio_input_positions=batch.audio_input_positions,
    )
    logits = model.token_logits(hidden_states)
    return gather_token_logps(logits[:, :-1], batch.input_ids[:, 1:])


def _response_mask(batch: ModelBatch) -> Tensor:
    return batch.token_labels[:, 1:].ne(-100)
