from __future__ import annotations

from typing import Any

from anytrain.framework.rl import GRPOLoss
from anytrain.loss import LossItem
from torch import Tensor

from ..datamodule.rollout import GRPOBatch
from ..datamodule.types import ModelBatch
from ..model.base import Model
from .logprob import target_token_logps
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
            validate=False,
        )
        item_details = {"preferences": loss.detach().new_ones(())}
        item_details.update(details)
        return {"loss": loss, "grpo": LossItem(loss, item_details)}


def _token_logps(batch: ModelBatch, model: Model) -> Tensor:
    response_mask = _response_mask(batch)
    hidden_states = model.token_hidden_states(
        batch.input_ids,
        attention_mask=batch.attention_mask,
        audio_input_positions=batch.audio_input_positions,
        prediction=batch.prediction_modality,
        **_embedding_kwargs(batch),
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


def _embedding_kwargs(batch: ModelBatch) -> dict[str, Any]:
    if batch.embedding_blocks is None:
        return {}
    return {
        "embedding_blocks": batch.embedding_blocks,
        "validate_input": False,
        "validate_audio_input_positions": not batch.audio_input_positions_validated,
    }
