from __future__ import annotations

from typing import TypedDict

from anytrain.idspace import Layout

from ..datamodule.types import ModelBatch
from .protocol import (
    FlowObjectiveModel,
    RVQObjectiveModel,
    TokenObjectiveModel,
)
from .causal_lm import CausalAcousticLoss
from .flow_matching import AcousticFlowLoss, FlowRuntime
from .objective import Objective
from .repa import RepaLoss, Teacher
from .token import TokenLoss
from .types import Outputs


class RepaConfig(TypedDict):
    weight: float
    teacher: Teacher


class TokenObjective(Objective[TokenObjectiveModel]):
    def __init__(self, layout: Layout) -> None:
        super().__init__()
        self.layout = layout
        self.token = TokenLoss(layout)

    def forward(self, batch: ModelBatch, model: TokenObjectiveModel) -> Outputs:
        if model.layout.blocks != self.layout.blocks:
            raise ValueError("model and loss must use the same runtime layout.")
        prompt = batch.acoustic_prompt
        hidden_states = model.token_hidden_states(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            acoustic_prompt_codes=None if prompt is None else prompt["codes"],
            acoustic_prompt_positions=None
            if prompt is None
            else prompt["token_positions"],
            acoustic_prompt_mask=batch.acoustic_prompt_mask,
        )
        token = self.token(
            hidden_states,
            batch.token_labels,
            batch.tasks[0].target_modality,
            model.token_logits,
        )
        return {"loss": token.loss.mean(), "token": token}


class FlowObjective(Objective[FlowObjectiveModel]):
    def __init__(
        self,
        layout: Layout,
        flow_runtime: FlowRuntime,
        *,
        repa: RepaConfig | None = None,
    ) -> None:
        super().__init__()
        if repa is not None and repa["weight"] <= 0:
            raise ValueError("REPA weight must be positive")
        self.layout = layout
        self.token = TokenLoss(layout)
        self.flow_matching = AcousticFlowLoss()
        self.repa_loss = RepaLoss()
        self.repa_teacher = None if repa is None else repa["teacher"]
        self.flow_runtime = flow_runtime
        self.repa_weight = None if repa is None else repa["weight"]

    def forward(self, batch: ModelBatch, model: FlowObjectiveModel) -> Outputs:
        if model.layout.blocks != self.layout.blocks:
            raise ValueError("model and loss must use the same runtime layout.")
        prompt = batch.acoustic_prompt
        target_data = batch.acoustic_target
        hidden_states = model.token_hidden_states(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            acoustic_prompt_codes=None if prompt is None else prompt["codes"],
            acoustic_prompt_positions=None
            if prompt is None
            else prompt["token_positions"],
            acoustic_prompt_mask=batch.acoustic_prompt_mask,
        )
        token = self.token(
            hidden_states,
            batch.token_labels,
            batch.tasks[0].target_modality,
            model.token_logits,
        )
        result: Outputs = {"loss": token.loss.mean(), "token": token}

        if target_data is not None:
            if batch.acoustic_target_mask is None:
                raise RuntimeError(
                    "model batch did not produce an acoustic target mask."
                )
            condition = model.target_frame_condition(
                hidden_states, target_data["token_positions"]
            )
            target = model.acoustic_target_latent(target_data["codes"])
            if self.repa_weight is None:
                acoustic = self.flow_matching(
                    model.acoustic_decoder,
                    condition,
                    target,
                    batch.acoustic_target_mask,
                    self.flow_runtime,
                )
            else:
                if self.repa_teacher is None:
                    raise RuntimeError(
                        "REPA requires a teacher and target semantic codes"
                    )
                acoustic, representation = self.flow_matching.forward_with_features(
                    model.acoustic_decoder,
                    condition,
                    target,
                    batch.acoustic_target_mask,
                    self.flow_runtime,
                )
                teacher = self.repa_teacher(
                    target_data["semantic_codes"],
                    target_data["codes"],
                    batch.acoustic_target_mask,
                )
                repa = self.repa_loss(
                    representation,
                    teacher,
                    batch.acoustic_target_mask,
                )
                result["repa"] = repa
                result["loss"] = result["loss"] + self.repa_weight * repa.loss.mean()
            result["flow_matching"] = acoustic
            result["loss"] = result["loss"] + acoustic.loss.mean()
        return result


class RVQObjective(Objective[RVQObjectiveModel]):
    def __init__(self, layout: Layout) -> None:
        super().__init__()
        self.layout = layout
        self.token = TokenLoss(layout)
        self.rvq = CausalAcousticLoss()

    def forward(self, batch: ModelBatch, model: RVQObjectiveModel) -> Outputs:
        if model.layout.blocks != self.layout.blocks:
            raise ValueError("model and loss must use the same runtime layout.")
        prompt = batch.acoustic_prompt
        target_data = batch.acoustic_target
        hidden_states = model.token_hidden_states(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            acoustic_prompt_codes=None if prompt is None else prompt["codes"],
            acoustic_prompt_positions=None
            if prompt is None
            else prompt["token_positions"],
            acoustic_prompt_mask=batch.acoustic_prompt_mask,
        )
        token = self.token(
            hidden_states,
            batch.token_labels,
            batch.tasks[0].target_modality,
            model.token_logits,
        )
        result: Outputs = {"loss": token.loss.mean(), "token": token}

        if target_data is not None:
            if batch.acoustic_target_mask is None:
                raise RuntimeError(
                    "model batch did not produce an acoustic target mask."
                )
            labels = target_data["codes"]
            logits = model.acoustic_logits(
                hidden_states,
                target_data["token_positions"],
                labels.masked_fill(~batch.acoustic_target_mask[..., None], 0),
            )
            acoustic = self.rvq(
                logits,
                labels,
                batch.acoustic_target_mask,
                validate=False,
            )
            result["rvq"] = acoustic
            result["loss"] = result["loss"] + acoustic.loss.mean()
        return result
