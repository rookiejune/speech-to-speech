from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, TypeVar, TypedDict

from anytrain.loss import (
    MaskedCodebookCrossEntropyLoss,
    MaskedCosineAlignmentLoss,
    loss_item_mean,
)
from anytrain.module.idspace import Layout
from semantic_acoustic_codec.loss.flow import FlowLoss, FlowRuntime
from semantic_acoustic_codec.loss.repa import Teacher
from torch import Tensor, nn

from ..datamodule.types import ModelBatch
from .protocol import (
    FlowObjectiveModel,
    RVQObjectiveModel,
    TokenObjectiveModel,
)
from .token import TokenLoss
from .types import LossItem, Outputs, combine_outputs


ModelT_contra = TypeVar("ModelT_contra", bound=TokenObjectiveModel, contravariant=True)


class Objective(nn.Module, Generic[ModelT_contra], ABC):
    @abstractmethod
    def forward(self, batch: ModelBatch, model: ModelT_contra) -> Outputs: ...

    def validation(self, batch: ModelBatch, model: ModelT_contra) -> Outputs:
        return self.forward(batch, model)

    def reduce(self, outputs: Sequence[Outputs]) -> Outputs:
        return combine_outputs(outputs)


def _weighted_mean(item: LossItem, key: str) -> Tensor:
    return loss_item_mean(
        item,
        unit=key,
        fallback_to_mean=False,
        validate=False,
    )


def _audio_hidden(
    batch: ModelBatch,
    model: TokenObjectiveModel,
    hidden_states: Tensor,
) -> Tensor | None:
    if not batch.prediction_modality.supervises_audio:
        return None
    start, end = model.layout.blocks["audio"]
    targets = batch.token_labels[:, 1:]
    target_mask = targets.ge(start) & targets.lt(end)
    selection_mask = target_mask.new_zeros(batch.token_labels.shape)
    selection_mask[:, :-1] = target_mask
    selected, _ = model.project_audio_hidden(
        hidden_states,
        attention_mask=batch.attention_mask,
        selection_mask=selection_mask,
    )
    if selected.dim() != 2:
        raise ValueError(
            "masked audio output projection must return one row per selected token."
        )
    output = selected.new_zeros((*hidden_states.shape[:2], selected.size(-1)))
    output[selection_mask] = selected
    return output


class RepaConfig(TypedDict):
    weight: float
    teacher: Teacher


def _token_forward(
    batch: ModelBatch,
    model: TokenObjectiveModel,
    token: TokenLoss,
) -> LossItem:
    hidden_states = _token_hidden_states(
        batch,
        model,
    )
    audio_hidden = _audio_hidden(batch, model, hidden_states)
    return token(
        hidden_states,
        batch.token_labels,
        batch.prediction_modality,
        model.token_logits,
        audio_hidden_states=audio_hidden,
        attention_mask=batch.attention_mask,
        validate=False,
    )


def _token_hidden_states(
    batch: ModelBatch,
    model: TokenObjectiveModel,
) -> Tensor:
    kwargs: dict[str, Any] = {}
    if batch.embedding_blocks is not None:
        kwargs["embedding_blocks"] = batch.embedding_blocks
        kwargs["validate_input"] = False
        kwargs["validate_audio_input_positions"] = (
            not batch.audio_input_positions_validated
        )
    return model.token_hidden_states(
        batch.input_ids,
        attention_mask=batch.attention_mask,
        audio_input_positions=batch.audio_input_positions,
        **kwargs,
    )


class TokenObjective(Objective[TokenObjectiveModel]):
    def __init__(
        self,
        layout: Layout,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.token = TokenLoss(layout)

    def forward(self, batch: ModelBatch, model: TokenObjectiveModel) -> Outputs:
        if model.layout.blocks != self.layout.blocks:
            raise ValueError("model and loss must use the same runtime layout.")
        token = _token_forward(batch, model, self.token)
        return {"loss": _weighted_mean(token, "tokens"), "token": token}


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
        self.flow_matching = FlowLoss()
        self.repa_loss = MaskedCosineAlignmentLoss()
        self.repa_teacher = None if repa is None else repa["teacher"]
        self.flow_runtime = flow_runtime
        self.repa_weight = None if repa is None else repa["weight"]

    def forward(self, batch: ModelBatch, model: FlowObjectiveModel) -> Outputs:
        if model.layout.blocks != self.layout.blocks:
            raise ValueError("model and loss must use the same runtime layout.")
        target_data = batch.acoustic_target
        if (
            target_data is None
            and batch.prediction_modality.supervises_audio
        ):
            raise ValueError(
                "FlowObjective requires acoustic target data for audio-supervised batches."
            )
        hidden_states = _token_hidden_states(batch, model)
        audio_hidden = _audio_hidden(batch, model, hidden_states)
        token = self.token(
            hidden_states,
            batch.token_labels,
            batch.prediction_modality,
            model.token_logits,
            audio_hidden_states=audio_hidden,
            attention_mask=batch.attention_mask,
            validate=False,
        )
        result: Outputs = {"loss": _weighted_mean(token, "tokens"), "token": token}

        if target_data is not None:
            target_mask = batch.acoustic_target_mask
            if target_mask is None:
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
                    target_mask,
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
                    target_mask,
                    self.flow_runtime,
                )
                teacher = self.repa_teacher(
                    target_data["semantic_codes"],
                    target_data["codes"],
                    target_mask,
                )
                repa = self.repa_loss(
                    representation,
                    teacher,
                    target_mask,
                )
                result["repa"] = repa
                result["loss_weights"] = {"repa": self.repa_weight}
                result["loss"] = result["loss"] + self.repa_weight * _weighted_mean(
                    repa, "frames"
                )
            result["flow_matching"] = acoustic
            result["loss"] = result["loss"] + _weighted_mean(acoustic, "frames")
        return result


class RVQObjective(Objective[RVQObjectiveModel]):
    def __init__(
        self,
        layout: Layout,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.token = TokenLoss(layout)
        self.rvq = MaskedCodebookCrossEntropyLoss()

    def forward(self, batch: ModelBatch, model: RVQObjectiveModel) -> Outputs:
        return self._outputs(batch, model, include_top1=False)

    def validation(self, batch: ModelBatch, model: RVQObjectiveModel) -> Outputs:
        return self._outputs(batch, model, include_top1=True)

    def _outputs(
        self,
        batch: ModelBatch,
        model: RVQObjectiveModel,
        *,
        include_top1: bool,
    ) -> Outputs:
        if model.layout.blocks != self.layout.blocks:
            raise ValueError("model and loss must use the same runtime layout.")
        target_data = batch.acoustic_target
        if (
            target_data is None
            and batch.prediction_modality.supervises_audio
        ):
            raise ValueError(
                "RVQObjective requires acoustic target data for audio-supervised batches."
            )
        hidden_states = _token_hidden_states(batch, model)
        audio_hidden = _audio_hidden(batch, model, hidden_states)
        token = self.token(
            hidden_states,
            batch.token_labels,
            batch.prediction_modality,
            model.token_logits,
            audio_hidden_states=audio_hidden,
            attention_mask=batch.attention_mask,
            validate=False,
        )
        result: Outputs = {"loss": _weighted_mean(token, "tokens"), "token": token}

        if target_data is not None:
            target_mask = batch.acoustic_target_mask
            if target_mask is None:
                raise RuntimeError(
                    "model batch did not produce an acoustic target mask."
                )
            labels = target_data["codes"]
            packed = model.acoustic_packed_logits(
                hidden_states,
                target_data["token_positions"],
                labels,
                mask=target_mask,
                validate=False,
            )
            acoustic = self.rvq.forward_packed(
                packed,
                validate=False,
                include_top1=include_top1,
            )
            result["rvq"] = acoustic
            result["loss"] = result["loss"] + _weighted_mean(acoustic, "frames")
        return result
