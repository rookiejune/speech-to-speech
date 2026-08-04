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
from semantic_acoustic_generator.loss.flow import FlowLoss, FlowRuntime
from semantic_acoustic_generator.loss.repa import Teacher
from torch import Tensor, nn

from ..datamodule.batch import ModelBatch
from ..model.ctc import CTCRoute, ObjectiveHiddenOutput
from .ctc import CTCAlignmentLoss, CTCConfig
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


def _zero_safe_weighted_mean(item: LossItem, key: str) -> Tensor:
    details = item.details
    if details is None or key not in details:
        raise ValueError(f"loss item details must contain unit {key!r}.")
    weight = details[key].to(device=item.loss.device, dtype=item.loss.dtype)
    if weight.shape != item.loss.shape:
        raise ValueError("loss weights must align with loss rows.")
    safe_loss = item.loss.masked_fill(weight == 0, 0)
    return (safe_loss * weight).sum() / weight.sum().clamp_min(1)


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
    *,
    hidden_states: Tensor | None = None,
) -> LossItem:
    if hidden_states is None:
        hidden_states = _token_hidden_states(batch, model)
    audio_hidden = _audio_hidden(batch, model, hidden_states)
    return token(
        hidden_states,
        batch.token_labels,
        batch.prediction_modality,
        model.token_logits,
        audio_hidden_states=audio_hidden,
        attention_mask=batch.attention_mask,
        audio_neighbors=(
            model.audio_neighbor_targets
            if token.audio_neighbor_smoothing > 0
            else None
        ),
        validate=False,
    )


def _token_hidden_states(
    batch: ModelBatch,
    model: TokenObjectiveModel,
) -> Tensor:
    kwargs: dict[str, Any] = {}
    if batch.input_modalities is not None:
        kwargs["input_modalities"] = batch.input_modalities
        kwargs["validate_input"] = False
        kwargs["validate_audio_input_positions"] = (
            not batch.audio_input_positions_validated
        )
    return model.token_hidden_states(
        batch.input_ids,
        attention_mask=batch.attention_mask,
        audio_input_positions=batch.audio_input_positions,
        prediction=batch.prediction_modality,
        **kwargs,
    )


def _objective_hidden_output(
    batch: ModelBatch,
    model: TokenObjectiveModel,
    alignment: CTCAlignmentLoss | None,
) -> ObjectiveHiddenOutput:
    routes = _active_ctc_routes(batch, alignment)
    if not routes:
        return ObjectiveHiddenOutput(token=_token_hidden_states(batch, model))
    kwargs: dict[str, Any] = {}
    if batch.input_modalities is not None:
        kwargs["input_modalities"] = batch.input_modalities
        kwargs["validate_input"] = False
        kwargs["validate_audio_input_positions"] = (
            not batch.audio_input_positions_validated
        )
    return model.objective_hidden_output(
        batch.input_ids,
        ctc_routes=routes,
        attention_mask=batch.attention_mask,
        audio_input_positions=batch.audio_input_positions,
        prediction=batch.prediction_modality,
        **kwargs,
    )


def _active_ctc_routes(
    batch: ModelBatch,
    alignment: CTCAlignmentLoss | None,
) -> frozenset[CTCRoute]:
    if alignment is None:
        return frozenset()
    config = alignment.config
    return frozenset(
        route
        for route, enabled in (
            (
                CTCRoute.SOURCE,
                config.source.enabled and batch.source_ctc is not None,
            ),
            (
                CTCRoute.TARGET,
                config.target.enabled and batch.target_ctc is not None,
            ),
        )
        if enabled
    )


def _ctc_loss(
    config: CTCConfig | None,
    blank_token_id: int | None,
) -> CTCAlignmentLoss | None:
    resolved = CTCConfig() if config is None else config
    if not isinstance(resolved, CTCConfig):
        raise TypeError("CTC objective config must be a CTCConfig.")
    if not resolved.enabled:
        return None
    if blank_token_id is None:
        raise ValueError("enabled CTC objective requires a blank token id.")
    return CTCAlignmentLoss(blank_token_id, resolved)


def _add_ctc(
    outputs: Outputs,
    batch: ModelBatch,
    model: TokenObjectiveModel,
    hidden_states: ObjectiveHiddenOutput,
    alignment: CTCAlignmentLoss | None,
) -> None:
    if alignment is None:
        return
    config = alignment.config
    source = batch.source_ctc if config.source.enabled else None
    target = batch.target_ctc if config.target.enabled else None
    if source is None and target is None:
        return
    ctc = alignment(
        hidden_states.token,
        source_hidden_states=hidden_states.source_ctc,
        target_hidden_states=hidden_states.target_ctc,
        source=source,
        target=target,
        decode=model.ctc_logits,
    )
    outputs["ctc"] = ctc
    outputs["loss"] = outputs["loss"] + _zero_safe_weighted_mean(ctc, "sequences")


class TokenObjective(Objective[TokenObjectiveModel]):
    def __init__(
        self,
        layout: Layout,
        *,
        ctc: CTCConfig | None = None,
        ctc_blank_token_id: int | None = None,
        audio_neighbor_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.token = TokenLoss(
            layout,
            audio_neighbor_smoothing=audio_neighbor_smoothing,
        )
        self.ctc = _ctc_loss(ctc, ctc_blank_token_id)

    def forward(self, batch: ModelBatch, model: TokenObjectiveModel) -> Outputs:
        if model.layout.blocks != self.layout.blocks:
            raise ValueError("model and loss must use the same runtime layout.")
        hidden = _objective_hidden_output(batch, model, self.ctc)
        token = _token_forward(
            batch,
            model,
            self.token,
            hidden_states=hidden.token,
        )
        result: Outputs = {"loss": _weighted_mean(token, "tokens"), "token": token}
        _add_ctc(result, batch, model, hidden, self.ctc)
        return result


class FlowObjective(Objective[FlowObjectiveModel]):
    def __init__(
        self,
        layout: Layout,
        flow_runtime: FlowRuntime,
        *,
        repa: RepaConfig | None = None,
        ctc: CTCConfig | None = None,
        ctc_blank_token_id: int | None = None,
        audio_neighbor_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if repa is not None and repa["weight"] <= 0:
            raise ValueError("REPA weight must be positive")
        self.layout = layout
        self.token = TokenLoss(
            layout,
            audio_neighbor_smoothing=audio_neighbor_smoothing,
        )
        self.ctc = _ctc_loss(ctc, ctc_blank_token_id)
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
        hidden = _objective_hidden_output(batch, model, self.ctc)
        token = _token_forward(
            batch,
            model,
            self.token,
            hidden_states=hidden.token,
        )
        result: Outputs = {"loss": _weighted_mean(token, "tokens"), "token": token}
        _add_ctc(result, batch, model, hidden, self.ctc)

        if target_data is not None:
            target_mask = batch.acoustic_target_mask
            if target_mask is None:
                raise RuntimeError(
                    "model batch did not produce an acoustic target mask."
                )
            condition = model.target_frame_condition(
                hidden.token, target_data["token_positions"]
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
        *,
        ctc: CTCConfig | None = None,
        ctc_blank_token_id: int | None = None,
        audio_neighbor_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.token = TokenLoss(
            layout,
            audio_neighbor_smoothing=audio_neighbor_smoothing,
        )
        self.ctc = _ctc_loss(ctc, ctc_blank_token_id)
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
        hidden = _objective_hidden_output(batch, model, self.ctc)
        token = _token_forward(
            batch,
            model,
            self.token,
            hidden_states=hidden.token,
        )
        result: Outputs = {"loss": _weighted_mean(token, "tokens"), "token": token}
        _add_ctc(result, batch, model, hidden, self.ctc)

        if target_data is not None:
            target_mask = batch.acoustic_target_mask
            if target_mask is None:
                raise RuntimeError(
                    "model batch did not produce an acoustic target mask."
                )
            labels = target_data["codes"]
            packed = model.acoustic_packed_logits(
                hidden.token,
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
