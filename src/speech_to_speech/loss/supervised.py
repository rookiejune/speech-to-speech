from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence, Set
from typing import Any, Generic, TypeVar, TypedDict

import torch
from anydataset.types import Modality
from anytrain.evaluator.weighted import Metric
from anytrain.loss import (
    LossItem,
    MaskedCodebookCrossEntropyLoss,
    MaskedCosineAlignmentLoss,
    loss_item_mean,
)
from anytrain.module.idspace import Layout
from semantic_acoustic_generator.loss.flow import FlowLoss, FlowRuntime
from semantic_acoustic_generator.loss.repa import Teacher
from torch import Tensor, nn

from .._tensor import is_signed_integer_dtype
from ..datamodule.batch import ModelBatch
from ..model.ctc import CTCRoute, ObjectiveHiddenOutput
from ..model.embedding.fsq import FsqNeighbors
from ..task import PredictionModality
from .contract import (
    FlowObjectiveModel,
    Outputs,
    RVQObjectiveModel,
    TokenObjectiveModel,
    combine_outputs,
    loss_items,
    loss_unit,
)
from .ctc import CTCAlignmentLoss, CTCConfig


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


class TokenLoss(nn.Module):
    def __init__(
        self,
        layout: Layout,
        *,
        audio_neighbor_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if (
            isinstance(audio_neighbor_smoothing, bool)
            or not isinstance(audio_neighbor_smoothing, (int, float))
            or not 0 <= audio_neighbor_smoothing < 1
        ):
            raise ValueError("audio neighbor smoothing must be in [0, 1).")
        self.layout = layout
        self.audio_neighbor_smoothing = float(audio_neighbor_smoothing)

    def forward(
        self,
        hidden_states: Tensor,
        token_labels: Tensor,
        prediction: PredictionModality | Modality,
        token_logits: Callable[..., Tensor],
        *,
        audio_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
        audio_neighbors: Callable[[Tensor], FsqNeighbors | None] | None = None,
        validate: bool = True,
    ) -> LossItem:
        if hidden_states.dim() != 3 or token_labels.dim() != 2:
            raise ValueError(
                "token hidden states and labels must have shapes [B, T, H] and [B, T]."
            )
        if hidden_states.shape[:2] != token_labels.shape:
            raise ValueError("token hidden states and labels must align on sequence.")
        if not is_signed_integer_dtype(token_labels.dtype):
            raise TypeError("token labels must use a signed integer dtype.")
        if not isinstance(validate, bool):
            raise TypeError("validate must be a boolean.")
        modalities = _modalities(prediction)
        target = token_labels[:, 1:]
        prediction_states = hidden_states[:, :-1]

        valid = target.ne(-100)
        modality_mask = torch.zeros_like(valid)
        for modality in modalities:
            start, end = self.layout.blocks[modality.value]
            modality_mask |= target.ge(start) & target.lt(end)
        if validate:
            invalid = torch.stack(
                (
                    (valid & ~modality_mask).any(),
                    ~valid.any(dim=1).all(),
                )
            )
            if bool(invalid.any()):
                if bool(invalid[0]):
                    names = ", ".join(sorted(modality.value for modality in modalities))
                    raise ValueError(
                        f"labels contain an id outside the supervised layout blocks: {names}."
                    )
                raise ValueError(
                    "each token label row must contain at least one target token."
                )
        token_loss = self._loss(
            prediction_states,
            target,
            valid,
            modalities,
            token_logits,
            audio_hidden_states=(
                None if audio_hidden_states is None else audio_hidden_states[:, :-1]
            ),
            attention_mask=(None if attention_mask is None else attention_mask[:, :-1]),
            audio_neighbors=audio_neighbors,
        )
        text_start, text_end = self.layout.blocks[Modality.TEXT.value]
        audio_start, audio_end = self.layout.blocks[Modality.AUDIO.value]
        text_mask = valid & target.ge(text_start) & target.lt(text_end)
        audio_mask = valid & target.ge(audio_start) & target.lt(audio_end)
        text_count = text_mask.sum(dim=1)
        audio_count = audio_mask.sum(dim=1)
        text_loss = (token_loss * text_mask).sum(dim=1) / text_count.clamp_min(1)
        audio_loss = (token_loss * audio_mask).sum(dim=1) / audio_count.clamp_min(1)
        total_count = text_count + audio_count
        total_loss = (token_loss * valid).sum(dim=1) / total_count.clamp_min(1)
        return LossItem(
            loss=total_loss,
            details={
                "text_loss": text_loss,
                "audio_loss": audio_loss,
                "tokens": total_count.to(dtype=hidden_states.dtype),
                "text_tokens": text_count.to(dtype=hidden_states.dtype),
                "audio_tokens": audio_count.to(dtype=hidden_states.dtype),
            },
        )

    def _loss(
        self,
        prediction_states: Tensor,
        target: Tensor,
        valid: Tensor,
        modalities: Set[Modality],
        token_logits: Callable[..., Tensor],
        *,
        audio_hidden_states: Tensor | None,
        attention_mask: Tensor | None,
        audio_neighbors: Callable[[Tensor], FsqNeighbors | None] | None,
    ) -> Tensor:
        losses = prediction_states.new_zeros(target.shape)
        for modality in sorted(modalities, key=lambda value: value.value):
            start, end = self.layout.blocks[modality.value]
            mask = valid & target.ge(start) & target.lt(end)
            selected_target = (target[mask] - start).to(dtype=torch.long)
            if modality is Modality.AUDIO and audio_hidden_states is not None:
                logits = token_logits(
                    prediction_states[mask],
                    modality,
                    audio_hidden_state=audio_hidden_states[mask],
                )
            else:
                logits = token_logits(prediction_states[mask], modality)
            if logits.shape != (selected_target.numel(), end - start):
                raise ValueError(
                    "token logits do not match selected targets and modality vocabulary."
                )
            group_loss = self._group_loss(
                logits,
                selected_target,
                modality,
                audio_neighbors,
            )
            losses[mask] = group_loss.to(dtype=losses.dtype)
        return losses

    def _group_loss(
        self,
        logits: Tensor,
        target: Tensor,
        modality: Modality,
        audio_neighbors: Callable[[Tensor], FsqNeighbors | None] | None,
    ) -> Tensor:
        if modality is not Modality.AUDIO or self.audio_neighbor_smoothing == 0:
            return nn.functional.cross_entropy(logits, target, reduction="none")
        if audio_neighbors is None:
            raise ValueError(
                "audio neighbor smoothing requires an FSQ neighbor target provider."
            )
        neighbors = audio_neighbors(target)
        if neighbors is None:
            raise ValueError(
                "audio neighbor smoothing requires a factorized FSQ embedding."
            )
        expected = (target.numel(), neighbors.token_ids.size(-1))
        if (
            neighbors.token_ids.shape != expected
            or neighbors.weights.shape != expected
            or neighbors.valid.shape != expected
        ):
            raise ValueError("FSQ neighbor targets do not align with audio targets.")

        log_probabilities = nn.functional.log_softmax(logits, dim=-1)
        hard = -log_probabilities.gather(-1, target[:, None]).squeeze(-1)
        safe_ids = neighbors.token_ids.masked_fill(~neighbors.valid, 0)
        neighbor_nll = -log_probabilities.gather(-1, safe_ids)
        weights = neighbors.weights.to(
            device=neighbor_nll.device,
            dtype=neighbor_nll.dtype,
        )
        smooth = (neighbor_nll * weights).sum(dim=-1)
        mixed = (1 - self.audio_neighbor_smoothing) * hard
        mixed = mixed + self.audio_neighbor_smoothing * smooth
        return torch.where(neighbors.valid.any(dim=-1), mixed, hard)


def _modalities(prediction: PredictionModality | Modality) -> frozenset[Modality]:
    if isinstance(prediction, PredictionModality):
        modalities = prediction.supervised_modalities()
        if not modalities:
            raise ValueError(f"prediction modality {prediction.value} has no heads.")
        return modalities
    if prediction not in {Modality.TEXT, Modality.AUDIO}:
        raise ValueError(f"unsupported token modality: {prediction.value}")
    return frozenset({prediction})


_OBJECTIVE_NAME = {
    "token": "token/loss",
    "ctc": "alignment/ctc/loss",
    "flow_matching": "acoustic/flow_matching/loss",
    "repa": "acoustic/repa/loss",
    "rvq": "acoustic/rvq/loss",
}


def validation_metrics(outputs: Outputs) -> dict[str, Metric]:
    metrics: dict[str, Metric] = {}
    for objective, item in loss_items(outputs):
        name = _OBJECTIVE_NAME[objective]
        metrics[name] = _metric(item, item.loss, loss_unit(objective))
        if objective == "rvq":
            metrics.update(_rvq_metrics(item))
    return metrics


def _rvq_metrics(item: LossItem) -> dict[str, Metric]:
    details = item.details
    if details is None:
        raise TypeError("RVQ validation requires loss details.")
    metrics = {}
    for key, values in sorted(details.items()):
        name = _rvq_name(key)
        if name is not None:
            metrics[name] = _metric(item, values, "frames")
    return metrics


def _rvq_name(key: str) -> str | None:
    parts = key.split("_")
    if len(parts) == 2 and parts[0] == "codebook" and parts[1].isdigit():
        return f"acoustic/rvq/{key}"
    if (
        len(parts) == 3
        and parts[0] == "codebook"
        and parts[1].isdigit()
        and parts[2] == "top1"
    ):
        return f"acoustic/rvq/{key}"
    return None


def _metric(item: LossItem, values: Tensor, unit: str) -> Metric:
    details = item.details
    if details is None or unit not in details:
        raise TypeError(f"validation loss item requires {unit!r} details.")
    return Metric(values, details[unit])
