from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, TypedDict, cast

import torch
from anydataset.types import Modality
from anytrain.loss import PackedCodebookLogits
from anytrain.module.idspace import Layout
from torch import Tensor
from typing_extensions import NotRequired

from ..model.ctc import CTCRoute, ObjectiveHiddenOutput
from ..model.embedding.fsq import FsqNeighbors
from ..task import PredictionModality


class Outputs(TypedDict):
    loss: Tensor
    dpo: NotRequired[Tensor]
    grpo: NotRequired[Tensor]
    token: NotRequired[Tensor]
    ctc: NotRequired[Tensor]
    flow_matching: NotRequired[Tensor]
    repa: NotRequired[Tensor]
    rvq: NotRequired[Tensor]
    mimo: NotRequired[Tensor]


class TokenObjectiveModel(Protocol):
    @property
    def layout(self) -> Layout: ...

    def token_hidden_states(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        audio_input_positions: Tensor | None = None,
        input_modalities: frozenset[Modality] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
        prediction: PredictionModality | None = None,
    ) -> Tensor: ...

    def objective_hidden_output(
        self,
        input_ids: Tensor,
        *,
        ctc_routes: frozenset[CTCRoute],
        attention_mask: Tensor | None = None,
        audio_input_positions: Tensor | None = None,
        input_modalities: frozenset[Modality] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
        prediction: PredictionModality | None = None,
    ) -> ObjectiveHiddenOutput: ...

    def ctc_logits(
        self,
        route: CTCRoute,
        hidden_states: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]: ...

    def token_logits(
        self,
        hidden_state: Tensor,
        modality: Modality | None = None,
        *,
        attention_mask: Tensor | None = None,
        audio_hidden_state: Tensor | None = None,
    ) -> Tensor: ...

    def text_logits(
        self,
        hidden_state: Tensor,
        local_ids: Tensor | None = None,
    ) -> Tensor: ...

    def project_audio_hidden(
        self,
        hidden_state: Tensor,
        *,
        attention_mask: Tensor | None = None,
        selection_mask: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, object | None]: ...

    def audio_neighbor_targets(self, local_ids: Tensor) -> FsqNeighbors | None: ...


class AcousticDecoder(Protocol):
    def __call__(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor,
        validate: bool = True,
    ) -> Tensor: ...

    def forward_with_features(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor,
        validate: bool = True,
    ) -> tuple[Tensor, Tensor]: ...


class FlowObjectiveModel(TokenObjectiveModel, Protocol):
    @property
    def acoustic_decoder(self) -> AcousticDecoder: ...

    def target_frame_condition(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
    ) -> Tensor: ...

    def acoustic_target_latent(self, target_acoustic_codes: Tensor) -> Tensor: ...


class RVQObjectiveModel(TokenObjectiveModel, Protocol):
    def target_frame_condition(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
    ) -> Tensor: ...

    def acoustic_logits(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
        target_acoustic_codes: Tensor | None = None,
    ) -> tuple[Tensor, ...]: ...

    def acoustic_packed_logits(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
        target_acoustic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        validate: bool = True,
    ) -> PackedCodebookLogits: ...


_UNITS = {
    "dpo": "preferences",
    "grpo": "preferences",
    "token": "tokens",
    "ctc": "sequences",
    "flow_matching": "frames",
    "repa": "frames",
    "rvq": "frames",
    "mimo": "tokens",
}
_OBJECTIVES = tuple(_UNITS)


def combine_outputs(
    outputs: Sequence[Outputs],
    *,
    total_loss: Tensor | None = None,
    weights: Sequence[float] | None = None,
) -> Outputs:
    if not outputs:
        raise ValueError("cannot combine empty loss outputs.")
    resolved_weights = _weights(len(outputs), weights)
    reference = _scalar(outputs[0]["loss"], "loss")
    coefficients = reference.new_tensor(resolved_weights)
    denominator = coefficients.sum()
    result: dict[str, Tensor] = {
        "loss": (
            _weighted(outputs, "loss", coefficients, denominator, reference)
            if total_loss is None
            else _scalar(total_loss, "total_loss")
        )
    }
    for name in _OBJECTIVES:
        if any(name in output for output in outputs):
            result[name] = _weighted(
                outputs,
                name,
                coefficients,
                denominator,
                reference,
            )
    return cast(Outputs, result)


def _weighted(
    outputs: Sequence[Outputs],
    name: str,
    coefficients: Tensor,
    denominator: Tensor,
    reference: Tensor,
) -> Tensor:
    values = torch.stack(
        [
            _scalar(output[name], name) if name in output else reference.new_zeros(())
            for output in outputs
        ]
    )
    weights = coefficients.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / denominator.to(
        device=values.device,
        dtype=values.dtype,
    )


def _weights(size: int, values: Sequence[float] | None) -> tuple[float, ...]:
    if values is None:
        return (1.0,) * size
    resolved = tuple(float(value) for value in values)
    if len(resolved) != size:
        raise ValueError("loss weights must align with outputs.")
    if any(not math.isfinite(value) or value < 0 for value in resolved):
        raise ValueError("loss weights must be finite and non-negative.")
    if math.fsum(resolved) <= 0:
        raise ValueError("loss weights must have a positive total.")
    return resolved


def _scalar(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 0:
        raise TypeError(f"loss output {name!r} must be a scalar Tensor.")
    return value
