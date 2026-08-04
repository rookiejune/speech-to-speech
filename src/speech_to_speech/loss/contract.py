from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol, TypedDict, cast

from anydataset.types import Modality
from anytrain.loss import (
    LossItem,
    PackedCodebookLogits,
    combine_loss_outputs,
    iter_loss_items,
)
from anytrain.module.idspace import Layout
from torch import Tensor
from typing_extensions import NotRequired

from ..model.ctc import CTCRoute, ObjectiveHiddenOutput
from ..model.embedding.fsq import FsqNeighbors
from ..task import PredictionModality


class Outputs(TypedDict):
    loss: Tensor
    dpo: NotRequired[LossItem]
    grpo: NotRequired[LossItem]
    token: NotRequired[LossItem]
    ctc: NotRequired[LossItem]
    flow_matching: NotRequired[LossItem]
    repa: NotRequired[LossItem]
    rvq: NotRequired[LossItem]
    mimo: NotRequired[LossItem]
    loss_weights: NotRequired[dict[str, float]]


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
) -> Outputs:
    generic_outputs = cast(Sequence[dict[str, Any]], outputs)
    return cast(
        Outputs,
        combine_loss_outputs(
            generic_outputs,
            _UNITS,
            validate_item_weights=False,
            total_loss=total_loss,
        ),
    )


def loss_items(outputs: Mapping[str, Any]) -> Iterator[tuple[str, LossItem]]:
    yield from iter_loss_items(outputs, _OBJECTIVES)


def loss_unit(name: str) -> str:
    try:
        return _UNITS[name]
    except KeyError as error:
        raise ValueError(f"unsupported loss objective: {name}") from error
