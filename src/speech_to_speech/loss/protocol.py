from __future__ import annotations

from typing import Protocol

from anydataset.types import Modality
from anytrain.idspace import Layout
from torch import Tensor


class TokenObjectiveModel(Protocol):
    @property
    def layout(self) -> Layout: ...

    def token_hidden_states(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        acoustic_prompt_codes: Tensor | None = None,
        acoustic_prompt_positions: Tensor | None = None,
        acoustic_prompt_mask: Tensor | None = None,
    ) -> Tensor: ...

    def token_logits(
        self,
        hidden_state: Tensor,
        modality: Modality | None = None,
    ) -> Tensor: ...


class AcousticDecoder(Protocol):
    def __call__(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor: ...

    def forward_with_features(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor | None = None,
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
