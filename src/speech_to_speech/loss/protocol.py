from __future__ import annotations

from typing import Protocol

from anydataset.types import Modality
from anytrain.loss import PackedCodebookLogits
from anytrain.module.idspace import Layout
from torch import Tensor

class TokenObjectiveModel(Protocol):
    @property
    def layout(self) -> Layout: ...

    def token_hidden_states(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        audio_input_positions: Tensor | None = None,
        embedding_blocks: frozenset[str] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
    ) -> Tensor: ...

    def token_logits(
        self,
        hidden_state: Tensor,
        modality: Modality | None = None,
        *,
        attention_mask: Tensor | None = None,
        audio_hidden_state: Tensor | None = None,
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
