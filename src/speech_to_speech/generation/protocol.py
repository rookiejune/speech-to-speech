from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from anydataset.types import Modality
from anytrain.module.idspace import Layout
from torch import Tensor
from transformers.cache_utils import Cache

from ..model.generation import GenerationStepResult, TokenKind
from ..task import PredictionModality
from ..runtime.protocol import GenerationRuntime
from ..runtime.types import Backbone
from .types import AcousticGeneration


class TokenGenerator(Protocol):
    @property
    def runtime(self) -> GenerationRuntime: ...

    @property
    def backbone(self) -> Backbone: ...

    @property
    def audio_token_frame_spans(self) -> Tensor: ...

    def generation_step(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor,
        output_hidden_states: bool,
        token_ids: Tensor | None,
        token_kind: TokenKind | None = None,
        modality: Modality | None,
        past_key_values: Cache | None,
        use_cache: bool,
        audio_input_positions: Tensor | None = None,
        audio_head_past: object | None = None,
        input_modalities: frozenset[Modality] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
    ) -> GenerationStepResult: ...

    def select_audio_head_cache(
        self,
        past_key_values: object | None,
        indices: Tensor,
    ) -> object | None: ...

    def generate_tokens(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        prompt_attention_mask: Tensor | None = None,
        audio_input_positions: Tensor | None = None,
        stop_token_id: int | None = None,
        generation_modality: Modality | None = None,
        allowed_token_ids: Sequence[int] | Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> Tensor: ...


class TextEvaluationModel(TokenGenerator, Protocol):
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

    def token_logits(
        self,
        hidden_state: Tensor,
        modality: Modality | None = None,
        *,
        attention_mask: Tensor | None = None,
        audio_hidden_state: Tensor | None = None,
    ) -> Tensor: ...


@runtime_checkable
class AcousticFeatureGeneration(Protocol):
    """Optional model capability for conditioned acoustic generation."""

    def generate_audio_features(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        prompt_attention_mask: Tensor | None = None,
        audio_input_positions: Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> AcousticGeneration: ...


class AcousticFeatureGenerator(
    TokenGenerator,
    AcousticFeatureGeneration,
    Protocol,
):
    pass
