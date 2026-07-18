from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from anydataset.types import Modality
from anytrain.idspace import Layout
from torch import Tensor

from ..runtime.protocol import GenerationRuntime
from ..runtime.types import Backbone
from .types import AcousticGeneration


class TokenGenerator(Protocol):
    @property
    def runtime(self) -> GenerationRuntime: ...

    @property
    def backbone(self) -> Backbone: ...

    def generate_tokens(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        acoustic_prompt_codes: Tensor | None = None,
        acoustic_prompt_positions: Tensor | None = None,
        acoustic_prompt_mask: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
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
        acoustic_prompt_codes: Tensor | None = None,
        acoustic_prompt_positions: Tensor | None = None,
        acoustic_prompt_mask: Tensor | None = None,
    ) -> Tensor: ...

    def token_logits(
        self,
        hidden_state: Tensor,
        modality: Modality | None = None,
    ) -> Tensor: ...


@runtime_checkable
class AcousticFeatureGenerator(TokenGenerator, Protocol):
    def generate_audio_features(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        acoustic_prompt_codes: Tensor | None = None,
        acoustic_prompt_positions: Tensor | None = None,
        acoustic_prompt_mask: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> AcousticGeneration: ...
