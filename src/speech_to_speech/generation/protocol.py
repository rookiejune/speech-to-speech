from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import Protocol, runtime_checkable

from anydataset.types import Modality
from anytrain.idspace import Layout
from torch import Tensor

from ..runtime.types import AudioTokenizer, Backbone, Codec, TextTokenizer


class GenerationRuntime(Protocol):
    @cached_property
    def layout(self) -> Layout: ...

    @cached_property
    def text_tokenizer(self) -> TextTokenizer: ...

    @cached_property
    def audio_tokenizer(self) -> AudioTokenizer: ...

    @cached_property
    def codec(self) -> Codec: ...

    @cached_property
    def eos_token_id(self) -> int: ...

    @cached_property
    def bos_token_id(self) -> int: ...

    @property
    def eoa_token_id(self) -> int: ...

    @property
    def boa_token_id(self) -> int: ...

    @property
    def codec_audio_range(self) -> tuple[int, int]: ...

    @cached_property
    def audio_generation_allowed_ids(self) -> tuple[int, ...]: ...

    def generation_allowed_ids(self, modality: Modality) -> tuple[int, ...]: ...

    def is_codec_audio_id(self, token_id: int) -> bool: ...

    @cached_property
    def pad_token_id(self) -> int: ...


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

    def token_logits(self, hidden_state: Tensor) -> Tensor: ...


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
    ) -> tuple[Tensor, Tensor]: ...
