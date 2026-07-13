from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from anydataset.types import Modality
from anytrain.idspace import Layout
from torch import Tensor
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..runtime.types import AudioTokenizer, Backbone, Codec, TextTokenizer


class BaseModel(Protocol):
    @property
    def layout(self) -> Layout: ...

    def __call__(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        acoustic_input_ids: Tensor | None = None,
        acoustic_input_positions: Tensor | None = None,
        acoustic_input_mask: Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast: ...

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        acoustic_input_ids: Tensor | None = None,
        acoustic_input_positions: Tensor | None = None,
        acoustic_input_mask: Tensor | None = None,
        output_hidden_states: bool = False,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast: ...


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


class FlowMatching(BaseModel, Protocol):
    @property
    def acoustic_decoder(self) -> AcousticDecoder: ...

    def target_frame_condition(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
    ) -> Tensor: ...

    def acoustic_target_latent(self, acoustic_labels: Tensor) -> Tensor: ...


class FlowSample(Protocol):
    final: Tensor


class FlowSamplingRuntime(Protocol):
    def sample(
        self,
        model: nn.Module,
        x_0: Tensor,
        **model_extras: object,
    ) -> FlowSample: ...


class GenerationRuntime(Protocol):
    @property
    def layout(self) -> Layout: ...

    @property
    def text_tokenizer(self) -> TextTokenizer: ...

    @property
    def audio_tokenizer(self) -> AudioTokenizer: ...

    @property
    def codec(self) -> Codec: ...

    @property
    def eos_token_id(self) -> int: ...

    @property
    def eoa_token_id(self) -> int: ...

    @property
    def codec_audio_range(self) -> tuple[int, int]: ...

    @property
    def audio_generation_allowed_ids(self) -> tuple[int, ...]: ...

    def generation_allowed_ids(self, modality: Modality) -> tuple[int, ...]: ...

    def is_codec_audio_id(self, token_id: int) -> bool: ...


class ModelRuntime(GenerationRuntime, Protocol):
    @property
    def backbone(self) -> Backbone: ...

    @property
    def flow_matching(self) -> FlowSamplingRuntime: ...


class SemanticGeneration(BaseModel, Protocol):
    @property
    def runtime(self) -> GenerationRuntime: ...

    @property
    def backbone(self) -> Backbone: ...

    def generate_semantic(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        acoustic_input_ids: Tensor | None = None,
        acoustic_input_positions: Tensor | None = None,
        acoustic_input_mask: Tensor | None = None,
        stop_token_id: int | None = None,
        allowed_token_ids: Sequence[int] | Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> Tensor: ...


class FlowGeneration(SemanticGeneration, Protocol):

    def generate_audio(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        acoustic_input_ids: Tensor | None = None,
        acoustic_input_positions: Tensor | None = None,
        acoustic_input_mask: Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]: ...


class FlowModel(FlowMatching, FlowGeneration, Protocol):
    pass


class CausalLM(BaseModel, Protocol):
    pass
