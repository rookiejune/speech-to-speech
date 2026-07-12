from __future__ import annotations

from typing import Any, Protocol

from anytrain.idspace import Layout
from torch import Tensor
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast


class BaseModel(Protocol):
    layout: Layout

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


class FlowMatching(BaseModel, Protocol):
    acoustic_decoder: nn.Module

    def target_frame_condition(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
    ) -> Tensor: ...

    def target_frame_label_condition(
        self,
        labels: Tensor,
        target_positions: Tensor,
    ) -> Tensor: ...

    def acoustic_target_latent(self, acoustic_labels: Tensor) -> Tensor: ...

    def sample_acoustic(self, condition: Tensor) -> Tensor: ...

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
    ) -> tuple[Tensor, Tensor, Tensor]: ...


class CausalLM(BaseModel, Protocol):
    pass
