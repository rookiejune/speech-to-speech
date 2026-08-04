from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import torch
from torch import Tensor, nn

from ..generation import AcousticGeneration
from ...runtime.codec_contract import (
    AcousticCodec,
    acoustic_codec,
)
from ...runtime.backbone.contract import Backbone
from ..base import Config, Model
from ...runtime.protocol import TokenModelRuntime


class HiddenConditionAdapter(nn.Module):
    """Map causal token-model states into the acoustic condition space."""

    def __init__(self, hidden_dim: int, condition_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0 or condition_dim <= 0:
            raise ValueError("hidden and acoustic condition dimensions must be positive.")
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, condition_dim)
        if hidden_dim == condition_dim:
            with torch.no_grad():
                self.projection.weight.copy_(torch.eye(hidden_dim))
                self.projection.bias.zero_()

    def forward(self, hidden_state: Tensor) -> Tensor:
        if hidden_state.dim() != 3 or hidden_state.size(-1) != self.hidden_dim:
            raise ValueError("hidden state must have shape [batch, frame, hidden_dim].")
        parameter = self.projection.weight
        value = hidden_state.to(device=parameter.device, dtype=parameter.dtype)
        return self.projection(self.norm(value))


def code_features(
    codec: AcousticCodec,
    backbone: Backbone,
    codes: Tensor,
    *,
    like: Tensor | None = None,
) -> Tensor:
    features = codec.acoustic_codes_to_features(codes)
    reference = backbone.get_input_embeddings().weight if like is None else like
    return features.to(device=reference.device, dtype=reference.dtype)


class AcousticFeatureSampler(Protocol):
    def __call__(self, condition: Tensor, *, mask: Tensor) -> Tensor: ...


class AcousticModel(Model, ABC):
    """Shared hidden-state boundary for frame-aligned acoustic compositions."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        runtime: TokenModelRuntime,
        condition_dim: int | None = None,
    ) -> None:
        super().__init__(config=config, runtime=runtime)
        hidden_dim = self.backbone.config.hidden_size
        reference = self.backbone.get_input_embeddings().weight
        self.acoustic_condition = HiddenConditionAdapter(
            hidden_dim,
            hidden_dim if condition_dim is None else condition_dim,
        ).to(device=reference.device, dtype=torch.float32)

    @property
    def acoustic_codec(self) -> AcousticCodec:
        return acoustic_codec(self.runtime.codec)

    def target_frame_condition(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
    ) -> Tensor:
        condition = super().target_frame_condition(hidden_states, target_positions)
        return self.acoustic_condition(condition)

    def _decoder_input(self, value: Tensor) -> Tensor:
        parameter = next(self._decoder_module().parameters())
        return value.to(dtype=parameter.dtype)

    @abstractmethod
    def _decoder_module(self) -> nn.Module: ...

    @torch.no_grad()
    def _generate_audio_features(
        self,
        prompt_ids: Tensor,
        *,
        sample: AcousticFeatureSampler,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        prompt_attention_mask: Tensor | None,
        audio_input_positions: Tensor | None,
        do_sample: bool,
        use_cache: bool,
    ) -> AcousticGeneration:
        generated, condition, frame_mask = self.generate_audio_condition(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_attention_mask=prompt_attention_mask,
            audio_input_positions=audio_input_positions,
            do_sample=do_sample,
            use_cache=use_cache,
        )
        condition = self.acoustic_condition(condition)
        return AcousticGeneration(
            sequence=generated,
            features=sample(condition, mask=frame_mask),
            frame_counts=frame_mask.sum(dim=1),
        )


__all__ = ["AcousticFeatureSampler", "AcousticModel"]
