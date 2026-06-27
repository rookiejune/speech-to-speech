from __future__ import annotations

from typing import Unpack

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.utils.generic import TransformersKwargs

from ..qwen3 import Qwen3Attention, Qwen3Config, Qwen3MLP


class AdaLN(nn.Module):
    def __init__(
        self,
        hidden_size: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = max(1, int(round((8.0 / 3.0) * hidden_size)))
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size * 6, bias=True
        )
        self.act_fn = nn.SiLU()

        nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)

    def forward(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj.chunk(6, dim=-1)


def _apply_adaln(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    return x * (1.0 + scale) + shift


class DiTLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ) -> None:
        super().__init__()

        self.adaln = AdaLN(config.hidden_size)
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Qwen3MLP(config)

        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, elementwise_affine=False
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            elementwise_affine=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],  # type: ignore
    ) -> torch.Tensor:
        residual = hidden_states
        params = self.adaln(condition)
        scale_0, shift_0, gamma_0, scale_1, shift_1, gamma_1 = params

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = _apply_adaln(hidden_states, scale_0, shift_0)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,  # type: ignore
        )
        hidden_states = residual + hidden_states * gamma_0

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = _apply_adaln(hidden_states, scale_1, shift_1)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * gamma_1
        return hidden_states
