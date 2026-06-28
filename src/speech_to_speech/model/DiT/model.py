from __future__ import annotations

from typing import Unpack, cast

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils.generic import TransformersKwargs

from ..qwen3 import Qwen3Config, Qwen3RotaryEmbedding
from .module import DiTLayer


class DiT(nn.Module):
    """Wrapper around the acoustic decoder that owns conditioning and CFG policy."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()

        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "eager"
        self.null_acoustic_condition = nn.Parameter(torch.zeros(1, config.hidden_size))

        self.config = config

        self.layers = nn.ModuleList(
            [
                DiTLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in config.layer_types  # type: ignore

    def _fuse_condition(
        self,
        last_hidden_state: torch.Tensor,
        timesteps: torch.Tensor,
        acoustic_condition: torch.Tensor,
    ):
        time_emb = _timestep_embedding(
            timesteps, dim=self.config.hidden_size, max_period=10000.0
        ).unsqueeze(1)
        acoustic_condition = acoustic_condition.unsqueeze(1)
        return time_emb + acoustic_condition + last_hidden_state

    def forward(
        self,
        x_t: torch.Tensor,
        last_hidden_state: torch.Tensor,
        timesteps: torch.Tensor,
        acoustic_condition: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],  # type: ignore
    ) -> BaseModelOutputWithPast:

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(x_t.shape[1], device=x_t.device) + past_seen_tokens
            position_ids = cache_position.unsqueeze(0)  # type: ignore
        else:
            cache_position = position_ids.reshape(-1)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": x_t,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        position_embeddings = self.rotary_emb(x_t, position_ids)
        condition = self._fuse_condition(
            last_hidden_state,
            timesteps,
            acoustic_condition,
        )

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            decoder_layer = cast(DiTLayer, decoder_layer)
            x_t = decoder_layer.forward(
                x_t,
                condition=condition,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],  # type: ignore
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,  # type: ignore
            )

        return BaseModelOutputWithPast(
            last_hidden_state=x_t,  # type: ignore
            past_key_values=past_key_values if use_cache else None,
        )


def _timestep_embedding(
    timesteps: Tensor,
    *,
    dim: int,
    max_period: float,
) -> Tensor:
    half = dim // 2
    if half == 0:
        return timesteps.unsqueeze(-1)
    exponent = -torch.log(
        torch.tensor(max_period, device=timesteps.device, dtype=timesteps.dtype)
    )
    exponent = (
        exponent
        * torch.arange(half, device=timesteps.device, dtype=timesteps.dtype)
        / half
    )
    freqs = torch.exp(exponent)
    args = timesteps.unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
