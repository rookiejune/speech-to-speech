from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import lru_cache
from typing import cast

import torch
from torch import Tensor, nn
from transformers.cache_utils import Cache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

from ...config import AcousticAttentionMode, ConditionEncoderConfig
from ..qwen3 import Qwen3Config, Qwen3DecoderLayer, Qwen3RotaryEmbedding


class ConditionEncoder(nn.Module):
    """Lightweight temporal encoder for frame-level acoustic condition hidden states."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "eager"
        self.config = config
        self.attention_mode = AcousticAttentionMode(
            getattr(config, "attention_mode", AcousticAttentionMode.CAUSAL)
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.has_sliding_layers = "sliding_attention" in config.layer_types  # type: ignore[attr-defined]

    def forward(self, hidden_states: Tensor, *, attention_mask: Tensor) -> Tensor:
        if not self.layers:
            return hidden_states
        cache_position = torch.arange(hidden_states.size(1), device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)
        attention_mask_mapping = _attention_mask_mapping(
            mode=self.attention_mode,
            config=self.config,
            inputs=hidden_states,
            attention_mask=attention_mask.to(device=hidden_states.device, dtype=torch.long),
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
            has_sliding_layers=self.has_sliding_layers,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for i, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer = cast(Qwen3DecoderLayer, layer)
            hidden_states = layer.forward(
                hidden_states,
                attention_mask=attention_mask_mapping[self.config.layer_types[i]],  # type: ignore[index]
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
        return hidden_states


def condition_encoder_config(
    config: ConditionEncoderConfig,
    *,
    hidden_size: int,
    attention_mode: AcousticAttentionMode,
) -> Qwen3Config:
    if config.num_hidden_layers <= 0:
        raise ValueError("condition_encoder.num_hidden_layers must be positive.")
    qwen_config = Qwen3Config()
    qwen_config.hidden_size = hidden_size
    qwen_config.num_hidden_layers = config.num_hidden_layers
    qwen_config.intermediate_size = config.intermediate_size or _default_intermediate_size(
        hidden_size
    )
    qwen_config.num_attention_heads = config.num_attention_heads or _default_heads(
        hidden_size
    )
    qwen_config.num_key_value_heads = config.num_key_value_heads or qwen_config.num_attention_heads
    qwen_config.layer_types = ["full_attention"] * config.num_hidden_layers
    qwen_config.attention_mode = attention_mode
    return qwen_config


def _attention_mask_mapping(
    *,
    mode: AcousticAttentionMode,
    config: Qwen3Config,
    inputs: Tensor,
    attention_mask: Tensor | None,
    cache_position: Tensor,
    past_key_values: Cache | None,
    position_ids: Tensor,
    has_sliding_layers: bool,
) -> dict[str, Tensor | None]:
    if mode is AcousticAttentionMode.BIDIRECTIONAL:
        mask = _bidirectional_mask(inputs, attention_mask)
        return {
            "full_attention": mask,
            "sliding_attention": mask,
        }
    mapping: dict[str, Tensor | None] = {
        "full_attention": create_causal_mask(
            **_mask_kwargs(
                create_causal_mask,
                config=config,
                inputs=inputs,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        ),
    }
    if has_sliding_layers:
        mapping["sliding_attention"] = create_sliding_window_causal_mask(
            **_mask_kwargs(
                create_sliding_window_causal_mask,
                config=config,
                inputs=inputs,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        )
    return mapping


def _bidirectional_mask(inputs: Tensor, attention_mask: Tensor | None) -> Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.dim() != 2:
        raise ValueError("bidirectional condition attention_mask must have shape [batch, time].")
    if attention_mask.shape != inputs.shape[:2]:
        raise ValueError("bidirectional condition attention_mask must align with hidden states.")
    if bool(attention_mask.to(device=inputs.device, dtype=torch.bool).all()):
        return None
    keep = attention_mask.to(device=inputs.device, dtype=torch.bool)
    mask = inputs.new_zeros((inputs.size(0), 1, inputs.size(1), inputs.size(1)))
    mask = mask.masked_fill(~keep[:, None, None, :], torch.finfo(inputs.dtype).min)
    return mask


def _mask_kwargs(
    mask_fn: Callable[..., object],
    *,
    config: Qwen3Config,
    inputs: Tensor,
    attention_mask: Tensor | None,
    cache_position: Tensor,
    past_key_values: Cache | None,
    position_ids: Tensor,
) -> dict[str, object]:
    embeds_name, accepts_cache_position = _mask_signature(mask_fn)
    kwargs: dict[str, object] = {
        "config": config,
        embeds_name: inputs,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    if accepts_cache_position:
        kwargs["cache_position"] = cache_position
    return kwargs


@lru_cache(maxsize=None)
def _mask_signature(mask_fn: Callable[..., object]) -> tuple[str, bool]:
    parameters = inspect.signature(mask_fn).parameters
    if "input_embeds" in parameters:
        embeds_name = "input_embeds"
    elif "inputs_embeds" in parameters:
        embeds_name = "inputs_embeds"
    else:
        raise TypeError("Transformers mask function must accept input embeddings.")
    return embeds_name, "cache_position" in parameters


def _default_intermediate_size(hidden_size: int) -> int:
    return max(1, int(round((8.0 / 3.0) * hidden_size)))


def _default_heads(hidden_size: int) -> int:
    for heads in (8, 4, 2, 1):
        if hidden_size % heads == 0:
            return heads
    return 1


__all__ = [
    "ConditionEncoder",
    "condition_encoder_config",
]
