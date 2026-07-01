from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
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

from ...config import AcousticAttentionMode
from ..qwen3 import Qwen3Config, Qwen3RotaryEmbedding
from .module import DiTLayer


@dataclass(frozen=True)
class DiTConditionTensors:
    time: Tensor
    hidden: Tensor
    acoustic: Tensor


class DiT(nn.Module):
    """Wrapper around the acoustic decoder that owns conditioning and CFG policy."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()

        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "eager"
        self.null_acoustic_condition = nn.Parameter(torch.zeros(1, config.hidden_size))

        self.config = config
        self.attention_mode = AcousticAttentionMode(
            getattr(config, "attention_mode", AcousticAttentionMode.CAUSAL)
        )
        self.time_norm = _condition_norm(config, "norm_time")
        self.hidden_norm = _condition_norm(config, "norm_hidden")
        self.acoustic_norm = _condition_norm(config, "norm_acoustic")

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
        tensors = self.condition_tensors(
            last_hidden_state=last_hidden_state,
            timesteps=timesteps,
            acoustic_condition=acoustic_condition,
        )
        return tensors.time + tensors.acoustic + tensors.hidden

    def condition_tensors(
        self,
        *,
        last_hidden_state: Tensor,
        timesteps: Tensor,
        acoustic_condition: Tensor,
    ) -> DiTConditionTensors:
        time_emb = _timestep_embedding(
            timesteps, dim=self.config.hidden_size, max_period=10000.0
        ).unsqueeze(1)
        time_emb = self.time_norm(time_emb)
        last_hidden_state = self.hidden_norm(last_hidden_state)
        acoustic_condition = acoustic_condition.unsqueeze(1)
        acoustic_condition = self.acoustic_norm(acoustic_condition)
        return DiTConditionTensors(
            time=time_emb,
            hidden=last_hidden_state,
            acoustic=acoustic_condition,
        )

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

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(x_t.shape[1], device=x_t.device) + past_seen_tokens
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)  # type: ignore
        mask_position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if isinstance(mask_mapping := attention_mask, dict):
            attention_mask_mapping = mask_mapping
        else:
            attention_mask_mapping = _attention_mask_mapping(
                mode=self.attention_mode,
                config=self.config,
                inputs=x_t,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=mask_position_ids,
                has_sliding_layers=self.has_sliding_layers,
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
                attention_mask=attention_mask_mapping[self.config.layer_types[i]],  # type: ignore
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
    # The sliding window alternating layers are not always activated depending on the config.
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
        raise ValueError("bidirectional DiT attention_mask must have shape [batch, time].")
    if attention_mask.shape != inputs.shape[:2]:
        raise ValueError("bidirectional DiT attention_mask must align with x_t.")
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


def _condition_norm(config: Qwen3Config, name: str) -> nn.Module:
    if bool(getattr(config, name, False)):
        return nn.LayerNorm(config.hidden_size, elementwise_affine=False)
    return nn.Identity()


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
