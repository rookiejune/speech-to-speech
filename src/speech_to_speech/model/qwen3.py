from __future__ import annotations

from typing import Protocol

from transformers.models.qwen3 import Qwen3Config, Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)


class Qwen3DecoderConfigSource(Protocol):
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int | None
    num_key_value_heads: int | None


def decoder_config(
    source: Qwen3DecoderConfigSource,
    *,
    num_hidden_layers: int | None = None,
) -> Qwen3Config:
    config = Qwen3Config()
    config.hidden_size = source.hidden_size
    config.intermediate_size = source.intermediate_size
    config.num_hidden_layers = (
        source.num_hidden_layers if num_hidden_layers is None else num_hidden_layers
    )
    if source.num_attention_heads is not None:
        config.num_attention_heads = source.num_attention_heads
    if source.num_key_value_heads is not None:
        config.num_key_value_heads = source.num_key_value_heads
    if getattr(config, "_attn_implementation", None) is None:
        config._attn_implementation = "eager"
    return config


__all__ = [
    "Qwen3Attention",
    "Qwen3Config",
    "Qwen3DecoderConfigSource",
    "Qwen3DecoderLayer",
    "Qwen3MLP",
    "Qwen3Model",
    "Qwen3RMSNorm",
    "Qwen3RotaryEmbedding",
    "decoder_config",
]
