from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import torch
from torch import Tensor, nn
from transformers import Qwen3Config, Qwen3Model

from ..runtime.backbone import BackboneReadout
from ..runtime.backbone.contract import Backbone
from .mimo import MimoModel, MimoModelConfig


@dataclass(frozen=True)
class ToyConfig:
    """Random tiny backbone settings for model and training contract tests."""

    hidden_size: int = 32
    intermediate_size: int = 64
    layers: int = 1
    heads: int = 2
    max_position_embeddings: int = 256

    def __post_init__(self) -> None:
        values = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "layers": self.layers,
            "heads": self.heads,
            "max_position_embeddings": self.max_position_embeddings,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"toy model {name} must be an integer.")
            if value <= 0:
                raise ValueError(f"toy model {name} must be positive.")
        if self.hidden_size % self.heads != 0:
            raise ValueError("toy model hidden_size must be divisible by heads.")


def create_toy_backbone(config: ToyConfig, text_vocab_size: int) -> Backbone:
    if text_vocab_size <= 0:
        raise ValueError("toy model text vocabulary must be positive.")
    backbone = Qwen3Model(
        Qwen3Config(
            vocab_size=text_vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.layers,
            num_attention_heads=config.heads,
            num_key_value_heads=config.heads,
            head_dim=config.hidden_size // config.heads,
            max_position_embeddings=config.max_position_embeddings,
            use_cache=True,
        )
    )
    return cast(Backbone, cast(object, backbone))


class _ToyMimoBody(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, *, inputs_embeds: Tensor, **_: object) -> object:
        hidden = self.norm(torch.tanh(self.proj(inputs_embeds)))
        return SimpleNamespace(
            last_hidden_state=(hidden, hidden),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


def create_toy_mimo_model(
    *,
    text_vocab_size: int = 128,
    audio_vocab_size: int = 259,
    hidden_size: int = 32,
) -> MimoModel:
    for name, value in (
        ("text_vocab_size", text_vocab_size),
        ("audio_vocab_size", audio_vocab_size),
        ("hidden_size", hidden_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    return MimoModel(
        _ToyMimoBody(hidden_size),
        text_embedding=nn.Embedding(text_vocab_size, hidden_size),
        audio_embedding=nn.Embedding(audio_vocab_size, hidden_size),
        text_readout=BackboneReadout("last_hidden_state[0]"),
        audio_readout=BackboneReadout("last_hidden_state[1]"),
        config=MimoModelConfig(supports_cache_position=False),
    )


__all__ = ["ToyConfig", "create_toy_backbone", "create_toy_mimo_model"]
