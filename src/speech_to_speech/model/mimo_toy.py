"""Small local MIMO model used by CPU smoke runs."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import Tensor, nn

from ..runtime.backbone import BackboneReadout
from .mimo import MimoModel, MimoModelConfig


class _ToyBody(nn.Module):
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
        _ToyBody(hidden_size),
        text_embedding=nn.Embedding(text_vocab_size, hidden_size),
        audio_embedding=nn.Embedding(audio_vocab_size, hidden_size),
        text_readout=BackboneReadout("last_hidden_state[0]"),
        audio_readout=BackboneReadout("last_hidden_state[1]"),
        config=MimoModelConfig(supports_cache_position=False),
    )


__all__ = ["create_toy_mimo_model"]
