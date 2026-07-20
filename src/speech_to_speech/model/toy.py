from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from transformers import Qwen3Config, Qwen3ForCausalLM

from ..runtime.types import Backbone


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
    backbone = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=text_vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.layers,
            num_attention_heads=config.heads,
            num_key_value_heads=config.heads,
            head_dim=config.hidden_size // config.heads,
            max_position_embeddings=config.max_position_embeddings,
            tie_word_embeddings=True,
            use_cache=True,
        )
    )
    return cast(Backbone, cast(object, backbone))


__all__ = ["ToyConfig", "create_toy_backbone"]
