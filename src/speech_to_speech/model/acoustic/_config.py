from __future__ import annotations

from typing import TypedDict

from ..._compat import StrEnum, auto


class AcousticType(StrEnum):
    FLOW = auto()
    RVQ = auto()


class DecoderConfig(TypedDict):
    hidden_dim: int | None
    layers: int
    heads: int
    ffn_ratio: int


class FlowRepaConfig(TypedDict):
    feature_dim: int
    student_layer: int | None


def decoder_options(config: DecoderConfig | None) -> DecoderConfig:
    if config is not None:
        return config
    return DecoderConfig(
        hidden_dim=None,
        layers=8,
        heads=8,
        ffn_ratio=4,
    )
