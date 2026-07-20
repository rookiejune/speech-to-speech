from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, TypedDict, Union, cast

from ..._compat import StrEnum, auto


class AcousticType(StrEnum):
    FLOW = auto()
    RVQ = auto()


@dataclass(frozen=True)
class DecoderConfig:
    hidden_dim: Optional[int] = None
    layers: int = 8
    heads: int = 8
    ffn_ratio: int = 4


class FlowRepaConfig(TypedDict):
    feature_dim: int
    student_layer: int | None


def decoder_options(
    config: Optional[Union[DecoderConfig, Mapping[str, object]]],
) -> DecoderConfig:
    if config is None:
        return DecoderConfig()
    if isinstance(config, DecoderConfig):
        return config
    return DecoderConfig(
        hidden_dim=cast(Optional[int], config["hidden_dim"]),
        layers=cast(int, config["layers"]),
        heads=cast(int, config["heads"]),
        ffn_ratio=cast(int, config["ffn_ratio"]),
    )
