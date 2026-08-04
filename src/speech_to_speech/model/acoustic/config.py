from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional, TypedDict, Union, cast

from ..._compat import StrEnum, auto


class AcousticType(StrEnum):
    NONE = auto()
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
    student_layer: Optional[int]


@dataclass
class RepaConfig:
    weight: Optional[float] = None
    teacher_checkpoint: str = "microsoft/wavlm-base"
    teacher_layer: int = 9
    student_layer: Optional[int] = None


@dataclass
class AcousticNoneConfig:
    type: str = AcousticType.NONE.value
    name: str = "token"


@dataclass
class FlowConfig:
    type: str = AcousticType.FLOW.value
    name: str = "flow"
    init_artifact: Optional[str] = None
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    repa: RepaConfig = field(default_factory=RepaConfig)

    def __post_init__(self) -> None:
        _validate_init_artifact(self.init_artifact)


@dataclass
class RVQConfig:
    type: str = AcousticType.RVQ.value
    name: str = "rvq"
    init_artifact: Optional[str] = None
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    def __post_init__(self) -> None:
        _validate_init_artifact(self.init_artifact)


AcousticConfig = Union[AcousticNoneConfig, FlowConfig, RVQConfig]


def _validate_init_artifact(value: Optional[str]) -> None:
    if value is not None and not value:
        raise ValueError("model acoustic init_artifact must not be empty.")


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
