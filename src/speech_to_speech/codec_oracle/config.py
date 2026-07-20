from __future__ import annotations

from dataclasses import dataclass, field

from ..model import DecoderConfig
from .data import DataConfig
from .types import Initialization


@dataclass(frozen=True)
class Config:
    initialization: Initialization = Initialization.CODEC
    normalize_features: bool = True
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    data: DataConfig = field(default_factory=DataConfig)
