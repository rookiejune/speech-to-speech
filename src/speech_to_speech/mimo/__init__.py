"""Shared contracts and task grammar for aligned text/audio MIMO flows."""

from .contract import MIMO_IGNORE_INDEX, MimoBatch, MimoGenerationStep, MimoSample
from .task import (
    KIMI_PRETRAIN_TASK_WEIGHTS,
    MimoSegment,
    MimoSpecialTokens,
    MimoTask,
    build_mimo_sample,
)

__all__ = [
    "KIMI_PRETRAIN_TASK_WEIGHTS",
    "MIMO_IGNORE_INDEX",
    "MimoBatch",
    "MimoGenerationStep",
    "MimoSample",
    "MimoSegment",
    "MimoSpecialTokens",
    "MimoTask",
    "build_mimo_sample",
]
