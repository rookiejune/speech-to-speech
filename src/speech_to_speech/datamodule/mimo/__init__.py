"""Aligned dual-stream MIMO data contracts, tasks, datasets, and loaders."""

from typing import TYPE_CHECKING

from .batch import MIMO_IGNORE_INDEX, MimoBatch, MimoSample, collate_mimo
from .dataset import (
    JsonlMimoSegmentDataset,
    MimoDatasetConfig,
    MimoTaskDataset,
    ToyMimoSegmentDataset,
)
from .task import (
    KIMI_PRETRAIN_TASK_WEIGHTS,
    MimoSegment,
    MimoSpecialTokens,
    MimoTask,
    build_mimo_sample,
)

if TYPE_CHECKING:
    from .loader import MimoDataModule

__all__ = [
    "KIMI_PRETRAIN_TASK_WEIGHTS",
    "MIMO_IGNORE_INDEX",
    "JsonlMimoSegmentDataset",
    "MimoBatch",
    "MimoDataModule",
    "MimoDatasetConfig",
    "MimoSample",
    "MimoSegment",
    "MimoSpecialTokens",
    "MimoTask",
    "MimoTaskDataset",
    "ToyMimoSegmentDataset",
    "build_mimo_sample",
    "collate_mimo",
]


def __getattr__(name: str) -> object:
    if name == "MimoDataModule":
        from .loader import MimoDataModule

        return MimoDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
