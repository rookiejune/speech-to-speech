from typing import TYPE_CHECKING

from .diagnostic import SampleSplit
from .mimo import (
    KIMI_PRETRAIN_TASK_WEIGHTS,
    MIMO_IGNORE_INDEX,
    JsonlMimoSegmentDataset,
    MimoBatch,
    MimoDatasetConfig,
    MimoSample,
    MimoSegment,
    MimoSpecialTokens,
    MimoTask,
    MimoTaskDataset,
    ToyMimoSegmentDataset,
    build_mimo_sample,
    collate_mimo,
)

if TYPE_CHECKING:
    from .mimo import MimoDataModule
    from .module import DataModule

__all__ = [
    "DataModule",
    "MIMO_IGNORE_INDEX",
    "KIMI_PRETRAIN_TASK_WEIGHTS",
    "JsonlMimoSegmentDataset",
    "MimoBatch",
    "MimoDataModule",
    "MimoSegment",
    "MimoSample",
    "MimoSpecialTokens",
    "MimoTask",
    "SampleSplit",
    "collate_mimo",
    "build_mimo_sample",
    "MimoDatasetConfig",
    "MimoTaskDataset",
    "ToyMimoSegmentDataset",
]


def __getattr__(name: str) -> object:
    if name == "DataModule":
        from .module import DataModule

        return DataModule
    if name == "MimoDataModule":
        from .mimo import MimoDataModule

        return MimoDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
