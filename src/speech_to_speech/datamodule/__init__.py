from typing import TYPE_CHECKING

from .diagnostic import SampleSplit
from .mimo import MIMO_IGNORE_INDEX, MimoBatch, MimoSample, collate_mimo
from .mimo_tasks import (
    KIMI_PRETRAIN_TASK_WEIGHTS,
    MimoSegment,
    MimoSpecialTokens,
    MimoTask,
    build_mimo_sample,
)
from .mimo_dataset import (
    JsonlMimoSegmentDataset,
    MimoDatasetConfig,
    MimoTaskDataset,
    ToyMimoSegmentDataset,
)

if TYPE_CHECKING:
    from .mimo_loader import MimoDataModule
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
        from .mimo_loader import MimoDataModule

        return MimoDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
