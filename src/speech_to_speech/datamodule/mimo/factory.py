"""Factories for prepared MIMO datasets."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from torch.utils.data import Dataset

from ...mimo import MimoSample, MimoSegment, MimoSpecialTokens, MimoTask
from .dataset import MimoDatasetConfig, MimoTaskDataset, SegmentSource


def create_dataset(
    factory_path: str,
    kwargs: Mapping[str, Any],
    *,
    kind: str,
    special: MimoSpecialTokens,
    config: MimoDatasetConfig,
) -> Dataset[MimoSample]:
    """Instantiate and normalize a prepared sample or segment dataset."""

    value = import_factory(factory_path)(**dict(kwargs))
    if kind == "samples":
        return _sample_dataset(value)
    if kind != "segments":
        raise ValueError("MIMO dataset kind must be 'segments' or 'samples'.")
    return MimoTaskDataset(_segment_source(value), special, config=config)


def import_factory(path: str) -> Callable[..., Any]:
    if not isinstance(path, str) or not path:
        raise ValueError("dataset factory must be a non-empty import path.")
    if ":" in path:
        module_name, attribute = path.split(":", 1)
    else:
        module_name, _, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise ValueError("dataset factory must use module:attribute or module.attribute.")
    value = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(value):
        raise TypeError(f"dataset factory {path!r} is not callable.")
    return cast(Callable[..., Any], value)


def task_weights(values: Mapping[str, float]) -> dict[MimoTask, float] | None:
    if not values:
        return None
    return {MimoTask(key): float(weight) for key, weight in values.items()}


def _segment_source(value: object) -> SegmentSource:
    if isinstance(value, Dataset):
        if not hasattr(value, "__len__"):
            raise TypeError("segment factory datasets must define __len__.")
        return cast(SegmentSource, value)
    if isinstance(value, Sequence):
        values = tuple(value)
        if not values or any(not isinstance(item, MimoSegment) for item in values):
            raise TypeError("segment factory sequences must contain MimoSegment values.")
        return cast(SegmentSource, _SequenceDataset(values))
    raise TypeError("segment factory must return a Dataset or sequence of MimoSegment.")


def _sample_dataset(value: object) -> Dataset[MimoSample]:
    if isinstance(value, Dataset):
        return cast(Dataset[MimoSample], value)
    if isinstance(value, Sequence):
        values = tuple(value)
        if not values or any(not isinstance(item, MimoSample) for item in values):
            raise TypeError("sample factory sequences must contain MimoSample values.")
        return _SequenceDataset(values)
    raise TypeError("sample factory must return a Dataset or sequence of MimoSample.")


class _SequenceDataset(Dataset[Any]):
    def __init__(self, values: Sequence[Any]) -> None:
        self.values = tuple(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Any:
        return self.values[index]


__all__ = ["create_dataset", "import_factory", "task_weights"]
