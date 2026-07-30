from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from itertools import islice
from typing import cast

from anydataset import IterableAnyDataset
from anydataset.types import Sample as RawSample
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset

from ...prediction import PredictionModality
from ...task import Task
from ..collate.collator import TextCollator
from ..protocol import TextRuntime, TextRuntimeSnapshot
from ..dataset.text import TextConfig, load_text_dataset
from ..types import ModelBatch


class TextLoader:
    def __init__(
        self,
        config: TextConfig,
        runtime: TextRuntime,
        task_weights: Mapping[Task, float],
        *,
        prediction: PredictionModality | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.collator = TextCollator(runtime, task_weights, prediction=prediction)
        self.max_samples = max_samples
        self._train_dataset: Dataset[RawSample] | IterableAnyDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self._train_dataset is not None:
            return
        self._train_dataset = load_text_dataset(self.config.dataset)

    def train_samples(self, indices: Sequence[int]) -> list[RawSample]:
        if self._train_dataset is None:
            raise RuntimeError("TextLoader.setup() must run before reading samples.")
        return _samples(self._train_dataset, indices)

    def diagnostic_collator(self, task: Task) -> TextCollator:
        return TextCollator(
            self.runtime,
            {task: 1.0},
            prediction=self.collator.prediction,
        )

    def train_dataloader(self) -> Iterable[ModelBatch]:
        return self._dataloader()

    def validation_dataloader(self) -> Iterable[ModelBatch]:
        return self._dataloader()

    def _dataloader(self) -> Iterable[ModelBatch]:
        if self._train_dataset is None:
            raise RuntimeError(
                "text loader setup() must run before building a dataloader."
            )
        loader = self.config.dataloader
        num_workers = loader.num_workers
        if not isinstance(self.collator.runtime, TextRuntimeSnapshot):
            self.collator.runtime = cast(
                TextRuntime,
                cast(object, TextRuntimeSnapshot.from_runtime(self.runtime)),
            )
        dataset = _limit_dataset(self._train_dataset, self.max_samples)
        return DataLoader(
            dataset,
            batch_size=loader.batch_size,
            num_workers=num_workers,
            pin_memory=loader.pin_memory,
            persistent_workers=(loader.persistent_workers and num_workers > 0),
            collate_fn=self.collator,
        )


def _samples(
    dataset: Dataset[RawSample] | IterableAnyDataset,
    indices: Sequence[int],
) -> list[RawSample]:
    if not indices:
        return []
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise TypeError("text sample indices must contain integers.")
    if any(index < 0 for index in indices):
        raise ValueError("text sample indices must be non-negative.")
    if not isinstance(dataset, IterableAnyDataset):
        return [dataset[index] for index in indices]

    selected: dict[int, RawSample] = {}
    requested = set(indices)
    iterator: Iterator[RawSample] = dataset.iter_shard(1, 0)
    for index, sample in enumerate(islice(iterator, max(requested) + 1)):
        if index in requested:
            selected[index] = sample
    missing = requested - set(selected)
    if missing:
        raise IndexError(f"text sample index {min(missing)} is outside the dataset.")
    return [selected[index] for index in indices]


def _limit_dataset(
    dataset: Dataset[RawSample] | IterableAnyDataset,
    max_samples: int | None,
) -> Dataset[RawSample] | IterableAnyDataset | IterableDataset[RawSample]:
    if max_samples is None:
        return dataset
    if isinstance(max_samples, bool) or not isinstance(max_samples, int):
        raise TypeError("text max_samples must be an integer or None.")
    if max_samples <= 0:
        raise ValueError("text max_samples must be positive.")
    if not isinstance(dataset, IterableAnyDataset):
        length = len(cast(Sized, dataset))
        return Subset(dataset, range(min(max_samples, length)))
    return _LimitedAnyDataset(dataset, max_samples)


class _LimitedAnyDataset(IterableDataset[RawSample]):
    def __init__(self, dataset: IterableAnyDataset, max_samples: int) -> None:
        self.dataset = dataset
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[RawSample]:
        for index, sample in self.dataset.iter_indexed_runtime_shard():
            if index >= self.max_samples:
                break
            yield sample


__all__ = ["TextLoader"]
