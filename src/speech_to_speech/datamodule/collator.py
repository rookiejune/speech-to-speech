from __future__ import annotations

import math
import multiprocessing
from collections.abc import Mapping
from typing import Any

from anydataset.types import Sample as RawSample

from .parser import parse_sample, parse_text_sample
from .protocol import DataRuntime, TextRuntime
from .sample import build_sample, build_text_sample
from ..task import Task
from .types import ModelBatch, ModelSample

_TASKS = tuple(Task)
_ABSENT = -1.0


class _TaskWeights:
    def __init__(self, values: Mapping[Task, float]) -> None:
        self._values: Any = multiprocessing.Array(
            "d",
            [_ABSENT] * len(_TASKS),
            lock=True,
        )
        self.set(values)

    def set(self, values: Mapping[Task, float]) -> None:
        weights = dict(values)
        _validate_tasks(list(weights))
        _validate_weights(list(weights.values()))
        updated = [float(weights.get(task, _ABSENT)) for task in _TASKS]
        with self._values.get_lock():
            self._values[:] = updated

    def get(self) -> tuple[list[Task], list[float]]:
        with self._values.get_lock():
            values = list(self._values[:])
        tasks = [task for task, weight in zip(_TASKS, values) if weight > 0]
        weights = [weight for weight in values if weight > 0]
        return tasks, weights


class Collator:
    def __init__(
        self,
        runtime: DataRuntime,
        task_weights: Mapping[Task, float],
    ) -> None:
        self.runtime = runtime
        self._task_weights = _TaskWeights(task_weights)

    def set_task_weights(self, task_weights: Mapping[Task, float]) -> None:
        self._task_weights.set(task_weights)

    @property
    def tasks(self) -> list[Task]:
        tasks, _ = self._task_weights.get()
        return tasks

    def _model_samples(self, samples: list[RawSample]) -> list[ModelSample]:
        available, weights = self._task_weights.get()
        tasks = _allocate_tasks(available, weights, len(samples))
        return [
            build_sample(parse_sample(sample, self.runtime), task, self.runtime)
            for sample, task in zip(samples, tasks)
        ]

    def __call__(self, samples: list[RawSample]) -> ModelBatch:
        return ModelBatch.from_samples(
            self._model_samples(samples),
            pad_token_id=self.runtime.pad_token_id,
        )


class TextCollator:
    def __init__(
        self,
        runtime: TextRuntime,
        task_weights: Mapping[Task, float],
    ) -> None:
        self.runtime = runtime
        self._task_weights = _TaskWeights(task_weights)
        _validate_text_tasks(self.tasks)

    def set_task_weights(self, task_weights: Mapping[Task, float]) -> None:
        self._task_weights.set(task_weights)
        _validate_text_tasks(self.tasks)

    @property
    def tasks(self) -> list[Task]:
        tasks, _ = self._task_weights.get()
        return tasks

    def _model_samples(self, samples: list[RawSample]) -> list[ModelSample]:
        available, weights = self._task_weights.get()
        tasks = _allocate_tasks(available, weights, len(samples))
        return [
            build_text_sample(parse_text_sample(sample, self.runtime), task, self.runtime)
            for sample, task in zip(samples, tasks)
        ]

    def __call__(self, samples: list[RawSample]) -> ModelBatch:
        return ModelBatch.from_samples(
            self._model_samples(samples),
            pad_token_id=self.runtime.pad_token_id,
        )


def _validate_tasks(tasks: list[Task]) -> None:
    if not tasks:
        raise ValueError("task weights must contain at least one task.")
    source = tasks[0].source_modality
    target = tasks[0].target_modality
    for task in tasks:
        if task.source_modality is not source or task.target_modality is not target:
            raise ValueError(
                "all weighted tasks must use the same source and target modalities."
            )


def _validate_weights(weights: list[float]) -> None:
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("task weights must be finite and non-negative.")
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("task weights must have a finite positive total.")


def _validate_text_tasks(tasks: list[Task]) -> None:
    for task in tasks:
        if (
            task.source_modality is not None
            and task.source_modality is not Task.MT.source_modality
        ):
            raise ValueError("text-only task weights must not require audio input.")
        if task.target_modality is not Task.MT.target_modality:
            raise ValueError("text-only task weights must target text.")


def _allocate_tasks(
    tasks: list[Task],
    weights: list[float],
    batch_size: int,
) -> list[Task]:
    if batch_size < 1:
        raise ValueError("task allocation requires a non-empty batch.")
    total = sum(weights)
    targets = [weight * batch_size / total for weight in weights]
    if any(target < 1 for target in targets):
        raise ValueError(
            "batch size is too small for fixed task weights; each non-zero task "
            "must receive at least one sample."
        )
    counts = [math.floor(target) for target in targets]
    remaining = batch_size - sum(counts)
    order = sorted(
        range(len(tasks)),
        key=lambda index: (targets[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return [task for task, count in zip(tasks, counts) for _ in range(count)]
