from __future__ import annotations

import math
import multiprocessing
from collections.abc import Mapping
from typing import Any

from ..task import Task

_TASKS = tuple(Task)
_ABSENT = -1.0


class TaskWeights:
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


def allocate_tasks(
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


__all__ = ["TaskWeights", "allocate_tasks"]
