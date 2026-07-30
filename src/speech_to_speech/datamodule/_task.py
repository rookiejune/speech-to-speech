from __future__ import annotations

import math
import multiprocessing
from collections.abc import Mapping

from ..task import Task


class TaskWeights:
    def __init__(self, values: Mapping[Task, float]) -> None:
        weights = dict(values)
        _validate_tasks(list(weights))
        _validate_weights(list(weights.values()))
        self._tasks = tuple(task for task, weight in weights.items() if weight > 0)
        self._weights = tuple(
            float(weight) for weight in weights.values() if weight > 0
        )
        self._credits = multiprocessing.Array(
            "d",
            [0.0] * len(self._tasks),
            lock=True,
        )

    @property
    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def allocate(self, batch_size: int) -> list[Task]:
        _validate_batch_size(batch_size)
        with self._credits.get_lock():
            credits = [self._credits[index] for index in range(len(self._tasks))]
            allocated = _allocate_tasks(
                list(self._tasks),
                list(self._weights),
                batch_size,
                credits,
            )
            self._credits[:] = credits
        return allocated


def allocate_tasks(
    tasks: list[Task],
    weights: list[float],
    batch_size: int,
) -> list[Task]:
    _validate_batch_size(batch_size)
    if len(tasks) != len(weights):
        raise ValueError("tasks and weights must have the same length.")
    _validate_weights(weights)
    return _allocate_tasks(tasks, weights, batch_size, [0.0] * len(tasks))


def _allocate_tasks(
    tasks: list[Task],
    weights: list[float],
    batch_size: int,
    credits: list[float],
) -> list[Task]:
    total = sum(weights)
    allocated = []
    for _ in range(batch_size):
        for index, weight in enumerate(weights):
            credits[index] += weight
        selected = max(
            range(len(tasks)),
            key=lambda index: (credits[index], -index),
        )
        credits[selected] -= total
        allocated.append(tasks[selected])
    return allocated


def _validate_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("task allocation batch size must be an integer.")
    if batch_size < 1:
        raise ValueError("task allocation requires a non-empty batch.")


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
