from __future__ import annotations

import math
import multiprocessing
from collections.abc import Mapping
from typing import cast

from ...task import PredictionModality, Task, execution_signature, resolve_prediction


class TaskWeights:
    def __init__(
        self,
        values: Mapping[Task, float],
        *,
        prediction: PredictionModality | None = None,
    ) -> None:
        weights = dict(values)
        positive = [task for task, weight in weights.items() if weight > 0]
        if prediction is not None:
            for task in positive:
                resolve_prediction(task, prediction)
        _validate_tasks(positive, prediction=prediction)
        _validate_weights(list(weights.values()))
        self._prediction = prediction
        self._tasks = tuple(positive)
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

    @property
    def prediction(self) -> PredictionModality | None:
        return self._prediction

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

    def __getstate__(self) -> dict[str, object]:
        with self._credits.get_lock():
            credits = [self._credits[index] for index in range(len(self._tasks))]
        return {
            "prediction": self._prediction,
            "tasks": self._tasks,
            "weights": self._weights,
            "credits": credits,
        }

    def __setstate__(self, state: Mapping[str, object]) -> None:
        prediction = state["prediction"]
        if prediction is not None and not isinstance(prediction, PredictionModality):
            raise TypeError("pickled task weights prediction is invalid.")
        tasks = state["tasks"]
        weights = state["weights"]
        credits = state["credits"]
        if not isinstance(tasks, tuple) or any(
            not isinstance(task, Task) for task in tasks
        ):
            raise TypeError("pickled task weights tasks are invalid.")
        if not isinstance(weights, tuple) or any(
            not isinstance(weight, float) for weight in weights
        ):
            raise TypeError("pickled task weights values are invalid.")
        if not isinstance(credits, list) or any(
            not isinstance(credit, float) for credit in credits
        ):
            raise TypeError("pickled task weights credits are invalid.")
        if len(tasks) != len(weights) or len(tasks) != len(credits):
            raise ValueError("pickled task weights state lengths must match.")
        self._prediction = prediction
        self._tasks = cast(tuple[Task, ...], tasks)
        self._weights = cast(tuple[float, ...], weights)
        _validate_tasks(list(self._tasks), prediction=self._prediction)
        _validate_weights(list(self._weights))
        self._credits = multiprocessing.Array(
            "d",
            cast(list[float], credits),
            lock=True,
        )


def allocate_tasks(
    tasks: list[Task],
    weights: list[float],
    batch_size: int,
    *,
    prediction: PredictionModality | None = None,
) -> list[Task]:
    _validate_batch_size(batch_size)
    if len(tasks) != len(weights):
        raise ValueError("tasks and weights must have the same length.")
    _validate_weights(weights)
    _validate_tasks(tasks, prediction=prediction)
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


def _validate_tasks(
    tasks: list[Task],
    *,
    prediction: PredictionModality | None = None,
) -> None:
    if not tasks:
        raise ValueError("task weights must contain at least one task.")
    signature = execution_signature(tasks[0], prediction=prediction)
    for task in tasks:
        if execution_signature(task, prediction=prediction) != signature:
            raise ValueError(
                "all weighted tasks must use the same execution signature "
                "(source layout and prediction modality)."
            )


def _validate_weights(weights: list[float]) -> None:
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("task weights must be finite and non-negative.")
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("task weights must have a finite positive total.")


__all__ = ["TaskWeights", "allocate_tasks"]
