from __future__ import annotations

from collections.abc import Mapping

from anydataset.types import Sample as RawSample

from ._task import TaskWeights, allocate_tasks
from .parser import parse_sample, parse_text_sample
from .protocol import DataRuntime, TextRuntime
from .sample import build_sample, build_text_sample
from ..task import Task
from .types import ModelBatch, ModelSample

class Collator:
    def __init__(
        self,
        runtime: DataRuntime,
        task_weights: Mapping[Task, float],
    ) -> None:
        self.runtime = runtime
        self._task_weights = TaskWeights(task_weights)

    def set_task_weights(self, task_weights: Mapping[Task, float]) -> None:
        self._task_weights.set(task_weights)

    @property
    def tasks(self) -> list[Task]:
        tasks, _ = self._task_weights.get()
        return tasks

    def _model_samples(self, samples: list[RawSample]) -> list[ModelSample]:
        available, weights = self._task_weights.get()
        tasks = allocate_tasks(available, weights, len(samples))
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
        _validate_text_tasks(_positive_tasks(task_weights))
        self._task_weights = TaskWeights(task_weights)

    def set_task_weights(self, task_weights: Mapping[Task, float]) -> None:
        _validate_text_tasks(_positive_tasks(task_weights))
        self._task_weights.set(task_weights)

    @property
    def tasks(self) -> list[Task]:
        tasks, _ = self._task_weights.get()
        return tasks

    def _model_samples(self, samples: list[RawSample]) -> list[ModelSample]:
        available, weights = self._task_weights.get()
        tasks = allocate_tasks(available, weights, len(samples))
        return [
            build_text_sample(parse_text_sample(sample, self.runtime), task, self.runtime)
            for sample, task in zip(samples, tasks)
        ]

    def __call__(self, samples: list[RawSample]) -> ModelBatch:
        return ModelBatch.from_samples(
            self._model_samples(samples),
            pad_token_id=self.runtime.pad_token_id,
        )


def _validate_text_tasks(tasks: list[Task]) -> None:
    for task in tasks:
        if (
            task.source_modality is not None
            and task.source_modality is not Task.MT.source_modality
        ):
            raise ValueError("text-only task weights must not require audio input.")
        if task.target_modality is not Task.MT.target_modality:
            raise ValueError("text-only task weights must target text.")


def _positive_tasks(values: Mapping[Task, float]) -> list[Task]:
    return [task for task, weight in values.items() if weight > 0]
