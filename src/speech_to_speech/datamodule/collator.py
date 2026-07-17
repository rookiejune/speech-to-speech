from __future__ import annotations

import random
from collections.abc import Mapping

from anydataset.types import Sample as RawSample

from .parser import parse_sample
from .protocol import DataRuntime
from .sample import build_sample
from ..task import Task
from .types import ModelBatch, ModelSample


class Collator:
    def __init__(
        self,
        runtime: DataRuntime,
        task_weights: Mapping[Task, float],
    ) -> None:
        self.runtime = runtime
        self._task_weights: Mapping[Task, float] = {}
        self.set_task_weights(task_weights)

    def set_task_weights(self, task_weights: Mapping[Task, float]) -> None:
        _validate_tasks(list(task_weights))
        self._task_weights = dict(task_weights)

    @property
    def tasks(self) -> list[Task]:
        return list(self._task_weights)

    def _model_samples(self, samples: list[RawSample]) -> list[ModelSample]:
        available = self.tasks
        weights = [self._task_weights[task] for task in available]
        tasks = random.choices(available, weights=weights, k=len(samples))
        return [
            build_sample(parse_sample(sample, self.runtime), task, self.runtime)
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
        if (
            task.source_modality is not source
            or task.target_modality is not target
        ):
            raise ValueError(
                "all weighted tasks must use the same source and target modalities."
            )
