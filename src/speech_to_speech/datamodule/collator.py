from __future__ import annotations

import random
from collections.abc import Mapping

from anydataset.types import Sample as RawSample

from .task import TaskBase, TaskFactory
from .types import ModelBatch, Sample, SpeechPair, Task


class Collator:
    def __init__(self, strategy: Mapping[Task, float]) -> None:
        self._strategy: Mapping[type[TaskBase], float] = {}
        self._tasks: list[type[TaskBase]] | None = None
        self._weights: list[float] | None = None
        self.set_strategy(strategy)

    def set_strategy(self, strategy: Mapping[Task, float]) -> None:
        resolved = {
            TaskFactory.get(task): weight for task, weight in strategy.items()
        }
        _validate_strategy_tasks(list(resolved))
        self._strategy = resolved
        self._tasks = None
        self._weights = None

    @property
    def tasks(self):
        if self._tasks is None:
            self._tasks = list(self._strategy.keys())
        return self._tasks

    @property
    def weights(self):
        if self._weights is None:
            self._weights = [self._strategy[task] for task in self.tasks]
            # 不再做weights的参数检验，因为random.choices会做
        return self._weights

    def task_samples(self, samples: list[SpeechPair]) -> list[Sample]:
        tasks = random.choices(self.tasks, weights=self.weights, k=len(samples))
        return [task.sample(sample) for sample, task in zip(samples, tasks)]

    def __call__(self, samples: list[RawSample]) -> ModelBatch:
        return ModelBatch.from_samples(
            self.task_samples([SpeechPair.from_raw(sample) for sample in samples])
        )


def _validate_strategy_tasks(tasks: list[type[TaskBase]]):
    if not tasks:
        raise ValueError("task strategy must contain at least one task.")
    source = tasks[0].name.source_modality
    target = tasks[0].name.target_modality
    for task in tasks:
        if (
            task.name.source_modality is not source
            or task.name.target_modality is not target
        ):
            raise ValueError(
                "all tasks in one strategy must use the same source and target modalities."
            )
