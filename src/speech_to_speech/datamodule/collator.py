from __future__ import annotations

import random
from typing import Mapping

from anydataset.types import Sample as RawSample

from .task import TaskBase, TaskFactory
from .types import Batch, Sample, SpeechPair, Task


class Collator:
    def __init__(self, weight_dict: Mapping[Task, float]) -> None:
        self._weight_dict: Mapping[type[TaskBase], float] = {
            TaskFactory.get(task): weight for task, weight in weight_dict.items()
        }
        self._tasks: list[type[TaskBase]] | None = None
        self._weights: list[float] | None = None

    @property
    def tasks(self):
        if self._tasks is None:
            self._tasks = list(self._weight_dict.keys())
            _validate_tasks(self._tasks)
        return self._tasks

    @property
    def weights(self):
        if self._weights is None:
            self._weights = [self._weight_dict[task] for task in self.tasks]
            # 不再做weights的参数检验，因为random.choices会做
        return self._weights

    def causal_samples(self, samples: list[SpeechPair]) -> list[Sample]:
        tasks = random.choices(self.tasks, weights=self.weights, k=len(samples))
        return [task.sample(sample) for sample, task in zip(samples, tasks)]

    def __call__(self, samples: list[RawSample]) -> Batch:
        return Batch.from_samples(
            self.causal_samples([SpeechPair.from_raw(sample) for sample in samples])
        )


def _validate_tasks(tasks: list[type[TaskBase]]):
    # 确保模型更新一样的参数
    source = tasks[0].source
    target = tasks[0].target
    for task in tasks:
        if task.source != source or task.target != target:
            raise ValueError()
