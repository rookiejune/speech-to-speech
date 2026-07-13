from __future__ import annotations

import random
from collections.abc import Mapping

from anydataset.types import Sample as RawSample

from .task import build_sample
from .types import ModelBatch, Sample, SpeechPair, Task


class Collator:
    def __init__(self, strategy: Mapping[Task, float]) -> None:
        self._strategy: Mapping[Task, float] = {}
        self.set_strategy(strategy)

    def set_strategy(self, strategy: Mapping[Task, float]) -> None:
        _validate_strategy_tasks(list(strategy))
        self._strategy = dict(strategy)

    @property
    def tasks(self) -> list[Task]:
        return list(self._strategy)

    def _samples(self, samples: list[SpeechPair]) -> list[Sample]:
        available = self.tasks
        weights = [self._strategy[task] for task in available]
        tasks = random.choices(available, weights=weights, k=len(samples))
        return [build_sample(sample, task) for sample, task in zip(samples, tasks)]

    def __call__(self, samples: list[RawSample]) -> ModelBatch:
        return ModelBatch.from_samples(
            self._samples([SpeechPair.from_raw(sample) for sample in samples])
        )


def _validate_strategy_tasks(tasks: list[Task]) -> None:
    if not tasks:
        raise ValueError("task strategy must contain at least one task.")
    source = tasks[0].source_modality
    target = tasks[0].target_modality
    for task in tasks:
        if (
            task.source_modality is not source
            or task.target_modality is not target
        ):
            raise ValueError(
                "all tasks in one strategy must use the same source and target modalities."
            )
