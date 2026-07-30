from __future__ import annotations

from collections.abc import Mapping

from anydataset.types import Sample as RawSample

from ._task import TaskWeights
from .parser import parse_task_sample, parse_text_sample
from .protocol import DataRuntime, TextRuntime
from .sample import build_task_sample, build_text_sample
from ..task import Task
from .types import ModelBatch, ModelSample, RawSpeechBatch, SpeechTaskSample

class Collator:
    def __init__(
        self,
        runtime: DataRuntime,
        task_weights: Mapping[Task, float],
        *,
        encode_missing_codes: bool = False,
    ) -> None:
        self.runtime = runtime
        self.encode_missing_codes = encode_missing_codes
        self._task_weights = TaskWeights(task_weights)

    @property
    def tasks(self) -> list[Task]:
        return self._task_weights.tasks

    def _task_samples(self, samples: list[RawSample]) -> list[SpeechTaskSample]:
        tasks = self._task_weights.allocate(len(samples))
        return [
            parse_task_sample(
                sample,
                task,
                self.runtime,
                encode_missing_codes=self.encode_missing_codes,
            )
            for sample, task in zip(samples, tasks)
        ]

    def __call__(self, samples: list[RawSample]) -> ModelBatch | RawSpeechBatch:
        task_samples = self._task_samples(samples)
        if any(sample.needs_codec for sample in task_samples):
            return RawSpeechBatch(
                samples=tuple(task_samples),
                pad_token_id=self.runtime.pad_token_id,
            )
        return ModelBatch.from_samples(
            [build_task_sample(sample, self.runtime) for sample in task_samples],
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

    @property
    def tasks(self) -> list[Task]:
        return self._task_weights.tasks

    def _model_samples(self, samples: list[RawSample]) -> list[ModelSample]:
        tasks = self._task_weights.allocate(len(samples))
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
