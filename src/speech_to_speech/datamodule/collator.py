from __future__ import annotations

from collections.abc import Mapping

from anydataset.types import Sample as RawSample

from ..prediction import PredictionModality
from ..task import Task
from ._task import TaskWeights
from .parser import parse_task_sample, parse_text_sample
from .protocol import DataRuntime, TextRuntime
from .sample import build_task_sample, build_text_sample
from .types import ModelBatch, ModelSample, RawSpeechBatch, SpeechTaskSample


class Collator:
    def __init__(
        self,
        runtime: DataRuntime,
        task_weights: Mapping[Task, float],
        *,
        encode_missing_codes: bool = False,
        interleave_audio_frames: int = 25,
        mask_text_ratio: float = 0.5,
        mask_audio_ratio: float = 0.5,
        prediction: PredictionModality | None = None,
    ) -> None:
        self.runtime = runtime
        self.encode_missing_codes = encode_missing_codes
        self.interleave_audio_frames = interleave_audio_frames
        self.mask_text_ratio = mask_text_ratio
        self.mask_audio_ratio = mask_audio_ratio
        self._task_weights = TaskWeights(task_weights, prediction=prediction)

    @property
    def tasks(self) -> list[Task]:
        return self._task_weights.tasks

    @property
    def prediction(self) -> PredictionModality | None:
        return self._task_weights.prediction

    def _task_samples(self, samples: list[RawSample]) -> list[SpeechTaskSample]:
        tasks = self._task_weights.allocate(len(samples))
        return [
            parse_task_sample(
                sample,
                task,
                self.runtime,
                encode_missing_codes=self.encode_missing_codes,
                prediction=self.prediction,
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
            [
                build_task_sample(
                    sample,
                    self.runtime,
                    interleave_audio_frames=self.interleave_audio_frames,
                    mask_text_ratio=self.mask_text_ratio,
                    mask_audio_ratio=self.mask_audio_ratio,
                )
                for sample in task_samples
            ],
            pad_token_id=self.runtime.pad_token_id,
        )


class TextCollator:
    def __init__(
        self,
        runtime: TextRuntime,
        task_weights: Mapping[Task, float],
        *,
        prediction: PredictionModality | None = None,
    ) -> None:
        self.runtime = runtime
        _validate_text_tasks(_positive_tasks(task_weights), prediction=prediction)
        self._task_weights = TaskWeights(task_weights, prediction=prediction)

    @property
    def tasks(self) -> list[Task]:
        return self._task_weights.tasks

    @property
    def prediction(self) -> PredictionModality | None:
        return self._task_weights.prediction

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


def _validate_text_tasks(
    tasks: list[Task],
    *,
    prediction: PredictionModality | None = None,
) -> None:
    from ..task_spec import resolve_prediction

    for task in tasks:
        if (
            task.source_modality is not None
            and task.source_modality is not Task.MT.source_modality
        ):
            raise ValueError("text-only task weights must not require audio input.")
        effective = resolve_prediction(task, prediction)
        if effective is not PredictionModality.TEXT:
            raise ValueError("text-only task weights must use text prediction.")


def _positive_tasks(values: Mapping[Task, float]) -> list[Task]:
    return [task for task, weight in values.items() if weight > 0]
