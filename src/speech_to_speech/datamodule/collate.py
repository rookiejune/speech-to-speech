from __future__ import annotations

import math
import multiprocessing
from collections.abc import Mapping
from typing import cast

import torch
from anydataset.types import Sample as RawSample

from .loader.contract import ARFraming, validate_ar_framing
from ..task import PredictionModality, Task, execution_signature, resolve_prediction
from .config import TaskConfig
from .parse import parse_task_sample, parse_text_sample
from .contract import DataRuntime, TextRuntime
from .builder import build_task_sample, build_text_sample
from .batch import (
    ModelBatch,
    ModelSample,
)
from .sample import (
    RawSpeechBatch,
    SpeechTaskSample,
)


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
        ar_framing: ARFraming = ARFraming.INSTRUCTION,
        tasks: Mapping[Task, TaskConfig] | None = None,
    ) -> None:
        self.runtime = runtime
        self.encode_missing_codes = encode_missing_codes
        self.interleave_audio_frames = interleave_audio_frames
        self.mask_text_ratio = mask_text_ratio
        self.mask_audio_ratio = mask_audio_ratio
        self.ar_framing = ar_framing
        self.task_configs = tasks
        validate_ar_framing(ar_framing, _positive_tasks(task_weights))
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
                interleave_audio_frames=self.interleave_audio_frames,
                mask_text_ratio=self.mask_text_ratio,
                mask_audio_ratio=self.mask_audio_ratio,
                ar_framing=self.ar_framing,
            )
        return ModelBatch.from_samples(
            [
                build_task_sample(
                    sample,
                    self.runtime,
                    interleave_audio_frames=self.interleave_audio_frames,
                    mask_text_ratio=self.mask_text_ratio,
                    mask_audio_ratio=self.mask_audio_ratio,
                    ar_framing=self.ar_framing,
                    tasks=self.task_configs,
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
        ar_framing: ARFraming = ARFraming.INSTRUCTION,
        tasks: Mapping[Task, TaskConfig] | None = None,
        max_tokens: int | None = None,
        pack_documents: bool = False,
    ) -> None:
        self.runtime = runtime
        self.ar_framing = ar_framing
        self.task_configs = tasks
        self.max_tokens = _validate_max_tokens(max_tokens)
        if not isinstance(pack_documents, bool):
            raise TypeError("text pack_documents must be a boolean.")
        # A budget alone is a useful shorthand; the explicit flag is retained
        # for configs where the batching intent should be visible.
        self.pack_documents = pack_documents or self.max_tokens is not None
        _validate_text_tasks(_positive_tasks(task_weights), prediction=prediction)
        validate_ar_framing(ar_framing, _positive_tasks(task_weights))
        if self.pack_documents:
            if self.max_tokens is None:
                raise ValueError("text pack_documents requires max_tokens.")
            if _positive_tasks(task_weights) != [Task.TEXT_AR]:
                raise ValueError(
                    "text document packing only supports a single TEXT_AR task."
                )
            if ar_framing is not ARFraming.PRETRAINING:
                raise ValueError(
                    "text document packing requires pretraining AR framing."
                )
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
            build_text_sample(
                parse_text_sample(sample, self.runtime),
                task,
                self.runtime,
                ar_framing=self.ar_framing,
                tasks=self.task_configs,
            )
            for sample, task in zip(samples, tasks)
        ]

    def __call__(self, samples: list[RawSample]) -> ModelBatch:
        model_samples = self._model_samples(samples)
        if self.pack_documents:
            assert self.max_tokens is not None
            model_samples = pack_text_samples(model_samples, self.max_tokens)
        return ModelBatch.from_samples(
            model_samples,
            pad_token_id=self.runtime.pad_token_id,
        )


class PackedTextCollator(TextCollator):
    """Reusable TEXT_AR collator with tokenizer-aware document packing."""

    def __init__(
        self,
        runtime: TextRuntime,
        task_weights: Mapping[Task, float],
        *,
        max_tokens: int,
        prediction: PredictionModality | None = None,
        ar_framing: ARFraming = ARFraming.PRETRAINING,
        tasks: Mapping[Task, TaskConfig] | None = None,
    ) -> None:
        super().__init__(
            runtime,
            task_weights,
            prediction=prediction,
            ar_framing=ar_framing,
            tasks=tasks,
            max_tokens=max_tokens,
            pack_documents=True,
        )


def pack_text_samples(samples: list[ModelSample], max_tokens: int) -> list[ModelSample]:
    """Pack pretraining text samples into sequences bounded by ``max_tokens``.

    Every source sample is expected to have a one-token BOS prompt and an EOS
    terminated response.  Long documents are split at token boundaries and an
    EOS is retained for every resulting chunk, so document boundaries remain
    explicit while no sequence exceeds the configured budget.
    """
    budget = _validate_max_tokens(max_tokens)
    if budget is None:  # pragma: no cover - validated above
        raise AssertionError("text packing requires a token budget")
    if not samples:
        return []
    task = samples[0].task
    prediction = samples[0].prediction
    if task is not Task.TEXT_AR or prediction is not PredictionModality.TEXT:
        raise ValueError("text packing requires TEXT_AR text-prediction samples.")
    prompt = samples[0].request["prompt_ids"]
    if prompt.numel() != 1:
        raise ValueError("packed TEXT_AR samples require a one-token BOS prompt.")

    chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, sample in enumerate(samples):
        if sample.task is not task or sample.prediction is not prediction:
            raise ValueError("packed text samples must share task and prediction.")
        sample_prompt = sample.request["prompt_ids"]
        if sample_prompt.numel() != 1 or not torch.equal(sample_prompt, prompt):
            raise ValueError("packed text samples must share the same BOS prompt.")
        response = sample.labels.response_ids
        labels = sample.labels.token_labels[1:]
        if response.numel() != labels.numel():
            raise ValueError(f"text sample {index} response and labels are misaligned.")
        if response.numel() == 0:
            raise ValueError(f"text sample {index} has an empty response.")
        if response.numel() <= budget - 1:
            chunks.append((response, labels))
            continue
        # The final response token is EOS.  Reserve one slot for it in each
        # chunk to preserve a document boundary after splitting.
        content = response[:-1]
        content_labels = labels[:-1]
        eos = response[-1:]
        eos_labels = labels[-1:]
        capacity = budget - 2
        if capacity < 1 and content.numel() > 0:
            raise ValueError(
                f"text sample {index} cannot fit in max_tokens={budget}; "
                "the budget must leave room for content and EOS."
            )
        if content.numel() == 0:
            chunks.append((eos, eos_labels))
            continue
        for start in range(0, content.numel(), capacity):
            end = min(start + capacity, content.numel())
            chunks.append(
                (
                    torch.cat((content[start:end], eos)),
                    torch.cat((content_labels[start:end], eos_labels)),
                )
            )

    packed: list[ModelSample] = []
    current_responses: list[torch.Tensor] = []
    current_labels: list[torch.Tensor] = []
    current_length = prompt.numel()

    def flush() -> None:
        nonlocal current_length
        if not current_responses:
            return
        response = torch.cat(current_responses)
        labels = torch.cat(
            (
                torch.full_like(prompt, -100),
                torch.cat(current_labels),
            )
        )
        packed.append(
            ModelSample.pack(
                prompt_ids=prompt,
                response_ids=response,
                token_labels=labels,
                task=task,
                prediction=prediction,
            )
        )
        current_responses.clear()
        current_labels.clear()
        current_length = prompt.numel()

    for response, labels in chunks:
        if current_responses and current_length + response.numel() > budget:
            flush()
        if current_length + response.numel() > budget:
            raise ValueError("text packing produced a sequence larger than max_tokens.")
        current_responses.append(response)
        current_labels.append(labels)
        current_length += response.numel()
    flush()
    return packed


def _validate_max_tokens(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("text max_tokens must be an integer or None.")
    if value < 2:
        raise ValueError("text max_tokens must be at least 2.")
    return value


def _validate_text_tasks(
    tasks: list[Task],
    *,
    prediction: PredictionModality | None = None,
) -> None:
    from ..task import resolve_prediction

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

__all__ = [
    "Collator",
    "PackedTextCollator",
    "TaskWeights",
    "TextCollator",
    "allocate_tasks",
    "pack_text_samples",
]
