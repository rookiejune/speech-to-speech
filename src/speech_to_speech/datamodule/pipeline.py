"""Task-sample streaming and main-process batch construction."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor, device as TorchDevice
from torch.utils.data import IterableDataset

from ..config import DatasetFactoryConfig, TaskConfig
from ..dataset import training_dataset
from .batch_builder import CausalLMBatchBuilder
from ..types.datamodule import (
    AutoregressionExample,
    CausalLMBatch,
    LongCatBPETokenizer,
    LongCatPair,
    LongCatSide,
    Task,
    TaskFamily,
    TranslationExample,
)
from .example import (
    encode_autoregression_example,
    encode_translation_example,
    longcat_pair_from_sample,
)
from .longcat import collate_longcat_sides


@dataclass(frozen=True)
class TaskSample:
    family: TaskFamily
    source: LongCatSide | None
    target: LongCatSide

    @property
    def example(self) -> AutoregressionExample | TranslationExample:
        if self.family in _AUTOREGRESSION_FAMILIES:
            return AutoregressionExample(audio_ids=self.target.semantic_ids)
        if self.family in _TRANSLATION_FAMILIES:
            if self.source is None:
                raise ValueError("translation task sample must carry a source side.")
            return TranslationExample(
                source_ids=self.source.semantic_ids,
                target_ids=self.target.semantic_ids,
            )
        raise ValueError(f"unsupported task family: {self.family.value}")

    @property
    def length(self) -> int:
        target_length = _sequence_length(self.target.semantic_ids)
        if self.source is None:
            return target_length
        return _sequence_length(self.source.semantic_ids) + target_length


_AUTOREGRESSION_FAMILIES = frozenset(
    {
        TaskFamily.SOURCE_AR,
        TaskFamily.TARGET_AR,
    }
)
_TRANSLATION_FAMILIES = frozenset(
    {
        TaskFamily.SOURCE_TO_TARGET,
        TaskFamily.TARGET_TO_SOURCE,
    }
)


class TaskSampleStream(IterableDataset[TaskSample]):
    def __init__(
        self,
        dataset_factory: DatasetFactoryConfig,
        *,
        tasks: TaskConfig,
    ) -> None:
        self.dataset_factory = dataset_factory
        self.families = _enabled_families(tasks)
        if not self.families:
            raise ValueError("tasks.enabled must contain at least one task.")
        self.weights = tasks.weights
        _validate_positive_enabled_weight(self.families, weights=self.weights)

    def __iter__(self) -> Iterator[TaskSample]:
        source = training_dataset(self.dataset_factory)
        accumulators = {family: 0.0 for family in TaskFamily}
        for sample in source:
            pair = longcat_pair_from_sample(sample)
            for task_sample in _task_samples_from_pair(pair, self.families):
                accumulators[task_sample.family] += self.weights.weight(task_sample.family)
                count = int(accumulators[task_sample.family])
                accumulators[task_sample.family] -= count
                for _ in range(count):
                    yield task_sample


class TaskSampleCollator:
    def __call__(self, samples: list[TaskSample]) -> list[TaskSample]:
        return samples


class TaskBatchMapper:
    def __init__(
        self,
        source: Iterable[Sequence[TaskSample]],
        *,
        builder: CausalLMBatchBuilder,
        bpe_tokenizer: LongCatBPETokenizer,
        device: TorchDevice,
    ) -> None:
        self.source = source
        self.batch_builder = TaskBatchBuilder(
            builder=builder,
            bpe_tokenizer=bpe_tokenizer,
            device=device,
        )

    def __iter__(self) -> Iterator[CausalLMBatch]:
        for samples in self.source:
            yield self.batch_builder(samples)


class TaskBatchBuilder:
    def __init__(
        self,
        *,
        builder: CausalLMBatchBuilder,
        bpe_tokenizer: LongCatBPETokenizer,
        device: TorchDevice,
    ) -> None:
        self.builder = builder
        self.bpe_tokenizer = bpe_tokenizer
        self.device = device

    def __call__(self, samples: Sequence[TaskSample]) -> CausalLMBatch:
        if not samples:
            raise ValueError("task sample batch must not be empty.")
        batch = self.builder.mixed(
            [
                _encode_task_sample(
                    sample,
                    self.bpe_tokenizer,
                    device=self.device,
                )
                for sample in samples
            ]
        )
        return CausalLMBatch(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
            logits_to_keep=batch.logits_to_keep,
            loss_weights=batch.loss_weights,
            source_audio=collate_longcat_sides(
                [sample.source for sample in samples],
                device=self.device,
            ),
            target_audio=collate_longcat_sides(
                [sample.target for sample in samples],
                device=self.device,
            ),
            task_family=torch.tensor(
                [sample.family.id for sample in samples],
                dtype=torch.long,
                device=self.device,
            ),
        )


def task_sample_length(sample: TaskSample) -> int:
    return sample.length


def _encode_task_sample(
    sample: TaskSample,
    tokenizer: LongCatBPETokenizer,
    *,
    device: TorchDevice,
) -> AutoregressionExample | TranslationExample:
    example = sample.example
    if isinstance(example, AutoregressionExample):
        return encode_autoregression_example(example, tokenizer, device=device)
    if isinstance(example, TranslationExample):
        return encode_translation_example(example, tokenizer, device=device)
    raise TypeError("task sample must contain a task example.")


def _sequence_length(ids: Tensor) -> int:
    if ids.dim() == 0:
        raise ValueError("task sample ids must have a sequence dimension.")
    return int(ids.numel())


def _enabled_tasks(tasks: TaskConfig) -> frozenset[Task]:
    enabled: set[Task] = set()
    for name in tasks.enabled:
        enabled.add(Task(name))
    return frozenset(enabled)


def _enabled_families(tasks: TaskConfig) -> tuple[TaskFamily, ...]:
    enabled = _enabled_tasks(tasks)
    families: list[TaskFamily] = []
    if Task.AUTOREGRESSION in enabled:
        families.extend(
            (
                TaskFamily.SOURCE_AR,
                TaskFamily.TARGET_AR,
            )
        )
    if Task.TRANSLATION in enabled:
        families.extend(
            (
                TaskFamily.SOURCE_TO_TARGET,
                TaskFamily.TARGET_TO_SOURCE,
            )
        )
    return tuple(families)


def _task_samples_from_pair(
    pair: LongCatPair,
    families: Sequence[TaskFamily],
) -> tuple[TaskSample, ...]:
    samples: list[TaskSample] = []
    for family in families:
        match family:
            case TaskFamily.SOURCE_AR:
                samples.append(TaskSample(family=family, source=None, target=pair.source))
            case TaskFamily.TARGET_AR:
                samples.append(TaskSample(family=family, source=None, target=pair.target))
            case TaskFamily.SOURCE_TO_TARGET:
                samples.append(TaskSample(family=family, source=pair.source, target=pair.target))
            case TaskFamily.TARGET_TO_SOURCE:
                samples.append(TaskSample(family=family, source=pair.target, target=pair.source))
    return tuple(samples)


def _validate_positive_enabled_weight(
    enabled_families: Sequence[TaskFamily],
    *,
    weights: object,
) -> None:
    total = 0.0
    for family in TaskFamily:
        weight = weights.weight(family)
        if not isinstance(weight, int | float) or isinstance(weight, bool):
            raise TypeError(f"task weight for {family.value} must be a number.")
        if not isfinite(float(weight)) or weight < 0.0:
            raise ValueError(f"task weight for {family.value} must be finite and non-negative.")
        if family in enabled_families:
            total += float(weight)
    if total <= 0.0:
        raise ValueError("enabled tasks must have at least one positive task weight.")
