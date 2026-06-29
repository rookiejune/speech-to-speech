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
from ..types import (
    AutoregressionExample,
    CausalLMBatch,
    LongCatBatchSide,
    LongCatSide,
    Task,
    TaskFamily,
    TranslationExample,
)
from .batch_builder import CausalLMBatchBuilder
from .example import (
    encode_autoregression_example,
    encode_translation_example,
    longcat_pair_from_sample,
)


@dataclass(frozen=True)
class SourceAutoregressionSample:
    target: LongCatSide

    @property
    def example(self) -> AutoregressionExample:
        return AutoregressionExample(audio_ids=self.target.semantic_ids)

    @property
    def length(self) -> int:
        return _sequence_length(self.target.semantic_ids)


@dataclass(frozen=True)
class TargetAutoregressionSample:
    target: LongCatSide

    @property
    def example(self) -> AutoregressionExample:
        return AutoregressionExample(audio_ids=self.target.semantic_ids)

    @property
    def length(self) -> int:
        return _sequence_length(self.target.semantic_ids)


@dataclass(frozen=True)
class SourceToTargetSample:
    source: LongCatSide
    target: LongCatSide

    @property
    def example(self) -> TranslationExample:
        return TranslationExample(
            source_ids=self.source.semantic_ids,
            target_ids=self.target.semantic_ids,
        )

    @property
    def length(self) -> int:
        return _sequence_length(self.source.semantic_ids) + _sequence_length(
            self.target.semantic_ids
        )


@dataclass(frozen=True)
class TargetToSourceSample:
    source: LongCatSide
    target: LongCatSide

    @property
    def example(self) -> TranslationExample:
        return TranslationExample(
            source_ids=self.source.semantic_ids,
            target_ids=self.target.semantic_ids,
        )

    @property
    def length(self) -> int:
        return _sequence_length(self.source.semantic_ids) + _sequence_length(
            self.target.semantic_ids
        )


type TaskSample = (
    SourceAutoregressionSample
    | TargetAutoregressionSample
    | SourceToTargetSample
    | TargetToSourceSample
)


class TaskSampleStream(IterableDataset[TaskSample]):
    def __init__(
        self,
        dataset_factory: DatasetFactoryConfig,
        *,
        tasks: TaskConfig,
    ) -> None:
        self.dataset_factory = dataset_factory
        enabled = _enabled_tasks(tasks)
        self.autoregression = Task.AUTOREGRESSION in enabled
        self.translation = Task.TRANSLATION in enabled
        if not self.autoregression and not self.translation:
            raise ValueError("tasks.enabled must contain at least one task.")
        self.weights = tasks.weights
        _validate_positive_enabled_weight(
            (
                TaskFamily.SOURCE_AR,
                TaskFamily.TARGET_AR,
            )
            if self.autoregression
            else (),
            (
                TaskFamily.SOURCE_TO_TARGET,
                TaskFamily.TARGET_TO_SOURCE,
            )
            if self.translation
            else (),
            weights=self.weights,
        )

    def __iter__(self) -> Iterator[TaskSample]:
        source = training_dataset(self.dataset_factory)
        accumulators = {family: 0.0 for family in TaskFamily}
        for sample in source:
            pair = longcat_pair_from_sample(sample)
            candidates: list[tuple[TaskFamily, TaskSample]] = []
            if self.autoregression:
                candidates.extend(
                    (
                        (
                            TaskFamily.SOURCE_AR,
                            SourceAutoregressionSample(pair.source),
                        ),
                        (
                            TaskFamily.TARGET_AR,
                            TargetAutoregressionSample(pair.target),
                        ),
                    )
                )
            if self.translation:
                candidates.extend(
                    (
                        (
                            TaskFamily.SOURCE_TO_TARGET,
                            SourceToTargetSample(source=pair.source, target=pair.target),
                        ),
                        (
                            TaskFamily.TARGET_TO_SOURCE,
                            TargetToSourceSample(source=pair.target, target=pair.source),
                        ),
                    )
                )
            for family, task_sample in candidates:
                accumulators[family] += self.weights.weight(family)
                count = int(accumulators[family])
                accumulators[family] -= count
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
        bpe_tokenizer: object,
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
        bpe_tokenizer: object,
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
            source_audio=_collate_sides(
                [_source_side(sample) for sample in samples],
                device=self.device,
            ),
            target_audio=_collate_sides(
                [_target_side(sample) for sample in samples],
                device=self.device,
            ),
            task_family=torch.tensor(
                [_task_family(sample).id for sample in samples],
                dtype=torch.long,
                device=self.device,
            ),
        )


def task_sample_length(sample: TaskSample) -> int:
    return sample.length


def _encode_task_sample(
    sample: TaskSample,
    tokenizer: object,
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


def _source_side(sample: TaskSample) -> LongCatSide | None:
    if isinstance(sample, SourceAutoregressionSample | TargetAutoregressionSample):
        return None
    if isinstance(sample, SourceToTargetSample | TargetToSourceSample):
        return sample.source
    raise TypeError("unknown task sample type.")


def _target_side(sample: TaskSample) -> LongCatSide:
    if isinstance(sample, SourceAutoregressionSample | TargetAutoregressionSample):
        return sample.target
    if isinstance(sample, SourceToTargetSample | TargetToSourceSample):
        return sample.target
    raise TypeError("unknown task sample type.")


def _task_family(sample: TaskSample) -> TaskFamily:
    if isinstance(sample, SourceAutoregressionSample):
        return TaskFamily.SOURCE_AR
    if isinstance(sample, TargetAutoregressionSample):
        return TaskFamily.TARGET_AR
    if isinstance(sample, SourceToTargetSample):
        return TaskFamily.SOURCE_TO_TARGET
    if isinstance(sample, TargetToSourceSample):
        return TaskFamily.TARGET_TO_SOURCE
    raise TypeError("unknown task sample type.")


def _collate_sides(
    sides: Sequence[LongCatSide | None],
    *,
    device: TorchDevice,
) -> LongCatBatchSide | None:
    present = [side for side in sides if side is not None]
    if not present:
        return None

    semantic_rows = [_semantic_row(side.semantic_ids) for side in present]
    acoustic_rows = [_acoustic_row(side.acoustic_ids) for side in present]
    for semantic, acoustic in zip(semantic_rows, acoustic_rows, strict=True):
        if semantic.numel() != acoustic.size(-1):
            raise ValueError("LongCat semantic and acoustic lengths must match.")

    max_semantic_length = max(row.numel() for row in semantic_rows)
    max_acoustic_length = max(row.size(-1) for row in acoustic_rows)
    codebook_count = acoustic_rows[0].size(0)
    if any(row.size(0) != codebook_count for row in acoustic_rows):
        raise ValueError("LongCat acoustic codebook count must be consistent within a batch.")

    semantic_ids = torch.zeros(
        (len(sides), max_semantic_length),
        dtype=torch.long,
        device=device,
    )
    semantic_mask = torch.zeros(
        (len(sides), max_semantic_length),
        dtype=torch.bool,
        device=device,
    )
    acoustic_ids = torch.zeros(
        (len(sides), codebook_count, max_acoustic_length),
        dtype=torch.long,
        device=device,
    )
    acoustic_mask = torch.zeros(
        (len(sides), max_acoustic_length),
        dtype=torch.bool,
        device=device,
    )

    present_index = 0
    for row_index, side in enumerate(sides):
        if side is None:
            continue
        semantic = semantic_rows[present_index].to(device=device)
        acoustic = acoustic_rows[present_index].to(device=device)
        present_index += 1
        semantic_ids[row_index, : semantic.numel()] = semantic
        semantic_mask[row_index, : semantic.numel()] = True
        acoustic_ids[row_index, :, : acoustic.size(-1)] = acoustic
        acoustic_mask[row_index, : acoustic.size(-1)] = True

    return LongCatBatchSide(
        semantic_ids=semantic_ids,
        semantic_mask=semantic_mask,
        acoustic_ids=acoustic_ids,
        acoustic_mask=acoustic_mask,
    )


def _semantic_row(ids: Tensor) -> Tensor:
    if ids.dim() == 0:
        raise ValueError("LongCat semantic ids must have a time dimension.")
    return ids.reshape(-1).detach().to(dtype=torch.long)


def _acoustic_row(ids: Tensor) -> Tensor:
    if ids.dim() == 3 and ids.size(0) == 1:
        ids = ids.squeeze(0)
    if ids.dim() != 2:
        raise ValueError("LongCat acoustic ids must have shape [nq, time].")
    return ids.detach().to(dtype=torch.long)


def _enabled_tasks(tasks: TaskConfig) -> frozenset[Task]:
    enabled: set[Task] = set()
    for name in tasks.enabled:
        enabled.add(Task(name))
    return frozenset(enabled)


def _validate_positive_enabled_weight(
    ar_families: Sequence[TaskFamily],
    translation_families: Sequence[TaskFamily],
    *,
    weights: object,
) -> None:
    enabled_families = (*ar_families, *translation_families)
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
