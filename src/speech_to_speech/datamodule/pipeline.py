"""Task-sample streaming and main-process batch construction."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from anydataset import AnyDataset, MultipleAnyDataset, WeightedRandomStrategy
from torch import Tensor, device as TorchDevice
from torch.utils.data import IterableDataset

from ..config import DatasetInput, TaskConfig
from ..types import (
    AutoregressionExample,
    CausalLMBatch,
    LongCatBatchSide,
    LongCatSide,
    Task,
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
        datasets: Sequence[DatasetInput],
        *,
        cache_root: str | Path | None,
        tasks: TaskConfig,
    ) -> None:
        if not datasets:
            raise ValueError("data.datasets must contain at least one dataset.")
        self.datasets = tuple(datasets)
        self.cache_root = cache_root
        enabled = _enabled_tasks(tasks)
        self.autoregression = Task.AUTOREGRESSION in enabled
        self.translation = Task.TRANSLATION in enabled
        if not self.autoregression and not self.translation:
            raise ValueError("tasks.enabled must contain at least one task.")

    def __iter__(self) -> Iterator[TaskSample]:
        source = _build_dataset(self.datasets, cache_root=self.cache_root)
        for sample in source:
            pair = longcat_pair_from_sample(sample)
            if self.autoregression:
                yield SourceAutoregressionSample(pair.source)
                yield TargetAutoregressionSample(pair.target)
            if self.translation:
                yield SourceToTargetSample(source=pair.source, target=pair.target)
                yield TargetToSourceSample(source=pair.target, target=pair.source)


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


def _build_dataset(
    datasets: Sequence[DatasetInput],
    *,
    cache_root: str | Path | None,
) -> AnyDataset | MultipleAnyDataset:
    sources = tuple(AnyDataset(dataset, cache_root=cache_root) for dataset in datasets)
    if len(sources) == 1:
        return sources[0]
    return MultipleAnyDataset(sources, strategy=WeightedRandomStrategy())
