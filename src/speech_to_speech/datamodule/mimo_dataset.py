"""Map-style task sampling for Kimi-style MIMO pretraining.

The regular speech datamodule owns serialized single-stream examples.  MIMO
needs a small, explicit bridge from prepared aligned segments to the seven
Kimi objectives.  This module keeps that bridge independent of codecs and
tokenizers: callers can prepare :class:`MimoSegment` values from any source,
then get deterministic weighted task sampling and optional length clipping.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch.utils.data import Dataset

from .mimo import MimoSample
from .mimo_tasks import (
    KIMI_PRETRAIN_TASK_WEIGHTS,
    MimoSegment,
    MimoSpecialTokens,
    MimoTask,
    build_mimo_sample,
)


class SegmentSource(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> MimoSegment: ...


@dataclass(frozen=True)
class MimoDatasetConfig:
    """Sampling controls persisted alongside a MIMO experiment."""

    samples_per_epoch: int | None = None
    seed: int = 0
    max_sequence_length: int | None = None
    task_weights: dict[MimoTask, float] | None = None

    def __post_init__(self) -> None:
        if self.samples_per_epoch is not None:
            _positive_int(self.samples_per_epoch, "samples_per_epoch")
        _non_negative_int(self.seed, "seed")
        if self.max_sequence_length is not None:
            _positive_int(self.max_sequence_length, "max_sequence_length")
        if self.task_weights is not None:
            _weights(self.task_weights)

    @property
    def resolved_task_weights(self) -> dict[MimoTask, float]:
        return dict(KIMI_PRETRAIN_TASK_WEIGHTS if self.task_weights is None else self.task_weights)


class MimoTaskDataset(Dataset[MimoSample]):
    """Deterministically sample weighted Kimi tasks from aligned segments.

    A contextual task consumes two adjacent segments from the same recording;
    one-segment tasks consume one segment.  The index-to-task mapping is
    deterministic for a fixed ``seed`` and ``epoch``, which makes worker and
    distributed replay reproducible without sharing mutable RNG state.
    """

    def __init__(
        self,
        segments: SegmentSource | Sequence[MimoSegment],
        special: MimoSpecialTokens,
        *,
        config: MimoDatasetConfig | None = None,
        ignore_index: int = -100,
    ) -> None:
        if not hasattr(segments, "__len__") or not hasattr(segments, "__getitem__"):
            raise TypeError("segments must be a sized indexable source.")
        if not isinstance(special, MimoSpecialTokens):
            raise TypeError("special must be a MimoSpecialTokens value.")
        if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
            raise TypeError("ignore_index must be an integer.")
        self.segments = cast(SegmentSource, segments)
        self.special = special
        self.config = MimoDatasetConfig() if config is None else config
        self.ignore_index = ignore_index
        self._epoch = 0
        count = len(self.segments)
        if count < 1:
            raise ValueError("MimoTaskDataset requires at least one segment.")
        self._windows = _windows(self.segments)
        contextual = set(self.config.resolved_task_weights) & _CONTEXT_TASKS
        if contextual and not self._windows[2]:
            names = ", ".join(sorted(task.value for task in contextual))
            raise ValueError(
                "MIMO contextual tasks require at least one pair of segments "
                "with matching recording_id and consecutive segment_index; "
                f"none is available for: {names}."
            )

    def __len__(self) -> int:
        requested = self.config.samples_per_epoch
        return len(self.segments) if requested is None else requested

    def __getitem__(self, index: int) -> MimoSample:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("MimoTaskDataset index must be an integer.")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = random.Random(self._seed(index))
        task = _choose_task(rng, self.config.resolved_task_weights)
        window_size = 2 if task in _CONTEXT_TASKS else 1
        candidates = self._windows[window_size]
        if not candidates:
            raise ValueError(
                f"MimoTaskDataset has no {window_size}-segment window for {task.value}."
            )
        selected = candidates[rng.randrange(len(candidates))]
        sample = build_mimo_sample(
            task,
            (self.segments[position] for position in selected),
            self.special,
            ignore_index=self.ignore_index,
        )
        return _clip(sample, self.config.max_sequence_length)

    def set_epoch(self, epoch: int) -> None:
        _non_negative_int(epoch, "epoch")
        self._epoch = epoch

    def task_for_index(self, index: int) -> MimoTask:
        """Return the deterministic task selected for an index without building it."""

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return _choose_task(random.Random(self._seed(index)), self.config.resolved_task_weights)

    def _seed(self, index: int) -> int:
        # Mix the epoch and index with fixed odd constants; avoid Python's
        # process-randomized hash so workers reproduce the same sequence.
        return (
            int(self.config.seed)
            + 0x9E3779B1 * (self._epoch + 1)
            + 0x85EBCA77 * (index + 1)
        ) & 0xFFFFFFFF


class ToyMimoSegmentDataset(Dataset[MimoSegment]):
    """Small deterministic segment source for local Hydra/CI smoke runs."""

    def __init__(
        self,
        *,
        samples: int = 8,
        text_tokens: int = 6,
        audio_tokens: int = 8,
        text_vocab_size: int = 128,
        audio_vocab_size: int = 256,
        feature_dim: int | None = None,
    ) -> None:
        for name, value in (
            ("samples", samples),
            ("text_tokens", text_tokens),
            ("audio_tokens", audio_tokens),
            ("text_vocab_size", text_vocab_size),
            ("audio_vocab_size", audio_vocab_size),
        ):
            _positive_int(value, name)
        if feature_dim is not None:
            _positive_int(feature_dim, "feature_dim")
        self.samples = samples
        self.text_tokens = text_tokens
        self.audio_tokens = audio_tokens
        self.text_vocab_size = text_vocab_size
        self.audio_vocab_size = audio_vocab_size
        self.feature_dim = feature_dim

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> MimoSegment:
        if index < 0:
            index += self.samples
        if index < 0 or index >= self.samples:
            raise IndexError(index)
        text = (torch.arange(self.text_tokens, dtype=torch.long) + index + 3) % self.text_vocab_size
        audio = (torch.arange(self.audio_tokens, dtype=torch.long) + index + 5) % self.audio_vocab_size
        features = None
        if self.feature_dim is not None:
            base = torch.arange(self.audio_tokens * self.feature_dim, dtype=torch.float32)
            features = base.view(self.audio_tokens, self.feature_dim) / max(1, self.feature_dim)
        # Consecutive toy rows form one recording so contextual tasks are
        # available even when samples_per_epoch exceeds the source length.
        return MimoSegment(
            text_input_ids=text,
            audio_input_ids=audio,
            audio_features=features,
            recording_id="toy",
            segment_index=index,
        )


class JsonlMimoSegmentDataset(Dataset[MimoSegment]):
    """Read prepared local-vocabulary MIMO segments from JSONL lazily.

    Each line is an object with ``text_input_ids`` and ``audio_input_ids``.
    Optional fields are ``audio_features``, ``recording_id`` and
    ``segment_index``.  Byte offsets are indexed once, while tensors are
    materialized only for requested rows so a prepared corpus is not retained
    twice in host memory.
    """

    _FIELDS = frozenset(
        {
            "text_input_ids",
            "audio_input_ids",
            "audio_features",
            "recording_id",
            "segment_index",
        }
    )

    def __init__(self, path: str | Path) -> None:
        if not isinstance(path, (str, Path)):
            raise TypeError("MIMO JSONL path must be a string or Path.")
        self.path = Path(path).expanduser()
        if not self.path.is_file():
            raise FileNotFoundError(f"MIMO JSONL path is not a file: {self.path}.")
        offsets: list[tuple[int, int]] = []
        with self.path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    raise ValueError(f"MIMO JSONL line {line_number} is blank.")
                offsets.append((offset, line_number))
        if not offsets:
            raise ValueError(f"MIMO JSONL file is empty: {self.path}.")
        self._offsets = tuple(offsets)

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> MimoSegment:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("MIMO JSONL index must be an integer.")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        offset, line_number = self._offsets[index]
        with self.path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.readline()
        try:
            line = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"MIMO JSONL line {line_number} is not valid UTF-8."
            ) from error
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on MIMO line {line_number}.") from error
        if not isinstance(value, dict):
            raise ValueError(f"MIMO JSONL line {line_number} must be an object.")
        unknown = set(value) - self._FIELDS
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(
                f"MIMO JSONL line {line_number} has unknown fields: {names}."
            )
        text = _ids(value.get("text_input_ids"), line_number, "text_input_ids")
        audio = _ids(value.get("audio_input_ids"), line_number, "audio_input_ids")
        features = _features(value.get("audio_features"), line_number, audio.numel())
        return MimoSegment(
            text_input_ids=text,
            audio_input_ids=audio,
            audio_features=features,
            recording_id=_recording_id(value.get("recording_id"), line_number),
            segment_index=_segment_index(value.get("segment_index"), line_number),
        )


_CONTEXT_TASKS = frozenset(
    {
        MimoTask.AUDIO_TO_NEXT_SEMANTIC,
        MimoTask.AUDIO_TO_NEXT_TEXT,
        MimoTask.AUDIO_TO_NEXT_SEMANTIC_AND_TEXT,
    }
)


def _windows(source: SegmentSource) -> dict[int, tuple[tuple[int, ...], ...]]:
    one = tuple((index,) for index in range(len(source)))
    keyed: dict[tuple[str, int], int] = {}
    for index in range(len(source)):
        segment = source[index]
        if not isinstance(segment, MimoSegment):
            raise TypeError("segment source must return MimoSegment values.")
        if segment.recording_id is None or segment.segment_index is None:
            continue
        key = (segment.recording_id, segment.segment_index)
        if key in keyed:
            raise ValueError(
                "segment source contains duplicate recording_id/segment_index metadata."
            )
        keyed[key] = index
    two = [
        (index, keyed[(recording_id, segment_index + 1)])
        for (recording_id, segment_index), index in keyed.items()
        if (recording_id, segment_index + 1) in keyed
    ]
    return {1: one, 2: tuple(two)}


def _choose_task(rng: random.Random, weights: dict[MimoTask, float]) -> MimoTask:
    tasks = tuple(weights)
    total = sum(weights.values())
    draw = rng.random() * total
    for task in tasks:
        draw -= weights[task]
        if draw < 0:
            return task
    return tasks[-1]


def _clip(sample: MimoSample, limit: int | None) -> MimoSample:
    if limit is None or sample.text_input_ids.numel() <= limit:
        return sample
    if limit < 2:
        raise ValueError("max_sequence_length must leave at least two causal positions.")
    stop = limit
    audio_features = None if sample.audio_features is None else sample.audio_features[:stop]
    audio_feature_mask = None if sample.audio_feature_mask is None else sample.audio_feature_mask[:stop]
    clipped = MimoSample(
        text_input_ids=sample.text_input_ids[:stop],
        audio_input_ids=sample.audio_input_ids[:stop],
        text_labels=sample.text_labels[:stop],
        audio_labels=sample.audio_labels[:stop],
        attention_mask=None if sample.attention_mask is None else sample.attention_mask[:stop],
        text_loss_mask=None if sample.text_loss_mask is None else sample.text_loss_mask[:stop],
        audio_loss_mask=None if sample.audio_loss_mask is None else sample.audio_loss_mask[:stop],
        audio_features=audio_features,
        audio_feature_mask=audio_feature_mask,
        task_id=sample.task_id,
        recording_id=sample.recording_id,
        ignore_index=sample.ignore_index,
    )
    if not bool(clipped.effective_text_loss_mask[1:].any() | clipped.effective_audio_loss_mask[1:].any()):
        raise ValueError("max_sequence_length clips all supervised MIMO targets.")
    return clipped


def _weights(value: object) -> dict[MimoTask, float]:
    if not isinstance(value, dict) or not value:
        raise TypeError("MIMO task_weights must be a non-empty mapping.")
    result: dict[MimoTask, float] = {}
    for raw_task, raw_weight in value.items():
        task = raw_task if isinstance(raw_task, MimoTask) else MimoTask(str(raw_task))
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise TypeError(f"MIMO task weight for {task.value} must be numeric.")
        weight = float(raw_weight)
        if not torch.isfinite(torch.tensor(weight)) or weight < 0:
            raise ValueError(f"MIMO task weight for {task.value} must be finite and non-negative.")
        if weight > 0:
            result[task] = weight
    if not result:
        raise ValueError("MIMO task_weights must contain a positive weight.")
    return result


def _positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _non_negative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _ids(value: object, line_number: int, name: str) -> torch.Tensor:
    if not isinstance(value, list) or not value:
        raise ValueError(f"MIMO JSONL line {line_number} {name} must be a non-empty list.")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(
            f"MIMO JSONL line {line_number} {name} must contain non-negative integers."
        )
    return torch.tensor(value, dtype=torch.long)


def _features(
    value: object,
    line_number: int,
    audio_tokens: int,
) -> torch.Tensor | None:
    if value is None:
        return None
    try:
        result = torch.as_tensor(value, dtype=torch.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"MIMO JSONL line {line_number} audio_features must be numeric."
        ) from error
    if result.dim() != 2 or result.size(0) != audio_tokens or result.size(1) < 1:
        raise ValueError(
            f"MIMO JSONL line {line_number} audio_features must have shape "
            "[audio_tokens, feature_dim]."
        )
    return result


def _recording_id(value: object, line_number: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"MIMO JSONL line {line_number} recording_id must be a non-empty string."
        )
    return value


def _segment_index(value: object, line_number: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"MIMO JSONL line {line_number} segment_index must be non-negative."
        )
    return value


__all__ = [
    "JsonlMimoSegmentDataset",
    "MimoDatasetConfig",
    "MimoTaskDataset",
    "ToyMimoSegmentDataset",
]
