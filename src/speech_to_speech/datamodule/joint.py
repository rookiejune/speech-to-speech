from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .types import ConcreteTrainInput, TrainInputBatch


@runtime_checkable
class _EpochSetter(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


@dataclass(frozen=True)
class LoaderSchedule:
    weights: dict[str, float]
    batches_per_step: int = 1

    def __post_init__(self) -> None:
        _validate_weights(self.weights)
        if (
            isinstance(self.batches_per_step, bool)
            or not isinstance(self.batches_per_step, int)
        ):
            raise TypeError("batches_per_step must be an integer.")
        if self.batches_per_step < 1:
            raise ValueError("batches_per_step must be positive.")
        if self.batches_per_step > 1:
            _allocate_loaders(self.weights, self.batches_per_step)


class ScheduledDataLoader:
    def __init__(
        self,
        loaders: Mapping[str, Iterable[ConcreteTrainInput]],
        schedule: LoaderSchedule,
    ) -> None:
        missing = set(schedule.weights) - set(loaders)
        if missing:
            raise ValueError(
                "scheduled loaders are missing: " + ", ".join(sorted(missing))
            )
        extra = set(loaders) - set(schedule.weights)
        if extra:
            raise ValueError(
                "loader weights are missing: " + ", ".join(sorted(extra))
            )
        self.loaders = dict(loaders)
        self.schedule = schedule

    def __iter__(self) -> Iterator[TrainInputBatch]:
        keys = tuple(self.schedule.weights)
        weights = self.schedule.weights
        iterators = {key: iter(self.loaders[key]) for key in keys}
        cycles = {key: 0 for key in keys}
        if self.schedule.batches_per_step > 1:
            fixed = _allocate_loaders(weights, self.schedule.batches_per_step)
            while True:
                yield tuple(
                    _next_batch(key, iterators, self.loaders, cycles)
                    for key in fixed
                )

        total = sum(weights.values())
        credits = {key: 0.0 for key in keys}
        while True:
            for key in keys:
                credits[key] += weights[key]
            selected = max(keys, key=lambda key: (credits[key], -keys.index(key)))
            credits[selected] -= total
            yield _next_batch(selected, iterators, self.loaders, cycles)


def _validate_weights(weights: Mapping[str, float]) -> None:
    if not weights:
        raise ValueError("loader weights must contain at least one loader.")
    values = list(weights.values())
    if any(not math.isfinite(weight) or weight < 0 for weight in values):
        raise ValueError("loader weights must be finite and non-negative.")
    total = sum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("loader weights must have a finite positive total.")
    if any(not key for key in weights):
        raise ValueError("loader names must not be empty.")


def _allocate_loaders(
    weights: Mapping[str, float],
    batches_per_step: int,
) -> tuple[str, ...]:
    keys = tuple(weights)
    total = sum(weights.values())
    targets = [weights[key] * batches_per_step / total for key in keys]
    if any(target < 1 for target in targets if target > 0):
        raise ValueError(
            "batches_per_step is too small for fixed loader weights; each non-zero "
            "loader must receive at least one batch."
        )
    counts = [math.floor(target) for target in targets]
    remaining = batches_per_step - sum(counts)
    order = sorted(
        range(len(keys)),
        key=lambda index: (targets[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return tuple(key for key, count in zip(keys, counts) for _ in range(count))


def _next_batch(
    key: str,
    iterators: dict[str, Iterator[ConcreteTrainInput]],
    loaders: Mapping[str, Iterable[ConcreteTrainInput]],
    cycles: dict[str, int],
) -> ConcreteTrainInput:
    try:
        return next(iterators[key])
    except StopIteration:
        cycles[key] += 1
        _set_epoch(loaders[key], cycles[key])
        iterators[key] = iter(loaders[key])
        try:
            return next(iterators[key])
        except StopIteration as error:
            raise RuntimeError(f"scheduled loader {key!r} produced no batches.") from error


def _set_epoch(loader: Iterable[ConcreteTrainInput], epoch: int) -> None:
    if isinstance(loader, _EpochSetter):
        loader.set_epoch(epoch)
        return
    batch_sampler = getattr(loader, "batch_sampler", None)
    if isinstance(batch_sampler, _EpochSetter):
        batch_sampler.set_epoch(epoch)


__all__ = [
    "LoaderSchedule",
    "ScheduledDataLoader",
]
