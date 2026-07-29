from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .types import ConcreteTrainInput


@runtime_checkable
class _EpochSetter(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


@dataclass(frozen=True)
class LoaderSchedule:
    weights: dict[str, float]
    accumulate_grad_batches: int = 1

    def __post_init__(self) -> None:
        _validate_weights(self.weights)
        if (
            isinstance(self.accumulate_grad_batches, bool)
            or not isinstance(self.accumulate_grad_batches, int)
        ):
            raise TypeError("accumulate_grad_batches must be an integer.")
        if self.accumulate_grad_batches < 1:
            raise ValueError("accumulate_grad_batches must be positive.")
        if self.accumulate_grad_batches > 1:
            _accumulation_window(self.weights, self.accumulate_grad_batches)


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

    def __iter__(self) -> Iterator[ConcreteTrainInput]:
        keys = tuple(self.schedule.weights)
        weights = self.schedule.weights
        iterators = {key: iter(self.loaders[key]) for key in keys}
        cycles = {key: 0 for key in keys}
        if self.schedule.accumulate_grad_batches > 1:
            window = _accumulation_window(
                weights,
                self.schedule.accumulate_grad_batches,
            )
            while True:
                for key in window:
                    yield _next_batch(key, iterators, self.loaders, cycles)

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


def _accumulation_window(
    weights: Mapping[str, float],
    accumulate_grad_batches: int,
) -> tuple[str, ...]:
    keys = tuple(weights)
    total = sum(weights.values())
    targets = [weights[key] * accumulate_grad_batches / total for key in keys]
    if any(target < 1 for target in targets if target > 0):
        raise ValueError(
            "accumulate_grad_batches is too small for fixed loader weights; each "
            "non-zero loader must receive at least one microbatch."
        )
    counts = [math.floor(target) for target in targets]
    remaining = accumulate_grad_batches - sum(counts)
    order = sorted(
        range(len(keys)),
        key=lambda index: (targets[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return _interleave(keys, counts)


def _interleave(keys: tuple[str, ...], counts: list[int]) -> tuple[str, ...]:
    total = sum(counts)
    credits = [0 for _ in counts]
    remaining = list(counts)
    result = []
    for _ in range(total):
        for index, count in enumerate(counts):
            credits[index] += count
        selected = max(
            (index for index, count in enumerate(remaining) if count > 0),
            key=lambda index: (credits[index], -index),
        )
        credits[selected] -= total
        remaining[selected] -= 1
        result.append(keys[selected])
    return tuple(result)


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
