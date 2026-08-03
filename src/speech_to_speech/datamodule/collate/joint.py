from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...loader_step import LoaderStepMode
from ..types import FusedBatch, LoaderBatch, TrainBatch, TrainInput


@runtime_checkable
class _EpochSetter(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


@dataclass(frozen=True)
class LoaderSchedule:
    weights: dict[str, float]
    accumulate_grad_batches: int = 1
    fuse_loaders_per_step: bool = False
    step_mode: str | None = None

    def __post_init__(self) -> None:
        _validate_weights(self.weights)
        if (
            isinstance(self.accumulate_grad_batches, bool)
            or not isinstance(self.accumulate_grad_batches, int)
        ):
            raise TypeError("accumulate_grad_batches must be an integer.")
        if self.accumulate_grad_batches < 1:
            raise ValueError("accumulate_grad_batches must be positive.")
        if not isinstance(self.fuse_loaders_per_step, bool):
            raise TypeError("fuse_loaders_per_step must be a boolean.")
        mode = self.mode
        if mode is LoaderStepMode.WEIGHTED_WINDOW:
            mode = None
        if mode is LoaderStepMode.FUSED_JOINT:
            if not self.fuse_loaders_per_step:
                raise ValueError("fused_joint requires fuse_loaders_per_step=true.")
            _validate_one_each_window(
                self.weights,
                self.accumulate_grad_batches,
                mode=mode,
            )
            return
        if mode is LoaderStepMode.SERIAL_JOINT:
            if self.fuse_loaders_per_step:
                raise ValueError("serial_joint requires fuse_loaders_per_step=false.")
            _validate_one_each_window(
                self.weights,
                self.accumulate_grad_batches,
                mode=mode,
            )
            return
        if self.accumulate_grad_batches > 1 or self.fuse_loaders_per_step:
            _accumulation_window(self.weights, self.accumulate_grad_batches)

    @property
    def mode(self) -> LoaderStepMode | None:
        if self.step_mode is None:
            return None
        try:
            return LoaderStepMode(self.step_mode)
        except ValueError as error:
            raise ValueError(
                "step_mode must be 'weighted_window', 'fused_joint', or "
                "'serial_joint'."
            ) from error


class ScheduledDataLoader:
    def __init__(
        self,
        loaders: Mapping[str, Iterable[TrainInput]],
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

    def __iter__(self) -> Iterator[TrainBatch]:
        keys = tuple(self.schedule.weights)
        weights = self.schedule.weights
        iterators = {key: iter(self.loaders[key]) for key in keys}
        cycles = {key: 0 for key in keys}
        if self.schedule.mode is LoaderStepMode.FUSED_JOINT:
            window = _one_each_window(weights)
            while True:
                yield FusedBatch(
                    tuple(
                        _next_batch(key, iterators, self.loaders, cycles)
                        for key in window
                    ),
                    loader_names=window,
                    loss_weights=_loss_weights(weights, window),
                )
        if self.schedule.mode is LoaderStepMode.SERIAL_JOINT:
            window = _one_each_window(weights)
            loss_weights = _loss_weights(weights, window)
            while True:
                for key, loss_weight in zip(window, loss_weights):
                    yield LoaderBatch(
                        _next_batch(key, iterators, self.loaders, cycles),
                        key,
                        len(window) * loss_weight,
                    )
        if self.schedule.accumulate_grad_batches > 1:
            credits = [0.0 for _ in keys]
            while True:
                window = _accumulation_window(
                    weights,
                    self.schedule.accumulate_grad_batches,
                    credits=credits,
                )
                if self.schedule.fuse_loaders_per_step:
                    yield FusedBatch(
                        tuple(
                            _next_batch(key, iterators, self.loaders, cycles)
                            for key in window
                        ),
                        loader_names=window,
                    )
                    continue
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


def _one_each_window(weights: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(key for key, weight in weights.items() if weight > 0)


def _loss_weights(
    weights: Mapping[str, float],
    window: tuple[str, ...],
) -> tuple[float, ...]:
    total = sum(weights[key] for key in window)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("joint loader loss weights must have a positive total.")
    return tuple(weights[key] / total for key in window)


def _validate_one_each_window(
    weights: Mapping[str, float],
    accumulate_grad_batches: int,
    *,
    mode: LoaderStepMode,
) -> None:
    loader_count = len(_one_each_window(weights))
    if accumulate_grad_batches != loader_count:
        raise ValueError(
            f"{mode.value} requires accumulate_grad_batches to equal the "
            f"number of positive loaders ({loader_count})."
        )


def _accumulation_window(
    weights: Mapping[str, float],
    accumulate_grad_batches: int,
    *,
    credits: list[float] | None = None,
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
    if credits is None:
        credits = [0.0 for _ in keys]
    if len(credits) != len(keys):
        raise ValueError("accumulation credits must align with loader weights.")
    for index, target in enumerate(targets):
        credits[index] += target - counts[index]
    available = set(range(len(keys)))
    for _ in range(remaining):
        index = max(available, key=lambda value: (credits[value], -value))
        available.remove(index)
        credits[index] -= 1.0
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
    iterators: dict[str, Iterator[TrainInput]],
    loaders: Mapping[str, Iterable[TrainInput]],
    cycles: dict[str, int],
) -> TrainInput:
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


def _set_epoch(loader: Iterable[TrainInput], epoch: int) -> None:
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
