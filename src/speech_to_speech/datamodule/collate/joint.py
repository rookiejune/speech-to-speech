from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol, runtime_checkable

from ...loader_step import LoaderStepMode
from ..mimo import MimoBatch
from ..types import (
    FusedBatch,
    LoaderBatch,
    ModelBatch,
    RawSpeechBatch,
    TrainBatch,
    TrainInput,
)


SupervisedTokenCounter = Callable[[object], int]


@runtime_checkable
class SupervisedTokenBatch(Protocol):
    """Optional structural hook for batches with a custom token contract."""

    @property
    def supervised_token_count(self) -> int: ...


def count_supervised_tokens(batch: object, *, ignore_index: int = -100) -> int:
    """Return the number of causal target tokens in one training batch.

    ``ModelBatch`` uses ``ignore_index`` (``-100`` by default).  MIMO keeps
    independent text/audio masks; its shifted masks are the authoritative
    count because a token at position ``t`` is supervised by the logit at
    position ``t - 1``.  The recursive wrapper handling keeps this counter
    useful for callers that inspect a scheduled child batch directly.
    """
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be an integer.")
    if isinstance(batch, LoaderBatch):
        return count_supervised_tokens(batch.batch, ignore_index=ignore_index)
    if isinstance(batch, FusedBatch):
        return sum(
            count_supervised_tokens(child, ignore_index=ignore_index)
            for child in batch.batches
        )
    if isinstance(batch, ModelBatch):
        return int(batch.token_labels.ne(ignore_index).sum().item())
    if isinstance(batch, MimoBatch):
        return batch.supervised_token_count
    if isinstance(batch, SupervisedTokenBatch):
        return batch.supervised_token_count
    if isinstance(batch, RawSpeechBatch):
        raise TypeError(
            "token-weighted scheduling requires materialized token batches; "
            "raw waveform batches do not expose supervised token counts."
        )
    raise TypeError(
        "supervised token counting expects ModelBatch or MimoBatch, "
        f"got {type(batch)!r}."
    )


# Short public spelling for callers that treat the counter as a batch method.
supervised_token_count = count_supervised_tokens


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
        if mode is LoaderStepMode.TOKEN_WEIGHTED:
            if self.fuse_loaders_per_step:
                raise ValueError(
                    "token_weighted requires fuse_loaders_per_step=false."
                )
            return
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
                "step_mode must be 'weighted_window', 'token_weighted', "
                "'fused_joint', or 'serial_joint'."
            ) from error


class ScheduledDataLoader:
    def __init__(
        self,
        loaders: Mapping[str, Iterable[TrainInput]],
        schedule: LoaderSchedule,
        *,
        token_counter: SupervisedTokenCounter = count_supervised_tokens,
        synchronize_token_counts: bool = True,
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
        if not callable(token_counter):
            raise TypeError("token_counter must be callable.")
        if not isinstance(synchronize_token_counts, bool):
            raise TypeError("synchronize_token_counts must be a boolean.")
        self.token_counter = token_counter
        self.synchronize_token_counts = synchronize_token_counts

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
        if self.schedule.mode is LoaderStepMode.TOKEN_WEIGHTED:
            positive_keys = tuple(key for key in keys if weights[key] > 0)
            token_weights = {
                key: Fraction(str(weights[key])) for key in positive_keys
            }
            total_weight = sum(token_weights.values(), start=Fraction())
            deficits = {key: Fraction() for key in positive_keys}
            while True:
                selected = max(
                    positive_keys,
                    key=lambda key: (deficits[key], -positive_keys.index(key)),
                )
                batch = _next_batch(selected, iterators, self.loaders, cycles)
                local_token_count = self.token_counter(batch)
                _validate_token_count(local_token_count, selected, batch)
                token_count = (
                    _synchronized_token_count(local_token_count)
                    if self.synchronize_token_counts
                    else local_token_count
                )
                for key in positive_keys:
                    deficits[key] += token_count * token_weights[key]
                deficits[selected] -= token_count * total_weight
                yield batch
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
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, (float, int))
        or not math.isfinite(weight)
        or weight < 0
        for weight in values
    ):
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


def _validate_token_count(value: int, loader_name: str, batch: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            "token_counter must return an integer; "
            f"loader {loader_name!r} returned {type(value)!r}."
        )
    if value <= 0:
        raise ValueError(
            "token-weighted scheduling requires every batch to contain at least "
            f"one supervised token; loader {loader_name!r} produced {value} from "
            f"{type(batch)!r}."
        )


def _synchronized_token_count(value: int) -> int | Fraction:
    """Use one count on every DDP rank when a process group is active."""
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return value
    world_size = dist.get_world_size()
    if world_size == 1:
        return value
    backend = str(dist.get_backend()).lower()
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if backend == "nccl"
        else torch.device("cpu")
    )
    total = torch.tensor([value], dtype=torch.int64, device=device)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return Fraction(int(total.item()), world_size)


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
    "SupervisedTokenBatch",
    "SupervisedTokenCounter",
    "count_supervised_tokens",
    "supervised_token_count",
]
