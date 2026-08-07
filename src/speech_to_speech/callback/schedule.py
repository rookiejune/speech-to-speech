from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from anytrain.lightning.schedule import (
    Constant,
    Cosine,
    LRSchedule,
    Linear,
    Phase,
    ScheduleRuntime,
    UnitBatch,
    UnitClock,
    require_unit_name,
)
from speech_to_speech.datamodule.batch import (
    FusedBatch,
    LoaderBatch,
)
from speech_to_speech.datamodule.sample import RawSpeechBatch

SUPPORTED_UNIT_NAMES = frozenset({"tokens", "frames", "audio_seconds"})


class LRCurveConfig(Protocol):
    @property
    def type(self) -> str: ...

    @property
    def value(self) -> float | None: ...

    @property
    def start(self) -> float | None: ...

    @property
    def end(self) -> float | None: ...


class PhaseConfig(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def duration(self) -> float: ...

    @property
    def lr(self) -> LRCurveConfig: ...


class ScheduleConfig(Protocol):
    @property
    def unit(self) -> str: ...

    @property
    def log_every_n_units(self) -> float | None: ...

    @property
    def measure_window_batches(self) -> int: ...

    @property
    def sync_cuda(self) -> bool: ...

    @property
    def sync_distributed(self) -> bool: ...

    @property
    def allow_external_lr_changes(self) -> bool: ...

    @property
    def stop_at(self) -> float | None: ...

    @property
    def stop_at_end(self) -> bool: ...

    @property
    def phases(self) -> Sequence[PhaseConfig]: ...


@runtime_checkable
class TrainingUnitProvider(Protocol):
    def training_units(self, unit: str) -> tuple[float, float | None]: ...


@dataclass(frozen=True)
class BatchUnits:
    unit: str = "tokens"

    def __post_init__(self) -> None:
        require_unit_name(self.unit)
        if self.unit not in SUPPORTED_UNIT_NAMES:
            raise ValueError(
                "speech-to-speech unit schedules support units: "
                + ", ".join(sorted(SUPPORTED_UNIT_NAMES))
            )

    def __call__(
        self,
        *,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> UnitBatch:
        del trainer, pl_module, outputs, batch_idx
        return _batch_units(batch, self.unit)


def build_unit_schedule(config: ScheduleConfig) -> ScheduleRuntime:
    schedule = LRSchedule(
        clock=UnitClock(
            unit=config.unit,
            provider=BatchUnits(config.unit),
            log_every_n_units=config.log_every_n_units,
            measure_window_batches=config.measure_window_batches,
            sync_cuda=config.sync_cuda,
            sync_distributed=config.sync_distributed,
            allow_external_lr_changes=config.allow_external_lr_changes,
        ),
        phases=tuple(
            Phase(phase.name, duration=phase.duration, lr=_curve(phase.lr))
            for phase in config.phases
        ),
        stop_at=config.stop_at,
        stop_at_end=config.stop_at_end,
    )
    return ScheduleRuntime(schedule)


def _batch_units(batch: object, unit: str) -> UnitBatch:
    if isinstance(batch, LoaderBatch):
        return _batch_units(batch.batch, unit)
    if isinstance(batch, FusedBatch):
        return _fused_units(batch, unit)
    if isinstance(batch, RawSpeechBatch):
        raise TypeError(
            "unit schedule requires materialized ModelBatch inputs; raw waveform "
            "batches are not unit-counted by the training callback."
        )
    if not isinstance(batch, TrainingUnitProvider):
        raise TypeError(
            "unit schedule expects a batch with training_units(), "
            f"got {type(batch)!r}."
        )
    valid, padded = batch.training_units(unit)
    return UnitBatch(valid=valid, padded=padded, unit=unit)


def _fused_units(batch: FusedBatch, unit: str) -> UnitBatch:
    batches = tuple(_batch_units(child, unit) for child in batch.batches)
    valid = sum(item.valid for item in batches)
    padded = _sum_optional(tuple(item.padded for item in batches))
    return UnitBatch(valid=valid, padded=padded, unit=unit)


def _sum_optional(values: Sequence[float | None]) -> float | None:
    total = 0.0
    for value in values:
        if value is None:
            return None
        total += float(value)
    return total


def _curve(config: LRCurveConfig) -> Constant | Linear | Cosine:
    curve_type = config.type
    if curve_type == "constant":
        return Constant() if config.value is None else Constant(config.value)
    if curve_type == "linear":
        start = 0.0 if config.start is None else config.start
        end = 1.0 if config.end is None else config.end
        return Linear(start, end)
    if curve_type == "cosine":
        start = 1.0 if config.start is None else config.start
        end = 0.1 if config.end is None else config.end
        return Cosine(start, end)
    raise ValueError(f"unsupported lr curve type: {curve_type!r}.")


__all__ = [
    "BatchUnits",
    "SUPPORTED_UNIT_NAMES",
    "TrainingUnitProvider",
    "build_unit_schedule",
]
