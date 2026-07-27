from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, TypedDict

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor


class ValidationRecord(TypedDict):
    step: int
    metrics: dict[str, float]


class ValidationReport(TypedDict):
    sanity: list[ValidationRecord]
    interval: list[ValidationRecord]


class ValidationSummary(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.sanity: list[ValidationRecord] = []
        self.interval: list[ValidationRecord] = []

    def on_validation_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        metrics = {
            name: _scalar(name, value)
            for name, value in sorted(trainer.callback_metrics.items())
            if name.startswith("val/")
        }
        if not metrics:
            raise RuntimeError("validation completed without val/* callback metrics.")
        record = ValidationRecord(step=trainer.global_step, metrics=metrics)
        values = self.sanity if trainer.sanity_checking else self.interval
        _append(values, record)

    def report(self) -> ValidationReport:
        return {
            "sanity": _copy(self.sanity),
            "interval": _copy(self.interval),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "sanity": _copy(self.sanity),
            "interval": _copy(self.interval),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.sanity = _records(state_dict, "sanity")
        self.interval = _records(state_dict, "interval")


def _scalar(name: str, value: Tensor) -> float:
    if value.numel() != 1:
        raise ValueError(f"validation metric {name!r} must be scalar.")
    return float(value.detach().item())


def _append(values: list[ValidationRecord], record: ValidationRecord) -> None:
    if values and record["step"] < values[-1]["step"]:
        raise RuntimeError("validation steps must not move backwards.")
    if values and record["step"] == values[-1]["step"]:
        values[-1] = record
        return
    values.append(record)


def _copy(values: list[ValidationRecord]) -> list[ValidationRecord]:
    return [
        ValidationRecord(step=value["step"], metrics=dict(value["metrics"]))
        for value in values
    ]


def _records(state: Mapping[str, Any], name: str) -> list[ValidationRecord]:
    values = state.get(name)
    if not isinstance(values, list):
        raise TypeError(f"validation callback state {name!r} must be a list.")
    records = [_record(value) for value in values]
    if any(
        current["step"] < previous["step"]
        for previous, current in zip(records, records[1:])
    ):
        raise ValueError(f"validation callback state {name!r} must be step ordered.")
    return records


def _record(value: object) -> ValidationRecord:
    if not isinstance(value, Mapping) or set(value) != {"step", "metrics"}:
        raise TypeError("validation callback records require step and metrics.")
    step = value["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise TypeError("validation callback record step must be non-negative integer.")
    raw_metrics = value["metrics"]
    if not isinstance(raw_metrics, Mapping):
        raise TypeError("validation callback record metrics must be a mapping.")
    metrics: dict[str, float] = {}
    for name, metric in raw_metrics.items():
        if not isinstance(name, str) or not name:
            raise TypeError(
                "validation callback metric names must be non-empty strings."
            )
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise TypeError("validation callback metrics must be numeric.")
        resolved = float(metric)
        if not math.isfinite(resolved):
            raise ValueError("validation callback metrics must be finite.")
        metrics[name] = resolved
    return ValidationRecord(step=step, metrics=metrics)
