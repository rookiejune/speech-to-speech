from __future__ import annotations

from typing import TypedDict

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
        values.append(record)

    def report(self) -> ValidationReport:
        return {
            "sanity": list(self.sanity),
            "interval": list(self.interval),
        }


def _scalar(name: str, value: Tensor) -> float:
    if value.numel() != 1:
        raise ValueError(f"validation metric {name!r} must be scalar.")
    return float(value.detach().item())
