from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from ...loss import Outputs, loss_items
from ...reporting import window_summary


class LossSummary(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, list[float]] = {}

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        if not isinstance(outputs, Mapping):
            return
        typed_outputs = cast(Outputs, outputs)
        self._append("loss", typed_outputs["loss"])
        for name, item in loss_items(typed_outputs):
            self._append(name, item.loss)

    def report(self, window: int = 20) -> dict[str, dict[str, float | int | None]]:
        return {
            name: window_summary(values, window)
            for name, values in self.values.items()
        }

    def _append(self, name: str, value: Tensor) -> None:
        self.values.setdefault(name, []).append(float(value.detach().float().mean()))
