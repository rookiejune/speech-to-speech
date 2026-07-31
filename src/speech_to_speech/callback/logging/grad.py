from __future__ import annotations

from collections.abc import Mapping, Sequence

from anytrain.lightning import (
    GradientComparison,
    GradientProbe,
    GradientProbeLoggerCallback,
)
from lightning import LightningModule, Trainer

from ..interval import TrainInterval


class GradLogger(GradientProbeLoggerCallback):
    def __init__(
        self,
        comparisons: Sequence[GradientComparison],
        probes: Sequence[GradientProbe] | Mapping[str, Sequence[str]],
        every_n_steps: int = 5_000,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            comparisons,
            probes,
            every_n_steps=every_n_steps,
            eps=eps,
        )
        self.interval = TrainInterval(every_n_steps=every_n_steps)
        self._run_current_batch = False

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        del batch_idx
        del batch
        self._run_current_batch = self.interval.should_run(int(trainer.global_step))

    def should_run(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> bool:
        del trainer, pl_module
        return self._run_current_batch

    def state_dict(self) -> dict[str, dict[str, int | None]]:
        return {"interval": self.interval.state_dict()}

    def load_state_dict(self, state_dict: dict[str, dict[str, int | None]]) -> None:
        self.interval.load_state_dict(state_dict.get("interval", {}))
