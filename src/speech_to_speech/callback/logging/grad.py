from __future__ import annotations

from anytrain.lightning import GradientLoggerCallback, GradientNormLoggerCallback
from lightning import LightningModule, Trainer

from ..interval import TrainInterval


class GradLogger(GradientLoggerCallback):
    def __init__(
        self,
        loss_pair: tuple[str, str],
        parameter_name: str,
        every_n_steps: int = 5_000,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            loss_pair,
            parameter_name,
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


class GradNormLogger(GradientNormLoggerCallback):
    def __init__(
        self,
        tag: str = "grad_norm",
        every_n_steps: int = 100,
    ) -> None:
        super().__init__(tag=tag, every_n_steps=every_n_steps)
        self.interval = TrainInterval(every_n_steps=every_n_steps)
        self._run_current_batch = False

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        del batch_idx, batch
        self._run_current_batch = self.interval.should_run(int(trainer.global_step))

    def should_run(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> bool:
        del trainer, pl_module
        return getattr(self, "_run_current_batch", False)

    def state_dict(self) -> dict[str, dict[str, int | None]]:
        return {"interval": self.interval.state_dict()}

    def load_state_dict(self, state_dict: dict[str, dict[str, int | None]]) -> None:
        self.interval.load_state_dict(state_dict.get("interval", {}))
