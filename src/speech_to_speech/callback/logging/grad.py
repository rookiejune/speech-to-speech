from __future__ import annotations

from anytrain.lightning import GradientLoggerCallback, GradientNormLoggerCallback
from lightning import LightningModule, Trainer

from ..interval import TrainInterval


class GradLogger(GradientLoggerCallback):
    def __init__(
        self,
        loss_pair: tuple[str, str],
        parameter_name: str,
        every_n_steps: int | None = 5_000,
        every_audio_seconds: float | None = None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            loss_pair,
            parameter_name,
            every_n_steps=every_n_steps,
            eps=eps,
        )
        self.interval = TrainInterval(
            every_n_steps=every_n_steps,
            every_audio_seconds=every_audio_seconds,
        )
        self._run_current_batch = False

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        del batch_idx
        self._run_current_batch = self.interval.should_run(
            trainer,
            pl_module,
            batch,
        )

    def should_run(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> bool:
        if self.interval.uses_audio_seconds:
            return self._run_current_batch
        return super().should_run(trainer, pl_module)

    def state_dict(self) -> dict[str, dict[str, float]]:
        return {"interval": self.interval.state_dict()}

    def load_state_dict(self, state_dict: dict[str, dict[str, float]]) -> None:
        self.interval.load_state_dict(state_dict.get("interval", {}))


class GradNormLogger(GradientNormLoggerCallback):
    def __init__(
        self,
        tag: str = "train/grad_norm",
        every_n_steps: int | None = 100,
        every_audio_seconds: float | None = None,
    ) -> None:
        super().__init__(tag=tag, every_n_steps=every_n_steps)
        self.interval = TrainInterval(
            every_n_steps=every_n_steps,
            every_audio_seconds=every_audio_seconds,
        )
        self._pending_log = False

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: object,
        batch_idx: int,
    ) -> None:
        del batch_idx
        self._pending_log = self._pending_log or self.interval.should_run(
            trainer,
            pl_module,
            batch,
        )

    def should_run(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> bool:
        if self.interval.uses_audio_seconds:
            return self._pending_log
        return super().should_run(trainer, pl_module)

    def on_log_attempt(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del trainer, pl_module
        if self.interval.uses_audio_seconds:
            self._pending_log = False

    def state_dict(self) -> dict[str, dict[str, float]]:
        return {"interval": self.interval.state_dict()}

    def load_state_dict(self, state_dict: dict[str, dict[str, float]]) -> None:
        self.interval.load_state_dict(state_dict.get("interval", {}))
