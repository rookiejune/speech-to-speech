from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import cast

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from ._lightning import attached_datamodule
from ..datamodule import DataModule
from ..task import Task


@dataclass(frozen=True)
class Config:
    task_weights_by_stage: list[dict[Task, float]]
    epoch_milestones: list[int]


class StageSwitcher(Callback):
    def __init__(self, config: Config) -> None:
        super().__init__()

        if len(config.task_weights_by_stage) != len(config.epoch_milestones) + 1:
            raise ValueError(
                "task_weights_by_stage must contain one more item than epoch_milestones"
            )
        if any(milestone < 1 for milestone in config.epoch_milestones) or (
            config.epoch_milestones
            != sorted(set(config.epoch_milestones))
        ):
            raise ValueError(
                "epoch_milestones must be positive and strictly increasing"
            )

        self.config = config
        self._stage: int | None = None

    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        self._set_stage(
            trainer,
            bisect_right(self.config.epoch_milestones, trainer.current_epoch),
        )

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        finished_epochs = trainer.current_epoch + 1
        self._set_stage(
            trainer,
            bisect_right(self.config.epoch_milestones, finished_epochs),
        )

    def _set_stage(self, trainer: Trainer, stage: int) -> None:
        if stage == self._stage:
            return
        self._stage = stage
        datamodule = cast(DataModule, attached_datamodule(trainer))
        datamodule.set_task_weights(self.config.task_weights_by_stage[stage])
