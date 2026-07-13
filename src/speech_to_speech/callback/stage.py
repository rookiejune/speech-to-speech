from bisect import bisect_right
from dataclasses import dataclass
from typing import cast

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from ._lightning import attached_datamodule
from ..datamodule import DataModule, Task


@dataclass(frozen=True)
class Config:
    strategies: list[dict[Task, float]]
    milestones: list[int]


class StageSwitcher(Callback):
    def __init__(self, config: Config) -> None:
        super().__init__()

        if len(config.strategies) != len(config.milestones) + 1:
            raise ValueError("len(strategies) should be len(milestones) + 1")
        if any(milestone < 1 for milestone in config.milestones) or (
            config.milestones != sorted(set(config.milestones))
        ):
            raise ValueError("milestones must be positive and strictly increasing")

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
            bisect_right(self.config.milestones, trainer.current_epoch),
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
            bisect_right(self.config.milestones, finished_epochs),
        )

    def _set_stage(self, trainer: Trainer, stage: int) -> None:
        if stage == self._stage:
            return
        self._stage = stage
        datamodule = cast(DataModule, attached_datamodule(trainer))
        datamodule.set_strategy(self.config.strategies[stage])
