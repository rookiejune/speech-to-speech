from dataclasses import dataclass
from typing import cast

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

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

        self.config = config
        self._stage = 0

    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        datamodule = cast(DataModule, trainer.datamodule)
        datamodule.set_strategy(self.config.strategies[self._stage])

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if self._stage >= len(self.config.milestones):
            return

        finished_epochs = trainer.current_epoch + 1
        milestone = self.config.milestones[self._stage]

        if finished_epochs == milestone:
            self._stage += 1

            datamodule = cast(DataModule, trainer.datamodule)
            datamodule.set_strategy(self.config.strategies[self._stage])
