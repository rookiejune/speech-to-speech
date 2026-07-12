from __future__ import annotations

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DistributedSampler

from .logging.trace import event


class DistributedContract(Callback):
    """Validate the launched world size and advance an explicit sampler."""

    def __init__(self, expected_world_size: int) -> None:
        super().__init__()
        self.expected_world_size = expected_world_size

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if trainer.world_size != self.expected_world_size:
            raise RuntimeError(
                f"expected DDP world size {self.expected_world_size}, "
                f"got {trainer.world_size}."
            )
        if trainer.is_global_zero:
            event(
                "distributed.contract",
                "ready",
                strategy=type(trainer.strategy).__name__,
                world_size=trainer.world_size,
            )

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        sampler = getattr(trainer.datamodule, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(trainer.current_epoch)


__all__ = ["DistributedContract"]
