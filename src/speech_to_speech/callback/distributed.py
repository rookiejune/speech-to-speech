from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class WorldSizeContract(Callback):
    def __init__(self, expected: int) -> None:
        super().__init__()
        self.expected = expected

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if trainer.world_size != self.expected:
            raise RuntimeError(
                f"expected DDP world size {self.expected}, got {trainer.world_size}."
            )
