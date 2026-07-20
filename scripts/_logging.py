from __future__ import annotations

from typing import TYPE_CHECKING

from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

if TYPE_CHECKING:
    if __package__:
        from ._config import LoggingConfig
    else:
        from _config import LoggingConfig


def build(config: LoggingConfig) -> TensorBoardLogger | CSVLogger:
    if config.name == "tensorboard":
        return TensorBoardLogger(save_dir=config.save_dir, name=config.run_name)
    if config.name == "csv":
        return CSVLogger(save_dir=config.save_dir, name=config.run_name)
    raise ValueError("logging.name must be tensorboard or csv.")
