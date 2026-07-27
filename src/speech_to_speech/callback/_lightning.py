"""Typed access to optional Lightning trainer integrations."""

from __future__ import annotations

from typing import cast

from lightning import Trainer
from lightning.pytorch import LightningDataModule


def attached_datamodule(trainer: Trainer) -> LightningDataModule:
    value = getattr(trainer, "datamodule", None)
    if value is None:
        raise RuntimeError("callback requires Trainer.fit(..., datamodule=...).")
    return cast(LightningDataModule, value)


__all__ = ["attached_datamodule"]
