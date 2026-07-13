"""Typed access to optional Lightning trainer integrations."""

from __future__ import annotations

from typing import Protocol, cast

from lightning import Trainer
from lightning.pytorch import LightningDataModule
from torch import Tensor


class TextExperiment(Protocol):
    def add_text(self, tag: str, text: str, global_step: int) -> object: ...


class ScalarExperiment(Protocol):
    def add_scalar(self, tag: str, value: float, global_step: int) -> object: ...


class AudioExperiment(Protocol):
    def add_audio(
        self,
        tag: str,
        waveform: Tensor,
        global_step: int,
        *,
        sample_rate: int,
    ) -> object: ...


class HistogramExperiment(Protocol):
    def add_histogram(self, tag: str, values: Tensor, global_step: int) -> object: ...


def attached_datamodule(trainer: Trainer) -> LightningDataModule:
    value = getattr(trainer, "datamodule", None)
    if value is None:
        raise RuntimeError("callback requires Trainer.fit(..., datamodule=...).")
    return cast(LightningDataModule, value)


def text_experiment(trainer: Trainer) -> TextExperiment | None:
    value = logger_experiment(trainer)
    if value is None or not callable(getattr(value, "add_text", None)):
        return None
    return cast(TextExperiment, value)


def scalar_experiment(trainer: Trainer) -> ScalarExperiment | None:
    value = logger_experiment(trainer)
    if value is None or not callable(getattr(value, "add_scalar", None)):
        return None
    return cast(ScalarExperiment, value)


def audio_experiment(trainer: Trainer) -> AudioExperiment | None:
    value = logger_experiment(trainer)
    if value is None or not callable(getattr(value, "add_audio", None)):
        return None
    return cast(AudioExperiment, value)


def histogram_experiment(trainer: Trainer) -> HistogramExperiment | None:
    value = logger_experiment(trainer)
    if value is None or not callable(getattr(value, "add_histogram", None)):
        return None
    return cast(HistogramExperiment, value)


def logger_experiment(trainer: Trainer) -> object | None:
    logger = trainer.logger
    if logger is None:
        return None
    return getattr(logger, "experiment", None)


__all__ = [
    "attached_datamodule",
    "audio_experiment",
    "histogram_experiment",
    "scalar_experiment",
    "text_experiment",
]
