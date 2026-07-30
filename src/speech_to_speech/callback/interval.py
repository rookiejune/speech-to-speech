from __future__ import annotations

import math
from typing import Any, cast

import torch
from lightning import LightningModule, Trainer
from torch import Tensor

from ..datamodule.types import ModelBatch


class TrainInterval:
    """Decide whether a train-batch callback should run.

    Step intervals keep the historical optimizer-step behavior. Audio intervals
    count globally processed audio seconds from ModelBatch.audio_seconds, so
    gradient accumulation and DDP both advance by the actual batch work.
    """

    def __init__(
        self,
        *,
        every_n_steps: int | None,
        every_audio_seconds: float | None = None,
    ) -> None:
        if every_n_steps is not None:
            _positive_int(every_n_steps, "every_n_steps")
        if every_audio_seconds is not None:
            _positive(every_audio_seconds, "every_audio_seconds")
        if every_n_steps is None and every_audio_seconds is None:
            raise ValueError("train interval requires a step or audio-seconds interval.")

        self.every_n_steps = every_n_steps
        self.every_audio_seconds = every_audio_seconds
        self.audio_seconds = 0.0
        self._next_audio_seconds = every_audio_seconds
        self._last_step: int | None = None

    @property
    def uses_audio_seconds(self) -> bool:
        return self.every_audio_seconds is not None

    def should_run(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
    ) -> bool:
        if self.every_audio_seconds is None:
            assert self.every_n_steps is not None
            step = int(trainer.global_step)
            if self._last_step == step:
                return False
            if self._last_step is not None and step < self._last_step:
                raise RuntimeError("trainer global_step moved backwards.")
            self._last_step = step
            return step > 0 and step % self.every_n_steps == 0
        return self._advance_audio_seconds(trainer, pl_module, batch)

    def state_dict(self) -> dict[str, float]:
        return {
            "audio_seconds": self.audio_seconds,
            "next_audio_seconds": (
                self._next_audio_seconds
                if self._next_audio_seconds is not None
                else math.nan
            ),
            "last_step": -1.0 if self._last_step is None else float(self._last_step),
        }

    def load_state_dict(self, state: dict[str, float]) -> None:
        audio_seconds = float(state.get("audio_seconds", 0.0))
        if not math.isfinite(audio_seconds) or audio_seconds < 0:
            raise ValueError("interval audio_seconds state must be finite and non-negative.")
        self.audio_seconds = audio_seconds
        last_step = float(state.get("last_step", -1.0))
        if (
            not math.isfinite(last_step)
            or last_step < -1
            or not last_step.is_integer()
        ):
            raise ValueError("interval last_step state must be -1 or a non-negative integer.")
        self._last_step = None if last_step == -1 else int(last_step)
        if self.every_audio_seconds is None:
            self._next_audio_seconds = None
            return
        next_audio_seconds = float(
            state.get("next_audio_seconds", self.every_audio_seconds)
        )
        if not math.isfinite(next_audio_seconds) or next_audio_seconds <= 0:
            next_audio_seconds = self.every_audio_seconds
        self._next_audio_seconds = next_audio_seconds

    def _advance_audio_seconds(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
    ) -> bool:
        assert self.every_audio_seconds is not None
        assert self._next_audio_seconds is not None
        seconds = _global_audio_seconds(trainer, pl_module, batch)
        if seconds <= 0:
            return False
        self.audio_seconds += seconds
        if self.audio_seconds + 1e-9 < self._next_audio_seconds:
            return False
        while self.audio_seconds + 1e-9 >= self._next_audio_seconds:
            self._next_audio_seconds += self.every_audio_seconds
        return True


def processed_audio_seconds(batch: ModelBatch) -> float:
    if not isinstance(batch, ModelBatch):
        raise TypeError("processed audio seconds require a ModelBatch.")
    return _batch_seconds(batch)


def _batch_seconds(batch: ModelBatch) -> float:
    return float(cast(Tensor, batch.audio_seconds).detach().sum().cpu())


def _global_audio_seconds(
    trainer: Trainer,
    pl_module: LightningModule,
    batch: Any,
) -> float:
    seconds = processed_audio_seconds(cast(ModelBatch, batch))
    world_size = int(getattr(trainer, "world_size", 1))
    if world_size <= 1:
        return seconds
    strategy = getattr(trainer, "strategy", None)
    reduce = getattr(strategy, "reduce", None)
    if not callable(reduce):
        raise RuntimeError("distributed audio interval requires strategy.reduce().")
    value = torch.tensor(
        seconds,
        dtype=torch.float64,
        device=_device(pl_module),
    )
    return float(cast(Tensor, reduce(value, reduce_op="sum")).detach().cpu())


def _device(pl_module: LightningModule) -> torch.device:
    try:
        device = pl_module.device
    except (AttributeError, RuntimeError):
        return torch.device("cpu")
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


__all__ = ["TrainInterval", "processed_audio_seconds"]
