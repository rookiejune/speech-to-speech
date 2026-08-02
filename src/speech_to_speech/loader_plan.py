from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

from anydataset.types import Modality

from .prediction import PredictionModality
from .task import Task


@dataclass
class LoaderConfig:
    weight: float
    task_weights: dict[str, float]
    prediction: Optional[str] = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (float, int))
            or not math.isfinite(self.weight)
            or self.weight <= 0
        ):
            raise ValueError("loader plan weight must be finite and positive.")
        _validate_weights(self.task_weights, name="loader plan task weights")
        if self.prediction is not None:
            if not isinstance(self.prediction, str):
                raise TypeError(
                    "loader plan prediction must be a string or None."
                )
            PredictionModality(self.prediction)
        self.is_text

    @property
    def tasks(self) -> dict[Task, float]:
        return {Task(name): weight for name, weight in self.task_weights.items()}

    @property
    def prediction_modality(self) -> PredictionModality | None:
        if self.prediction is None:
            return None
        return PredictionModality(self.prediction)

    @property
    def is_text(self) -> bool:
        from .task_spec import resolve_prediction

        active = [task for task, weight in self.tasks.items() if weight > 0]
        text = [
            task.source_modality is not Modality.AUDIO
            and resolve_prediction(task, self.prediction_modality)
            is PredictionModality.TEXT
            for task in active
        ]
        if any(text) and not all(text):
            raise ValueError("a loader plan cannot mix pure text and speech tasks.")
        return all(text)


@dataclass
class LoaderPlanConfig:
    loaders: dict[str, LoaderConfig] = field(default_factory=dict)
    accumulate_grad_batches: int = 1
    fuse_loaders_per_step: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.accumulate_grad_batches, bool)
            or not isinstance(self.accumulate_grad_batches, int)
        ):
            raise TypeError(
                "loader_plan.accumulate_grad_batches must be an integer."
            )
        if self.accumulate_grad_batches < 1:
            raise ValueError(
                "loader_plan.accumulate_grad_batches must be positive."
            )
        if not isinstance(self.fuse_loaders_per_step, bool):
            raise TypeError(
                "loader_plan.fuse_loaders_per_step must be a boolean."
            )
        if not isinstance(self.loaders, Mapping):
            raise TypeError("loader_plan.loaders must be a mapping.")
        if self.loaders:
            _validate_weights(self.loader_weights(), name="loader plan weights")
            for name in self.loaders:
                if not name:
                    raise ValueError("loader plan loader names must not be empty.")

    def loader_weights(self) -> dict[str, float]:
        return {name: loader.weight for name, loader in self.loaders.items()}


def _validate_weights(weights: Mapping[str, float], *, name: str) -> None:
    if not weights:
        raise ValueError(f"{name} must contain at least one item.")
    if any(not key for key in weights):
        raise ValueError(f"{name} names must not be empty.")
    values = list(weights.values())
    if any(
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value <= 0
        for value in values
    ):
        raise ValueError(f"{name} must be finite and positive.")
    total = sum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"{name} must have a finite positive total.")


__all__ = [
    "LoaderConfig",
    "LoaderPlanConfig",
]
