from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Optional

from anydataset.types import Modality

from ..._compat import StrEnum, auto
from ...task import PredictionModality, Task


class LoaderStepMode(StrEnum):
    WEIGHTED_WINDOW = auto()
    TOKEN_WEIGHTED = auto()
    FUSED_JOINT = auto()
    SERIAL_JOINT = auto()


class ARFraming(StrEnum):
    INSTRUCTION = auto()
    PRETRAINING = auto()


@dataclass
class LoaderConfig:
    weight: float
    task_weights: dict[str, float]
    prediction: Optional[str] = None
    ar_framing: str = ARFraming.INSTRUCTION.value

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
        if not isinstance(self.ar_framing, str):
            raise TypeError("loader plan ar_framing must be a string.")
        try:
            framing = ARFraming(self.ar_framing)
        except ValueError as error:
            raise ValueError(
                "loader plan ar_framing must be 'instruction' or 'pretraining'."
            ) from error
        self.ar_framing = framing.value
        validate_ar_framing(framing, self.tasks)
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
    def framing(self) -> ARFraming:
        return ARFraming(self.ar_framing)

    @property
    def is_text(self) -> bool:
        from ...task import resolve_prediction

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
    step_mode: str = LoaderStepMode.WEIGHTED_WINDOW.value
    fuse_loaders_per_step: bool = False

    def __post_init__(self) -> None:
        mode = self._resolved_step_mode()
        self.step_mode = mode.value
        self.fuse_loaders_per_step = self._resolved_fuse_loaders_per_step(mode)
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
        if not isinstance(self.loaders, Mapping):
            raise TypeError("loader_plan.loaders must be a mapping.")
        if self.loaders:
            _validate_weights(self.loader_weights(), name="loader plan weights")
            for name in self.loaders:
                if not name:
                    raise ValueError("loader plan loader names must not be empty.")
            if self.mode in {
                LoaderStepMode.FUSED_JOINT,
                LoaderStepMode.SERIAL_JOINT,
            }:
                _validate_single_task_loaders(self.loaders)

    def loader_weights(self) -> dict[str, float]:
        return {name: loader.weight for name, loader in self.loaders.items()}

    @property
    def mode(self) -> LoaderStepMode:
        return LoaderStepMode(self.step_mode)

    def _resolved_step_mode(self) -> LoaderStepMode:
        try:
            return LoaderStepMode(self.step_mode)
        except ValueError as error:
            raise ValueError(
                "loader_plan.step_mode must be 'weighted_window', "
                "'token_weighted', 'fused_joint', or 'serial_joint'."
            ) from error

    def _resolved_fuse_loaders_per_step(self, mode: LoaderStepMode) -> bool:
        if not isinstance(self.fuse_loaders_per_step, bool):
            raise TypeError(
                "loader_plan.fuse_loaders_per_step must be a boolean."
            )
        if mode is LoaderStepMode.FUSED_JOINT:
            return True
        if mode is LoaderStepMode.SERIAL_JOINT:
            return False
        if mode is LoaderStepMode.TOKEN_WEIGHTED:
            if self.fuse_loaders_per_step:
                raise ValueError(
                    "loader_plan.step_mode=token_weighted requires "
                    "fuse_loaders_per_step=false."
                )
            return False
        return self.fuse_loaders_per_step


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


def _validate_single_task_loaders(loaders: Mapping[str, LoaderConfig]) -> None:
    for name, loader in loaders.items():
        active = [
            task for task, weight in loader.task_weights.items() if weight > 0
        ]
        if len(active) != 1:
            raise ValueError(
                "joint loader_plan loaders must each declare exactly one "
                f"positive task; loader {name!r} declares {len(active)}."
            )


def validate_ar_framing(
    framing: ARFraming,
    tasks: Iterable[Task],
) -> None:
    if not isinstance(framing, ARFraming):
        raise TypeError("ar_framing must be an ARFraming.")
    if framing is ARFraming.INSTRUCTION:
        return
    unsupported = sorted(
        task.value
        for task in tasks
        if task not in {Task.AUDIO_AR, Task.TEXT_AR}
    )
    if unsupported:
        raise ValueError(
            "pretraining AR framing only supports AUDIO_AR and TEXT_AR; got: "
            + ", ".join(unsupported)
        )


__all__ = [
    "ARFraming",
    "LoaderConfig",
    "LoaderPlanConfig",
    "LoaderStepMode",
    "validate_ar_framing",
]
