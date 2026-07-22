from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, Union, cast, runtime_checkable

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from ._lightning import attached_datamodule
from ..stage import (
    STAGE_SPECS,
    StageConfig,
    StageName,
    StageSpec,
    StagedModel,
    apply_stage,
)
from ..task import Task


StageRef = Union[StageName, StageConfig, StageSpec, str]


@dataclass(frozen=True)
class Config:
    task_weights_by_stage: list[dict[Task, float]] | None
    epoch_milestones: list[int]
    loader_weights_by_stage: list[dict[str, float]] | None = None
    model_stages: list[StageRef] | None = None
    parameter_stages: list[StageRef] | None = None


@runtime_checkable
class _TaskWeightedDataModule(Protocol):
    def set_task_weights(self, task_weights: Mapping[Task, float]) -> None: ...


@runtime_checkable
class _LoaderWeightedDataModule(Protocol):
    def set_loader_weights(self, weights: Mapping[str, float]) -> None: ...


class StageSwitcher(Callback):
    def __init__(self, config: Config) -> None:
        super().__init__()

        stages = len(config.epoch_milestones) + 1
        _validate_stage_count(
            config.task_weights_by_stage,
            stages,
            name="task_weights_by_stage",
        )
        _validate_stage_count(
            config.loader_weights_by_stage,
            stages,
            name="loader_weights_by_stage",
        )
        _validate_stage_count(
            config.model_stages,
            stages,
            name="model_stages",
        )
        _validate_stage_count(
            config.parameter_stages,
            stages,
            name="parameter_stages",
        )
        if config.model_stages is not None and config.parameter_stages is not None:
            raise ValueError("model_stages and parameter_stages cannot both be set.")
        model_stages = (
            config.model_stages
            if config.model_stages is not None
            else config.parameter_stages
        )
        if (
            config.task_weights_by_stage is None
            and config.loader_weights_by_stage is None
            and model_stages is None
        ):
            raise ValueError("stage switcher must update at least one stage target.")
        if any(milestone < 1 for milestone in config.epoch_milestones) or (
            config.epoch_milestones
            != sorted(set(config.epoch_milestones))
        ):
            raise ValueError(
                "epoch_milestones must be positive and strictly increasing"
            )

        self.config = config
        self._model_stages = model_stages
        self._stage: int | None = None

    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self._set_stage(
            trainer,
            pl_module,
            bisect_right(self.config.epoch_milestones, trainer.current_epoch),
        )

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        finished_epochs = trainer.current_epoch + 1
        self._set_stage(
            trainer,
            pl_module,
            bisect_right(self.config.epoch_milestones, finished_epochs),
        )

    def _set_stage(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: int,
    ) -> None:
        if stage == self._stage:
            return
        self._stage = stage
        datamodule = attached_datamodule(trainer)
        if self.config.task_weights_by_stage is not None:
            if not isinstance(datamodule, _TaskWeightedDataModule):
                raise TypeError("stage task weights require set_task_weights().")
            datamodule.set_task_weights(self.config.task_weights_by_stage[stage])
        if self.config.loader_weights_by_stage is not None:
            if not isinstance(datamodule, _LoaderWeightedDataModule):
                raise TypeError("stage loader weights require set_loader_weights().")
            datamodule.set_loader_weights(self.config.loader_weights_by_stage[stage])
        if self._model_stages is not None:
            apply_stage(
                _model(pl_module),
                _stage_spec(self._model_stages[stage]),
            )


def _validate_stage_count(
    values: Sequence[object] | None,
    stages: int,
    *,
    name: str,
) -> None:
    if values is not None and len(values) != stages:
        raise ValueError(f"{name} must contain one more item than epoch_milestones")


def _stage_spec(value: StageRef) -> StageSpec:
    if isinstance(value, StageSpec):
        return value
    if isinstance(value, StageConfig):
        return value.spec()
    return STAGE_SPECS[StageName(value)]


def _model(pl_module: LightningModule) -> StagedModel:
    model = getattr(pl_module, "model", None)
    if model is None:
        raise TypeError("parameter stages require pl_module.model.")
    return cast(StagedModel, model)
