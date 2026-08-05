"""Shared construction helpers for executable training entries."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from anytrain.lightning import (
    GradientComparison,
    GradientProbe,
    GradientTarget,
    PerformanceCallback,
)
from anytrain.lightning.schedule import ScheduleRuntime
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from ..callback import OOMDiagnostics, build_parameter_policy
from ..callback.logging import GradLogger, TextRetentionLogger
from .config import (
    GradientComparisonConfig,
    GradientProbeConfig,
    LoggingConfig,
    PerformanceConfig,
    TextRetentionCallbackConfig,
)
from .parameter_policy import ParameterPolicyConfig
from ..model.ctc import CTCRoute
from .performance import TrainingFlops


class TrainValues(Protocol):
    @property
    def max_steps(self) -> int: ...


class TrainerValues(Protocol):
    @property
    def accelerator(self) -> str: ...

    @property
    def devices(self) -> int | str: ...

    @property
    def strategy(self) -> str: ...

    @property
    def use_distributed_sampler(self) -> bool: ...

    @property
    def precision(self) -> str: ...

    @property
    def max_epochs(self) -> int: ...

    @property
    def log_every_n_steps(self) -> int: ...

    @property
    def enable_checkpointing(self) -> bool: ...

    @property
    def gradient_clip_val(self) -> float: ...


class EntryConfig(Protocol):
    @property
    def train(self) -> TrainValues: ...

    @property
    def trainer(self) -> TrainerValues: ...


def build_logger(config: LoggingConfig) -> TensorBoardLogger | CSVLogger:
    if config.name == "tensorboard":
        if config.version is None:
            return TensorBoardLogger(save_dir=config.save_dir, name=config.run_name)
        return TensorBoardLogger(
            save_dir=config.save_dir,
            name=config.run_name,
            version=config.version,
        )
    if config.name == "csv":
        if config.version is None:
            return CSVLogger(save_dir=config.save_dir, name=config.run_name)
        return CSVLogger(
            save_dir=config.save_dir,
            name=config.run_name,
            version=config.version,
        )
    raise ValueError("logging.name must be tensorboard or csv.")


def create_trainer(
    config: EntryConfig,
    output_dir: Path,
    callbacks: list[Callback],
    *,
    logger: Any,
    factory: Callable[..., Any],
    accumulate_grad_batches: int = 1,
    val_check_interval: int | float | None = None,
    num_sanity_val_steps: int | None = None,
    reload_dataloaders_every_n_epochs: int = 0,
) -> Any:
    options = {
        "accelerator": config.trainer.accelerator,
        "devices": config.trainer.devices,
        "precision": config.trainer.precision,
        "max_steps": config.train.max_steps,
        "max_epochs": config.trainer.max_epochs,
        "default_root_dir": str(output_dir),
        "logger": logger,
        "callbacks": callbacks,
        "log_every_n_steps": config.trainer.log_every_n_steps,
        "enable_checkpointing": config.trainer.enable_checkpointing,
        "gradient_clip_val": config.trainer.gradient_clip_val,
        "accumulate_grad_batches": accumulate_grad_batches,
        "strategy": config.trainer.strategy,
        "use_distributed_sampler": config.trainer.use_distributed_sampler,
    }
    if val_check_interval is not None:
        options["val_check_interval"] = val_check_interval
        options["check_val_every_n_epoch"] = None
    if num_sanity_val_steps is not None:
        options["num_sanity_val_steps"] = num_sanity_val_steps
    if reload_dataloaders_every_n_epochs:
        options["reload_dataloaders_every_n_epochs"] = (
            reload_dataloaders_every_n_epochs
        )
    return factory(**options)


def build_performance(config: PerformanceConfig) -> Callback | None:
    if not config.enabled:
        return None
    return PerformanceCallback(
        model_flops_per_batch=TrainingFlops(),
        hardware_peak_flops=config.hardware_peak_flops,
        log_every_n_steps=config.log_every_n_steps,
        warmup_steps=config.warmup_steps,
        measure_window_steps=config.measure_window_steps,
        sync_cuda=config.sync_cuda,
        sync_distributed=config.sync_distributed,
    )


def base_callbacks(
    parameter_policy: ParameterPolicyConfig,
    performance: PerformanceConfig,
    schedule: ScheduleRuntime,
    *,
    active_ctc_routes: frozenset[CTCRoute] = frozenset(CTCRoute),
    before_schedule: Iterable[Callback] = (),
) -> tuple[list[Callback], Callback | None]:
    performance_callback = build_performance(performance)
    callbacks: list[Callback] = [
        build_parameter_policy(
            parameter_policy,
            active_ctc_routes=active_ctc_routes,
        )
    ]
    if performance_callback is not None:
        callbacks.append(performance_callback)
    callbacks.append(OOMDiagnostics())
    callbacks.extend(before_schedule)
    callbacks.extend(schedule.callbacks())
    return callbacks, performance_callback


def text_retention_logger(
    config: TextRetentionCallbackConfig,
) -> TextRetentionLogger | None:
    if not config.enabled:
        return None
    return TextRetentionLogger(
        {
            name: {
                "instruction": probe.instruction,
                "reference": probe.reference,
            }
            for name, probe in config.probes.items()
        },
        every_n_steps=config.every_n_steps,
        max_new_tokens=config.max_new_tokens,
    )


def gradient_probes(
    probes: Mapping[str, GradientProbeConfig],
) -> tuple[GradientProbe, ...]:
    return tuple(
        GradientProbe(
            name=name,
            parameters=tuple(probe.parameters),
            match=probe.match,
            trainable_only=probe.trainable_only,
        )
        for name, probe in probes.items()
    )


def gradient_comparisons(
    comparisons: Sequence[GradientComparisonConfig],
) -> tuple[GradientComparison, ...]:
    return tuple(
        GradientComparison(
            GradientTarget(comparison.left.loss, comparison.left.group),
            GradientTarget(comparison.right.loss, comparison.right.group),
        )
        for comparison in comparisons
    )


def gradient_logger(
    comparisons: Sequence[GradientComparison],
    probes: Mapping[str, GradientProbeConfig],
    *,
    every_n_steps: int,
) -> GradLogger:
    return GradLogger(
        tuple(comparisons),
        gradient_probes(probes),
        every_n_steps=every_n_steps,
    )


__all__ = [
    "base_callbacks",
    "build_logger",
    "build_performance",
    "create_trainer",
    "gradient_comparisons",
    "gradient_logger",
    "gradient_probes",
    "text_retention_logger",
]
