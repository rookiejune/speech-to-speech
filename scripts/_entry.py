from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

import torch
from lightning.pytorch.callbacks import Callback

from speech_to_speech.model import AcousticType
from speech_to_speech.runtime import Config as RuntimeConfig

if __package__:
    from ._config import PerformanceConfig
else:
    from _config import PerformanceConfig


class TrainConfig(Protocol):
    @property
    def max_steps(self) -> int: ...


class TrainerConfig(Protocol):
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
    def train(self) -> TrainConfig: ...

    @property
    def trainer(self) -> TrainerConfig: ...


class AcousticConfig(Protocol):
    type: str


class AcousticEntryConfig(Protocol):
    acoustic: AcousticConfig


TokenConfigT = TypeVar("TokenConfigT")


def runtime_config(config: RuntimeConfig) -> RuntimeConfig:
    device = None if config.device is None else torch.device(config.device)
    if device is not None and device.type == "cuda" and device.index is None:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return replace(
        config,
        device=None if device is None else str(device),
    )


def trainer(
    config: EntryConfig,
    output_dir: Path,
    callbacks: list[Callback],
    *,
    logger: Any,
    factory: Callable[..., Any],
) -> Any:
    return factory(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        max_steps=config.train.max_steps,
        max_epochs=config.trainer.max_epochs,
        default_root_dir=str(output_dir),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config.trainer.log_every_n_steps,
        enable_checkpointing=config.trainer.enable_checkpointing,
        gradient_clip_val=config.trainer.gradient_clip_val,
        strategy=config.trainer.strategy,
        use_distributed_sampler=config.trainer.use_distributed_sampler,
    )


def performance(
    config: PerformanceConfig,
    *,
    callback: Callable[..., Callback],
    flops: Any,
) -> Callback | None:
    if not config.enabled:
        return None
    return callback(
        model_flops_per_batch=flops,
        hardware_peak_flops=config.hardware_peak_flops,
        log_every_n_steps=config.log_every_n_steps,
        warmup_steps=config.warmup_steps,
        measure_window_steps=config.measure_window_steps,
        sync_cuda=config.sync_cuda,
        sync_distributed=config.sync_distributed,
    )


def acoustic_composition(
    config: object,
    *,
    token_type: type[TokenConfigT],
    uses_acoustic_decoder: bool,
) -> AcousticType | None:
    if isinstance(config, token_type):
        if uses_acoustic_decoder:
            raise ValueError(
                "codec exposes independent acoustic codebooks; configure "
                "model/acoustic=flow or model/acoustic=rvq."
            )
        return None
    if not uses_acoustic_decoder:
        raise ValueError(
            "codec has no independent acoustic codebooks; remove the acoustic "
            "config group with ~model/acoustic."
        )
    return AcousticType(cast(AcousticEntryConfig, config).acoustic.type)
