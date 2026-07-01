"""Optimizer and scheduler construction for Lightning training."""

from __future__ import annotations

from typing import cast

from anytrain.optim.llm import (
    LightningOptimizerConfig,
    OptimizationConfig,
    create_lightning_optimizers_from_config,
)
from anytrain.optim.options import MuonAdamWOptions, OptimizerOptions
from anytrain.optim.scheduler import make_named_scheduler_config
from torch import nn

from ..config import TrainConfig


def configure_optimizers(
    model: nn.Module,
    train: TrainConfig,
) -> LightningOptimizerConfig:
    validate_scheduler(train)
    validate_learning_rates(train)
    return create_lightning_optimizers_from_config(model, optimization_config(train))


def optimization_config(train: TrainConfig) -> OptimizationConfig:
    return OptimizationConfig(
        optimizer_options=optimizer_options(train),
        scheduler=make_named_scheduler_config(
            schedule=train.schedule,
            warmup_steps=scheduler_warmup_steps(train),
            total_steps=scheduler_total_steps(train),
            stable_steps=train.stable_steps,
            decay_steps=train.decay_steps,
            min_lr_ratio=train.min_lr_ratio,
        ),
    )


def optimizer_options(train: TrainConfig) -> OptimizerOptions:
    if train.optimizer != "muon":
        return OptimizationConfig.from_preset(
            train.optimizer_preset,
            optimizer=train.optimizer,
            lr=adamw_learning_rate(train),
            weight_decay=train.weight_decay,
        ).optimizer_options

    preset = OptimizationConfig.from_preset(
        train.optimizer_preset,
        optimizer=train.optimizer,
        lr=muon_learning_rate(train),
        weight_decay=train.weight_decay,
    )
    options = cast(MuonAdamWOptions, preset.optimizer_options)
    return {
        "muon": dict(options["muon"]),
        "adamw": {
            **options["adamw"],
            "lr": adamw_learning_rate(train),
        },
    }


def validate_scheduler(train: TrainConfig) -> None:
    if train.schedule != "warmup_cosine":
        raise ValueError("train.schedule must be 'warmup_cosine'.")


def validate_learning_rates(train: TrainConfig) -> None:
    if train.optimizer != "muon" and train.muon_learning_rate is not None:
        raise ValueError("train.muon_learning_rate requires train.optimizer='muon'.")


def scheduler_total_steps(train: TrainConfig) -> int:
    return train.max_steps


def scheduler_warmup_steps(train: TrainConfig) -> int:
    ratio = train.warmup_ratio
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError("train.warmup_ratio must be greater than or equal to 0 and less than 1.")
    return round(train.max_steps * ratio)


def learning_rate(train: TrainConfig) -> float:
    if train.optimizer == "muon":
        return muon_learning_rate(train)
    return adamw_learning_rate(train)


def adamw_learning_rate(train: TrainConfig) -> float:
    if train.adamw_learning_rate is None:
        return train.learning_rate
    return train.adamw_learning_rate


def muon_learning_rate(train: TrainConfig) -> float:
    if train.muon_learning_rate is None:
        return train.learning_rate
    return train.muon_learning_rate
