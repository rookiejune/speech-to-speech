from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import torch
from anydataset.types import Modality
from anytrain.lightning import PerformanceCallback
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from omegaconf import DictConfig

from speech_to_speech.callback import OnDeviceCodecMaterializer
from speech_to_speech.callback.logging import (
    GradNormLogger,
    LossSummary,
    OutputsLogger,
    ValidationSummary,
)
from speech_to_speech.datamodule import DataModule
from speech_to_speech.datamodule.joint import LoaderSchedule
from speech_to_speech.datamodule.module import (
    Config as SpeechDataModuleConfig,
    DataLoaderConfig,
    LoaderSpec,
)
from speech_to_speech.datamodule.text import TextConfig
from speech_to_speech.model.acoustic import AcousticType
from speech_to_speech.performance import TrainingFlops
from speech_to_speech.pl_module.composition import flow, rvq, token
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import Runtime
from speech_to_speech.stage import StageLoaderConfig, apply_parameter_policy
from speech_to_speech.task import Task

if TYPE_CHECKING:
    from scripts._config import StagedTrainConfig, TrainDataLoaderConfig
    from speech_to_speech.datamodule.dataset import DatasetConfig

if __package__:
    from ._config import (
        StagedTrainFlowConfig,
        StagedTrainTokenConfig,
        train as parse_config,
    )
    from ._entry import (
        acoustic_composition,
        performance,
        runtime_config as entry_runtime_config,
        trainer as entry_trainer,
    )
    from ._logging import build as build_logger
else:
    from _config import (
        StagedTrainFlowConfig,
        StagedTrainTokenConfig,
        train as parse_config,
    )
    from _entry import (
        acoustic_composition,
        performance,
        runtime_config as entry_runtime_config,
        trainer as entry_trainer,
    )
    from _logging import build as build_logger


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(config: DictConfig) -> None:
    run(parse_config(config))


def run(config: StagedTrainConfig) -> None:
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(config.train.seed, workers=True)
    rt_config = runtime_config(config)
    rt = Runtime(rt_config)

    torch.manual_seed(config.train.seed)
    acoustic_type = _composition(
        config,
        uses_acoustic_side_channel=rt.acoustic_side_channel,
    )
    if isinstance(config, StagedTrainTokenConfig):
        module, model = token(rt, config.pl_module, config.model)
    elif isinstance(config, StagedTrainFlowConfig):
        module, model, _ = flow(rt, config.pl_module, config.model, config.acoustic)
    else:
        module, model = rvq(rt, config.pl_module, config.model, config.acoustic)
    if config.data.encode_missing_codes is True:
        module.batch_materializer = OnDeviceCodecMaterializer(rt)
    apply_parameter_policy(model, config.parameter_policy.spec())

    datamodule = build_datamodule(config, rt)
    summary = LossSummary()
    validation_summary = ValidationSummary() if config.validation.enabled else None
    callbacks = training_callbacks(
        config,
        output_dir,
        summary,
        validation_summary,
    )

    trainer = build_trainer(config, output_dir, callbacks)
    trainer.fit(module, datamodule=datamodule, ckpt_path=config.train.ckpt_path)

    if not trainer.is_global_zero:
        return

    result: dict[str, object] = {
        "stage": config.stage.name.value,
        "parameter_policy": config.parameter_policy.name.value,
        "loaders": {
            name: {
                "weight": loader.weight,
                "task_weights": dict(loader.task_weights),
            }
            for name, loader in config.stage.loaders.items()
        },
        "batches_per_step": config.stage.batches_per_step,
        "max_steps": config.train.max_steps,
        "composition": acoustic_type.value,
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
        "metrics": summary.report(),
    }
    if validation_summary is not None:
        result["validation"] = validation_summary.report()
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True))


def build_datamodule(config: StagedTrainConfig, runtime: Runtime) -> DataModule:
    loaders = {
        name: _loader_spec(config, loader)
        for name, loader in config.stage.loaders.items()
    }
    return DataModule(
        runtime,
        loaders,
        LoaderSchedule(
            config.stage.loader_weights(),
            batches_per_step=config.stage.batches_per_step,
        ),
        validation=(
            _validation_spec(config) if config.validation.enabled else None
        ),
    )


def _loader_spec(
    config: StagedTrainConfig,
    loader: StageLoaderConfig,
):
    task_weights = _task_weights(loader)
    if _is_text_loader(task_weights):
        return LoaderSpec.text(
            TextConfig(
                dataloader=_dataloader(config.text_data.dataloader),
                dataset=config.text_data.dataset,
            ),
            task_weights,
        )
    return LoaderSpec.speech(_speech_config(config), task_weights)


def _validation_spec(config: StagedTrainConfig) -> LoaderSpec:
    loader = config.stage.loaders[config.validation.loader]
    task_weights = _task_weights(loader)
    if _is_text_loader(task_weights):
        raise ValueError("validation loader must be a speech loader.")
    dataset = replace(
        config.data.dataset,
        split_label=config.validation.split_label,
    )
    return LoaderSpec.speech(
        _speech_config(config, dataset=dataset),
        task_weights,
    )


def _speech_config(
    config: StagedTrainConfig,
    *,
    dataset: DatasetConfig | None = None,
) -> SpeechDataModuleConfig:
    return SpeechDataModuleConfig(
        codec=config.data.codec,
        dataloader=_dataloader(config.data.dataloader),
        shape=config.data.shape,
        encode_missing_codes=config.data.encode_missing_codes,
        dataset=config.data.dataset if dataset is None else dataset,
    )


def _task_weights(loader: StageLoaderConfig) -> dict[Task, float]:
    return {Task(name): weight for name, weight in loader.task_weights.items()}


def _is_text_loader(task_weights: dict[Task, float]) -> bool:
    text_tasks = [
        task.source_modality is not Modality.AUDIO
        and task.target_modality is Modality.TEXT
        for task in task_weights
    ]
    if any(text_tasks) and not all(text_tasks):
        raise ValueError("a staged loader cannot mix pure text and speech tasks.")
    return all(text_tasks)


def _dataloader(config: TrainDataLoaderConfig) -> DataLoaderConfig:
    return {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": config.persistent_workers,
    }


def build_trainer(
    config: StagedTrainConfig,
    output_dir: Path,
    callbacks: list[Callback],
) -> pl.Trainer:
    return cast(
        pl.Trainer,
        entry_trainer(
            config,
            output_dir,
            callbacks,
            logger=build_logger(config.logging),
            factory=pl.Trainer,
            val_check_interval=(
                config.validation.every_n_steps
                if config.validation.enabled
                else None
            ),
            num_sanity_val_steps=(
                config.validation.sanity_steps
                if config.validation.enabled
                else None
            ),
        ),
    )


def training_callbacks(
    config: StagedTrainConfig,
    output_dir: Path,
    summary: Callback,
    validation_summary: Callback | None = None,
) -> list[Callback]:
    callbacks: list[Callback] = []
    performance = _performance(config)
    if performance is not None:
        callbacks.append(performance)
    callbacks.extend(
        cast(
            list[Callback],
            [
                OutputsLogger(),
                summary,
            ],
        )
    )
    if validation_summary is not None:
        callbacks.append(validation_summary)
    if config.callbacks.grad_norm.enabled and performance is None:
        callbacks.append(
            GradNormLogger(
                every_n_steps=config.callbacks.grad_norm.every_n_steps,
                every_audio_seconds=config.callbacks.grad_norm.every_audio_seconds,
            )
        )
    if config.trainer.enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                filename=config.callbacks.checkpoint.filename,
                save_last=config.callbacks.checkpoint.save_last,
                save_top_k=config.callbacks.checkpoint.save_top_k,
                every_n_train_steps=config.callbacks.checkpoint.every_n_train_steps,
                auto_insert_metric_name=False,
            )
        )
    return callbacks


def runtime_config(config: StagedTrainConfig) -> RuntimeConfig:
    return entry_runtime_config(config.runtime)


def _performance(config: StagedTrainConfig) -> Callback | None:
    return performance(
        config.callbacks.performance,
        callback=PerformanceCallback,
        flops=TrainingFlops(),
    )


def _composition(
    config: StagedTrainConfig,
    *,
    uses_acoustic_side_channel: bool,
) -> AcousticType:
    return acoustic_composition(
        config,
        token_type=StagedTrainTokenConfig,
        uses_acoustic_side_channel=uses_acoustic_side_channel,
    )


if __name__ == "__main__":
    main()
