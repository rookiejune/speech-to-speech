from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import torch
from anydataset.types import Modality
from anytrain.lightning import ModelCheckpoint, PerformanceCallback, validation
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig

from speech_to_speech.callback import OnDeviceCodecMaterializer
from speech_to_speech.callback.logging import (
    GradNormLogger,
    LossSummary,
    OutputsLogger,
    TaskSampleLogger,
)
from speech_to_speech.datamodule import DataModule, SampleSplit
from speech_to_speech.datamodule.joint import LoaderSchedule
from speech_to_speech.datamodule.module import LoaderSpec
from speech_to_speech.model.acoustic import AcousticType
from speech_to_speech.performance import TrainingFlops
from speech_to_speech.pl_module.composition import flow, rvq, token
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import Runtime
from speech_to_speech.stage import StageLoaderConfig, apply_parameter_policy
from speech_to_speech.task import Task

if TYPE_CHECKING:
    from scripts._config import StagedTrainConfig

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
    validation_history = validation.History() if config.validation.enabled else None
    callbacks = training_callbacks(
        config,
        output_dir,
        summary,
        validation_history,
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
    if validation_history is not None:
        result["validation"] = validation_history.report()
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
            config.text_data,
            task_weights,
        )
    return LoaderSpec.speech(config.data, task_weights)


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
        replace(config.data, dataset=dataset),
        task_weights,
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
    validation_history: Callback | None = None,
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
    if validation_history is not None:
        callbacks.append(validation_history)
    if config.callbacks.task_sample.enabled:
        for panel in config.callbacks.task_sample.panels:
            loader_name = panel.loader
            if loader_name not in config.stage.loaders:
                raise ValueError(
                    f"task sample callback references unknown loader {loader_name!r}."
                )
            callbacks.append(
                TaskSampleLogger(
                    panel.indices,
                    config.callbacks.task_sample.every_n_steps,
                    loader_name=loader_name,
                    split=SampleSplit(panel.split),
                    task=Task(panel.task),
                    seed=config.callbacks.task_sample.seed,
                    every_audio_seconds=config.callbacks.task_sample.every_audio_seconds,
                    max_new_tokens=config.callbacks.task_sample.max_new_tokens,
                    temperature=config.callbacks.task_sample.temperature,
                    top_p=config.callbacks.task_sample.top_p,
                    do_sample=config.callbacks.task_sample.do_sample,
                    use_cache=config.callbacks.task_sample.use_cache,
                )
            )
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
                enable_version_counter=False,
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
