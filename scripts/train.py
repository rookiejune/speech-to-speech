from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch
from anydataset.types import Modality
from anytrain.lightning import PerformanceCallback
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from omegaconf import DictConfig

from speech_to_speech.callback import StageConfig as CallbackStageConfig
from speech_to_speech.callback import StageSwitcher
from speech_to_speech.callback.logging import GradNormLogger, LossSummary, OutputsLogger
from speech_to_speech.datamodule import (
    Config as SpeechDataModuleConfig,
    DataLoaderConfig,
    DataModule,
    JointDataModule,
    LoaderSchedule,
    TextConfig,
    TextDataModule,
)
from speech_to_speech.model import AcousticType
from speech_to_speech.performance import TrainingFlops
from speech_to_speech.pl_module.composition import flow, rvq, token
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import init_runtime
from speech_to_speech.stage import StageLoaderConfig, apply_parameter_policy
from speech_to_speech.task import Task

if TYPE_CHECKING:
    from scripts._config import StagedTrainConfig, TrainDataLoaderConfig

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
    rt = init_runtime(rt_config)

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
    apply_parameter_policy(model, config.parameter_policy.spec())

    datamodule = build_datamodule(config, rt)
    summary = LossSummary()
    callbacks = training_callbacks(config, output_dir, summary)

    trainer = build_trainer(config, output_dir, callbacks)
    trainer.fit(module, datamodule=datamodule)

    if not trainer.is_global_zero:
        return

    result = {
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
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True))


def build_datamodule(config: StagedTrainConfig, runtime: object) -> JointDataModule:
    datamodules = {
        name: _loader_datamodule(config, runtime, name, loader)
        for name, loader in config.stage.loaders.items()
    }
    return JointDataModule(
        datamodules,
        LoaderSchedule(
            config.stage.loader_weights(),
            batches_per_step=config.stage.batches_per_step,
        ),
    )


def _loader_datamodule(
    config: StagedTrainConfig,
    runtime: object,
    name: str,
    loader: StageLoaderConfig,
):
    task_weights = _task_weights(loader)
    if _is_text_loader(task_weights):
        return TextDataModule(
            TextConfig(
                dataloader=_dataloader(config.text_data.dataloader),
                dataset=config.text_data.dataset,
            ),
            cast(Any, runtime),
            task_weights,
            output_dir=Path(config.output_dir).expanduser(),
            loader_name=name,
        )
    return DataModule(
        SpeechDataModuleConfig(
            codec=config.data.codec,
            dataloader=_dataloader(config.data.dataloader),
            dataset=config.data.dataset,
        ),
        cast(Any, runtime),
        task_weights,
        output_dir=Path(config.output_dir).expanduser(),
        loader_name=name,
    )


def _task_weights(loader: StageLoaderConfig) -> dict[Task, float]:
    return {Task(name): weight for name, weight in loader.task_weights.items()}


def _is_text_loader(task_weights: dict[Task, float]) -> bool:
    text_tasks = [
        task.source_modality is Modality.TEXT and task.target_modality is Modality.TEXT
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
        "lba": config.lba,
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
        ),
    )


def training_callbacks(
    config: StagedTrainConfig,
    output_dir: Path,
    summary: Callback,
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
                StageSwitcher(
                    CallbackStageConfig(
                        task_weights_by_stage=None,
                        epoch_milestones=[],
                        loader_weights_by_stage=[config.stage.loader_weights()],
                        model_stages=[config.parameter_policy],
                    )
                ),
                summary,
            ],
        )
    )
    if config.callbacks.grad_norm.enabled and performance is None:
        callbacks.append(
            GradNormLogger(every_n_steps=config.callbacks.grad_norm.every_n_steps)
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
