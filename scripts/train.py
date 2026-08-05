from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import torch
from anytrain.lightning import (
    ModelCheckpoint,
    validation,
)
from anytrain.lightning.schedule import ScheduleRuntime
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig

from speech_to_speech.callback import (
    AssetMaterialization,
    OnDeviceCodecMaterializer,
    StreamingSynthesis,
    SynthesisSampleLogger,
    build_unit_schedule,
)
from speech_to_speech.callback.logging import (
    LossSummary,
    OutputsLogger,
    TaskSampleLogger,
)
from speech_to_speech.datamodule import SampleSplit
from speech_to_speech.datamodule.module import DataModule
from speech_to_speech.datamodule.loader import LoaderConfig, LoaderSchedule
from speech_to_speech.datamodule.module import LoaderSpec
from speech_to_speech.pl_module.composition import build
from speech_to_speech.runtime import config_for_local_rank, runtime_for_sequence_layout
from speech_to_speech.training.composition import (
    base_callbacks,
    build_logger,
    create_trainer,
    gradient_comparisons,
    gradient_logger,
    text_retention_logger,
)
from speech_to_speech.task import Task

if TYPE_CHECKING:
    from speech_to_speech.runtime import Runtime
    from scripts._config.train import StagedTrainConfig

if __package__:
    from ._config.train import train as parse_config
else:
    from _config.train import train as parse_config


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(config: DictConfig) -> None:
    run(parse_config(config))


def run(config: StagedTrainConfig) -> None:
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(config.train.seed, workers=True)
    rt_config = config_for_local_rank(config.runtime)
    rt = runtime_for_sequence_layout(rt_config, config.audio_sequence_layout)

    torch.manual_seed(config.train.seed)
    acoustic_type, module, model = build(
        rt,
        config.pl_module,
        config.model,
        config.model.acoustic,
    )
    schedule_runtime = build_unit_schedule(config.optim.schedule)
    module.optim = config.optim
    module.schedule_runtime = schedule_runtime
    if config.datamodule.encode_missing_codes is True:
        module.batch_materializer = OnDeviceCodecMaterializer(rt)

    datamodule = build_datamodule(config, rt)
    summary = LossSummary()
    validation_history = validation.History() if config.validation.enabled else None
    callbacks = training_callbacks(
        config,
        output_dir,
        summary,
        validation_history,
        schedule_runtime,
    )

    trainer = build_trainer(config, output_dir, callbacks)
    trainer.fit(
        module,
        datamodule=datamodule,
        ckpt_path=_resume_checkpoint(config, output_dir),
    )

    if not trainer.is_global_zero:
        return

    result: dict[str, object] = {
        "parameter_policy": config.callbacks.parameter_policy.name.value,
        "loaders": {
            name: {
                "weight": loader.weight,
                "task_weights": dict(loader.task_weights),
                "ar_framing": loader.ar_framing,
            }
            for name, loader in config.loader_plan.loaders.items()
        },
        "accumulate_grad_batches": config.loader_plan.accumulate_grad_batches,
        "step_mode": config.loader_plan.step_mode,
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
        for name, loader in config.loader_plan.loaders.items()
    }
    return DataModule(
        runtime,
        loaders,
        LoaderSchedule(
            config.loader_plan.loader_weights(),
            accumulate_grad_batches=config.loader_plan.accumulate_grad_batches,
            fuse_loaders_per_step=config.loader_plan.fuse_loaders_per_step,
            step_mode=config.loader_plan.step_mode,
        ),
        validation=(
            _validation_spec(config) if config.validation.enabled else None
        ),
    )


def _loader_spec(
    config: StagedTrainConfig,
    loader: LoaderConfig,
):
    if loader.is_text:
        return LoaderSpec.text(
            config.text_datamodule,
            loader.tasks,
            prediction=loader.prediction_modality,
            trace=loader.trace,
            ar_framing=loader.framing,
            tasks=config.datamodule.tasks,
        )
    return LoaderSpec.speech(
        config.datamodule,
        loader.tasks,
        prediction=loader.prediction_modality,
        trace=loader.trace,
        ar_framing=loader.framing,
    )


def _validation_spec(config: StagedTrainConfig) -> LoaderSpec:
    loader = config.loader_plan.loaders[config.validation.loader]
    if loader.is_text:
        dataset = replace(
            config.text_datamodule.dataset,
            split=config.validation.text_split,
        )
        return LoaderSpec.text(
            replace(config.text_datamodule, dataset=dataset),
            loader.tasks,
            prediction=loader.prediction_modality,
            trace=loader.trace,
            ar_framing=loader.framing,
            max_samples=config.validation.max_samples,
        )
    dataset = replace(
        config.datamodule.dataset,
        split_label=config.validation.split_label,
    )
    return LoaderSpec.speech(
        replace(config.datamodule, dataset=dataset),
        loader.tasks,
        prediction=loader.prediction_modality,
        trace=loader.trace,
        ar_framing=loader.framing,
    )


def build_trainer(
    config: StagedTrainConfig,
    output_dir: Path,
    callbacks: list[Callback],
) -> pl.Trainer:
    return cast(
        pl.Trainer,
        create_trainer(
            config,
            output_dir,
            callbacks,
            logger=build_logger(config.logging),
            factory=pl.Trainer,
            accumulate_grad_batches=_trainer_accumulate_grad_batches(config),
            val_check_interval=(
                config.validation.every_n_steps
                * _trainer_accumulate_grad_batches(config)
                if config.validation.enabled
                else None
            ),
            num_sanity_val_steps=(
                config.validation.sanity_steps
                if config.validation.enabled
                else None
            ),
            reload_dataloaders_every_n_epochs=(
                1 if config.datamodule.materialization.enabled else 0
            ),
        ),
    )


def _trainer_accumulate_grad_batches(config: StagedTrainConfig) -> int:
    return (
        1
        if config.loader_plan.fuse_loaders_per_step
        else config.loader_plan.accumulate_grad_batches
    )


def training_callbacks(
    config: StagedTrainConfig,
    output_dir: Path,
    summary: Callback,
    validation_history: Callback | None = None,
    schedule_runtime: ScheduleRuntime | None = None,
) -> list[Callback]:
    if schedule_runtime is None:
        schedule_runtime = build_unit_schedule(config.optim.schedule)
    callbacks, performance_callback = base_callbacks(
        config.callbacks.parameter_policy,
        config.callbacks.performance,
        schedule_runtime,
        active_ctc_routes=config.pl_module.ctc.active_routes,
        before_schedule=(
            _lifecycle_callbacks(config)
        ),
    )
    callbacks.extend(_logging_callbacks(summary, validation_history))
    callbacks.extend(_task_sample_loggers(config))
    callbacks.extend(_synthesis_sample_loggers(config))
    text_retention = text_retention_logger(config.callbacks.text_retention)
    if text_retention is not None:
        callbacks.append(text_retention)
    gradient = _gradient_logger(config, performance_callback)
    if gradient is not None:
        callbacks.append(gradient)
    checkpoint = _checkpoint_callback(config, output_dir)
    if checkpoint is not None:
        callbacks.append(checkpoint)
    return callbacks


def _lifecycle_callbacks(config: StagedTrainConfig) -> tuple[Callback, ...]:
    callbacks: list[Callback] = []
    if config.datamodule.materialization.enabled:
        callbacks.append(AssetMaterialization())
    if config.datamodule.streaming.enabled:
        callbacks.append(StreamingSynthesis())
    return tuple(callbacks)


def _resume_checkpoint(config: StagedTrainConfig, output_dir: Path) -> str | None:
    if config.train.ckpt_path is not None:
        return config.train.ckpt_path
    if not config.train.auto_resume:
        return None
    path = output_dir / "checkpoints" / "last.ckpt"
    return str(path) if path.is_file() else None


def _logging_callbacks(
    summary: Callback,
    validation_history: Callback | None,
) -> list[Callback]:
    callbacks = cast(list[Callback], [OutputsLogger(), summary])
    if validation_history is not None:
        callbacks.append(validation_history)
    return callbacks


def _task_sample_loggers(config: StagedTrainConfig) -> list[Callback]:
    task_sample = config.callbacks.task_sample
    if not task_sample.enabled:
        return []
    return [
        TaskSampleLogger(
            panel.indices,
            task_sample.every_n_steps,
            loader_name=panel.loader,
            split=SampleSplit(panel.split),
            task=Task(panel.task),
            seed=task_sample.seed,
            max_new_tokens=task_sample.max_new_tokens,
            temperature=task_sample.temperature,
            top_p=task_sample.top_p,
            do_sample=task_sample.do_sample,
            use_cache=task_sample.use_cache,
        )
        for panel in task_sample.panels
    ]


def _synthesis_sample_loggers(config: StagedTrainConfig) -> list[Callback]:
    callback = config.callbacks.synthesis_sample
    if not callback.enabled:
        return []
    return [
        SynthesisSampleLogger(
            callback.indices,
            callback.every_n_steps,
            loader_name=callback.loader,
        )
    ]


def _checkpoint_callback(
    config: StagedTrainConfig,
    output_dir: Path,
) -> Callback | None:
    if not config.trainer.enable_checkpointing:
        return None
    checkpoint = config.callbacks.checkpoint
    return ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename=checkpoint.filename,
        save_last=checkpoint.save_last,
        save_top_k=checkpoint.save_top_k,
        every_n_train_steps=checkpoint.every_n_train_steps,
        auto_insert_metric_name=False,
        enable_version_counter=False,
    )


def _gradient_logger(
    config: StagedTrainConfig,
    performance_callback: Callback | None,
) -> Callback | None:
    callback = config.callbacks.gradient_probe
    if not callback.enabled or performance_callback is not None:
        return None
    return gradient_logger(
        gradient_comparisons(callback.comparisons),
        callback.probes,
        every_n_steps=callback.every_n_steps,
    )


if __name__ == "__main__":
    main()
