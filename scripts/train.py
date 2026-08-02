from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import torch
from anytrain.lightning import (
    GradientComparison,
    GradientProbe,
    GradientTarget,
    ModelCheckpoint,
    validation,
)
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig

from speech_to_speech.callback import (
    OOMDiagnostics,
    OnDeviceCodecMaterializer,
    build_parameter_policy,
)
from speech_to_speech.callback.logging import (
    GradLogger,
    LossSummary,
    OutputsLogger,
    TaskSampleLogger,
    TextRetentionLogger,
)
from speech_to_speech.datamodule import DataModule, SampleSplit
from speech_to_speech.datamodule.collate.joint import LoaderSchedule
from speech_to_speech.datamodule.module import LoaderSpec
from speech_to_speech.pl_module.composition import build
from speech_to_speech.runtime import runtime_for_sequence_layout
from speech_to_speech.stage import StageLoaderConfig
from speech_to_speech.task import Task

if TYPE_CHECKING:
    from scripts._train_config import (
        GradientComparisonConfig,
        GradientProbeConfig,
        StagedTrainConfig,
    )

if __package__:
    from ._train_config import train as parse_config
    from ._entry import (
        performance,
        runtime_config,
        trainer as entry_trainer,
    )
    from ._logging import build as build_logger
else:
    from _train_config import train as parse_config
    from _entry import (
        performance,
        runtime_config,
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
    rt_config = runtime_config(config.runtime)
    rt = runtime_for_sequence_layout(rt_config, config.audio_sequence_layout)

    torch.manual_seed(config.train.seed)
    acoustic_type, module, model = build(
        rt,
        config.pl_module,
        config.model,
        config.model.acoustic,
    )
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
    )

    trainer = build_trainer(config, output_dir, callbacks)
    trainer.fit(module, datamodule=datamodule, ckpt_path=config.train.ckpt_path)

    if not trainer.is_global_zero:
        return

    result: dict[str, object] = {
        "stage": config.stage_id,
        "parameter_policy": config.callbacks.parameter_policy.name.value,
        "loaders": {
            name: {
                "weight": loader.weight,
                "task_weights": dict(loader.task_weights),
            }
            for name, loader in config.stage.loaders.items()
        },
        "accumulate_grad_batches": config.stage.accumulate_grad_batches,
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
            accumulate_grad_batches=config.stage.accumulate_grad_batches,
            fuse_loaders_per_step=config.stage.fuse_loaders_per_step,
        ),
        validation=(
            _validation_spec(config) if config.validation.enabled else None
        ),
    )


def _loader_spec(
    config: StagedTrainConfig,
    loader: StageLoaderConfig,
):
    if loader.is_text:
        return LoaderSpec.text(
            config.text_datamodule,
            loader.tasks,
            prediction=loader.prediction_modality,
            tasks=config.datamodule.tasks,
        )
    return LoaderSpec.speech(
        config.datamodule,
        loader.tasks,
        prediction=loader.prediction_modality,
    )


def _validation_spec(config: StagedTrainConfig) -> LoaderSpec:
    loader = config.stage.loaders[config.validation.loader]
    if loader.is_text:
        dataset = replace(
            config.text_datamodule.dataset,
            split=config.validation.text_split,
        )
        return LoaderSpec.text(
            replace(config.text_datamodule, dataset=dataset),
            loader.tasks,
            prediction=loader.prediction_modality,
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
    )


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
        ),
    )


def _trainer_accumulate_grad_batches(config: StagedTrainConfig) -> int:
    return 1 if config.stage.fuse_loaders_per_step else config.stage.accumulate_grad_batches


def training_callbacks(
    config: StagedTrainConfig,
    output_dir: Path,
    summary: Callback,
    validation_history: Callback | None = None,
) -> list[Callback]:
    callbacks: list[Callback] = [
        build_parameter_policy(config.callbacks.parameter_policy)
    ]
    performance_callback = performance(config.callbacks.performance)
    if performance_callback is not None:
        callbacks.append(performance_callback)
    callbacks.append(OOMDiagnostics())
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
            callbacks.append(
                TaskSampleLogger(
                    panel.indices,
                    config.callbacks.task_sample.every_n_steps,
                    loader_name=loader_name,
                    split=SampleSplit(panel.split),
                    task=Task(panel.task),
                    seed=config.callbacks.task_sample.seed,
                    max_new_tokens=config.callbacks.task_sample.max_new_tokens,
                    temperature=config.callbacks.task_sample.temperature,
                    top_p=config.callbacks.task_sample.top_p,
                    do_sample=config.callbacks.task_sample.do_sample,
                    use_cache=config.callbacks.task_sample.use_cache,
                )
            )
    if config.callbacks.text_retention.enabled:
        callbacks.append(
            TextRetentionLogger(
                {
                    name: {
                        "instruction": probe.instruction,
                        "reference": probe.reference,
                    }
                    for name, probe in config.callbacks.text_retention.probes.items()
                },
                every_n_steps=config.callbacks.text_retention.every_n_steps,
                max_new_tokens=config.callbacks.text_retention.max_new_tokens,
            )
        )
    gradient = _gradient_logger(config, performance_callback)
    if gradient is not None:
        callbacks.append(gradient)
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


def _gradient_logger(
    config: StagedTrainConfig,
    performance_callback: Callback | None,
) -> GradLogger | None:
    callback = config.callbacks.gradient_probe
    if not callback.enabled or performance_callback is not None:
        return None
    return GradLogger(
        _gradient_comparisons(callback.comparisons),
        _gradient_probes(callback.probes),
        every_n_steps=callback.every_n_steps,
    )


def _gradient_probes(
    probes: dict[str, "GradientProbeConfig"],
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


def _gradient_comparisons(
    comparisons: list["GradientComparisonConfig"],
) -> tuple[GradientComparison, ...]:
    return tuple(
        GradientComparison(
            GradientTarget(comparison.left.loss, comparison.left.group),
            GradientTarget(comparison.right.loss, comparison.right.group),
        )
        for comparison in comparisons
    )


if __name__ == "__main__":
    main()
