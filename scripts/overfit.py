from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Union, cast

import hydra
import torch
from anytrain.lightning import GradientComparison, GradientTarget
from anytrain.lightning.schedule import ScheduleRuntime
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig

from speech_to_speech.callback import (
    OnDeviceCodecMaterializer,
    build_unit_schedule,
)
from speech_to_speech.callback.logging import (
    AcousticEvaluation,
    FlowMatchingLogger,
    LossSummary,
    OutputsLogger,
    TaskSampleLogger,
)
from speech_to_speech.datamodule.module import DataModule
from speech_to_speech.datamodule.module import LoaderSpec
from speech_to_speech.datamodule.batch import (
    FusedBatch,
    ModelBatch,
    TrainInput,
)
from speech_to_speech.generation.eval.acoustic import evaluate_autoregressive
from speech_to_speech.model.acoustic import AcousticType
from speech_to_speech.model.acoustic.flow import FlowModel
from speech_to_speech.model.acoustic.rvq import RVQModel
from speech_to_speech.pl_module import SpeechToSpeechModule
from speech_to_speech.pl_module.composition import build
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import config_for_local_rank, runtime_for_sequence_layout
from speech_to_speech.runtime.codec_contract import codec_sample_rate
from speech_to_speech.training.parameter_policy import ParameterGroup
from speech_to_speech.training.composition import (
    base_callbacks,
    build_logger,
    create_trainer,
    gradient_logger,
    text_retention_logger,
)
from speech_to_speech.task import Task

if TYPE_CHECKING:
    from speech_to_speech.runtime import Runtime
    from scripts._config.overfit import OverfitConfig

if __package__:
    from ._config.overfit import (
        OverfitFlowConfig,
        overfit as parse_config,
    )
else:
    from _config.overfit import (
        OverfitFlowConfig,
        overfit as parse_config,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="overfit")
def main(config: DictConfig) -> None:
    run(parse_config(config))


def run(config: OverfitConfig) -> None:
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(config.train.seed, workers=True)
    rt_config = config_for_local_rank(config.runtime)
    rt = runtime_for_sequence_layout(rt_config, config.audio_sequence_layout)
    codec = rt.codec
    task = Task(config.task)
    datamodule = build_datamodule(config, rt, task)

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
    uses_acoustic_decoder = acoustic_type is not AcousticType.NONE
    evaluation: AcousticEvaluation | None = None
    repa_weight = (
        config.model.acoustic.repa.weight
        if isinstance(config, OverfitFlowConfig)
        else None
    )
    if config.datamodule.encode_missing_codes is True:
        module.batch_materializer = OnDeviceCodecMaterializer(rt)
    if uses_acoustic_decoder and config.callbacks.evaluation.enabled:
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
        if isinstance(batch, FusedBatch):
            raise TypeError("acoustic evaluation requires a single overfit batch.")
        train_batch = cast(TrainInput, batch)
        if config.datamodule.encode_missing_codes is True:
            train_batch = module.materialize_batch(train_batch)
            if not isinstance(train_batch, ModelBatch):
                raise TypeError(
                    "acoustic evaluation requires a materialized ModelBatch."
                )
        evaluation_batch = cast(ModelBatch, train_batch)
        acoustic_model = cast(Union[FlowModel, RVQModel], model)
        evaluation = AcousticEvaluation(
            acoustic_model,
            evaluation_batch,
            acoustic_model.acoustic_codec,
            output_dir,
            every_n_steps=max(1, config.train.max_steps // 5),
            seeds=range(4),
    )
    summary = LossSummary()
    gradient_comparison: GradientComparison | None = None
    if acoustic_type is AcousticType.FLOW:
        left_loss, right_loss = (
            ("flow_matching", "repa")
            if repa_weight is not None
            else ("token", "flow_matching")
        )
        gradient_comparison = GradientComparison(
            GradientTarget(left_loss),
            GradientTarget(right_loss),
        )
    elif acoustic_type is AcousticType.RVQ:
        gradient_comparison = GradientComparison(
            GradientTarget("token"),
            GradientTarget("rvq"),
        )
    callbacks = training_callbacks(
        config,
        rt,
        acoustic_type=acoustic_type,
        gradient_comparison=gradient_comparison,
        task=task,
        summary=summary,
        evaluation=evaluation,
        schedule_runtime=schedule_runtime,
    )
    trainer = build_trainer(config, output_dir, callbacks)
    trainer.fit(module, datamodule=datamodule)

    if not trainer.is_global_zero:
        return

    if evaluation is not None:
        _prepare_generation_module(module, _device(rt_config))
        generation = evaluate_autoregressive(
            module,
            evaluation.batch,
            sample_rate=codec_sample_rate(codec),
        )
        (output_dir / "generation.json").write_text(
            json.dumps(generation, indent=2, sort_keys=True) + "\n"
        )

    acoustic_decoder_parameters = (
        sum(parameter.numel() for parameter in model.acoustic_flow.decoder.parameters())
        if isinstance(model, FlowModel)
        else sum(parameter.numel() for parameter in model.acoustic_decoder.parameters())
        if isinstance(model, RVQModel)
        else 0
    )
    result = {
        "task": task.value,
        "parameter_policy": config.callbacks.parameter_policy.name.value,
        "sample_index": config.sample_index,
        "max_steps": config.train.max_steps,
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "acoustic_decoder": acoustic_decoder_parameters,
        },
        "metrics": summary.report(),
    }
    result_path = output_dir / "metrics.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


def build_datamodule(
    config: OverfitConfig,
    runtime: Runtime,
    task: Task,
) -> DataModule:
    return DataModule(
        runtime,
        {
            "train": LoaderSpec.speech(
                config.datamodule,
                {task: 1.0},
                sample_index=config.sample_index,
            )
        },
    )


def training_callbacks(
    config: OverfitConfig,
    runtime: Runtime,
    *,
    acoustic_type: AcousticType,
    gradient_comparison: GradientComparison | None,
    task: Task,
    summary: LossSummary,
    evaluation: AcousticEvaluation | None,
    schedule_runtime: ScheduleRuntime | None = None,
) -> list[Callback]:
    if schedule_runtime is None:
        schedule_runtime = build_unit_schedule(config.optim.schedule)
    callbacks, _ = base_callbacks(
        config.callbacks.parameter_policy,
        config.callbacks.performance,
        schedule_runtime,
        active_ctc_routes=config.pl_module.ctc.active_routes,
    )
    callbacks.append(OutputsLogger())

    flow = config.callbacks.flow_matching
    if flow.enabled and acoustic_type is AcousticType.FLOW:
        callbacks.append(
            FlowMatchingLogger(
                runtime.flow_matching,
                every_n_steps=flow.every_n_steps,
            )
        )
    gradient = _gradient_logger(config, acoustic_type, gradient_comparison)
    if gradient is not None:
        callbacks.append(gradient)
    task_sample = config.callbacks.task_sample
    if task_sample.enabled:
        callbacks.append(
            TaskSampleLogger(
                [config.sample_index],
                every_n_steps=task_sample.every_n_steps,
                loader_name="train",
                task=task,
            )
        )
    text_retention = text_retention_logger(config.callbacks.text_retention)
    if text_retention is not None:
        callbacks.append(text_retention)
    callbacks.append(summary)
    if evaluation is not None:
        callbacks.append(evaluation)
    return callbacks


def build_trainer(
    config: OverfitConfig,
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
            num_sanity_val_steps=0,
        ),
    )


def _prepare_generation_module(
    module: SpeechToSpeechModule,
    device: torch.device | None,
) -> torch.device:
    if device is None:
        return next(module.parameters()).device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    module.to(device)
    return next(module.parameters()).device


def _device(config: RuntimeConfig) -> torch.device | None:
    return None if config.device is None else torch.device(config.device)


def _gradient_logger(
    config: OverfitConfig,
    acoustic_type: AcousticType,
    gradient_comparison: GradientComparison | None,
) -> Callback | None:
    callback = config.callbacks.gradient_probe
    if (
        not callback.enabled
        or acoustic_type is AcousticType.NONE
        or config.callbacks.performance.enabled
    ):
        return None
    policy = config.callbacks.parameter_policy.spec()
    if ParameterGroup.BACKBONE not in policy.trainable_groups:
        return None
    if gradient_comparison is None:
        raise RuntimeError("acoustic composition metadata is unavailable.")
    selected_probes = (
        callback.partial_probes
        if (
            policy.backbone_top_fraction is not None
            and policy.backbone_top_fraction < 1
            and callback.partial_probes
        )
        else callback.probes
    )
    return gradient_logger(
        (gradient_comparison,),
        selected_probes,
        every_n_steps=callback.every_n_steps,
    )


if __name__ == "__main__":
    main()
