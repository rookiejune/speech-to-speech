from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Union, cast

import hydra
import torch
from anytrain.lightning import GradientComparison, GradientProbe, GradientTarget
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig

from speech_to_speech.callback import (
    OOMDiagnostics,
    OnDeviceCodecMaterializer,
    build_parameter_policy,
)
from speech_to_speech.callback.logging import (
    AcousticEvaluation,
    FlowMatchingLogger,
    GradLogger,
    LossSummary,
    OutputsLogger,
    TaskSampleLogger,
    TextRetentionLogger,
)
from speech_to_speech.datamodule import DataModule
from speech_to_speech.datamodule.module import LoaderSpec
from speech_to_speech.datamodule.types import FusedBatch, ModelBatch, TrainInput
from speech_to_speech.generation.eval.acoustic import evaluate_autoregressive
from speech_to_speech.model.acoustic import AcousticType, FlowModel, RVQModel
from speech_to_speech.pl_module import SpeechToSpeechModule
from speech_to_speech.pl_module.composition import build
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import runtime_for_sequence_layout
from speech_to_speech.runtime.types import codec_sample_rate
from speech_to_speech.stage import ParameterGroup
from speech_to_speech.task import Task

if TYPE_CHECKING:
    from scripts._overfit_config import GradientProbeConfig, OverfitConfig

if __package__:
    from ._overfit_config import (
        OverfitFlowConfig,
        overfit as parse_config,
    )
    from ._entry import (
        performance,
        runtime_config,
        trainer as entry_trainer,
    )
    from ._logging import build as build_logger
else:
    from _overfit_config import (
        OverfitFlowConfig,
        overfit as parse_config,
    )
    from _entry import (
        performance,
        runtime_config,
        trainer as entry_trainer,
    )
    from _logging import build as build_logger


@hydra.main(version_base=None, config_path="../configs", config_name="overfit")
def main(config: DictConfig) -> None:
    run(parse_config(config))


def run(config: OverfitConfig) -> None:
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(config.train.seed, workers=True)
    rt_config = runtime_config(config.runtime)
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
        "stage": config.stage_id,
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
) -> list[Callback]:
    performance_callback = performance(config.callbacks.performance)
    callbacks: list[Callback] = [
        build_parameter_policy(config.callbacks.parameter_policy)
    ]
    if performance_callback is not None:
        callbacks.append(performance_callback)
    callbacks.extend([OOMDiagnostics(), OutputsLogger()])

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
    text_retention = config.callbacks.text_retention
    if text_retention.enabled:
        callbacks.append(
            TextRetentionLogger(
                {
                    name: {
                        "instruction": probe.instruction,
                        "reference": probe.reference,
                    }
                    for name, probe in text_retention.probes.items()
                },
                every_n_steps=text_retention.every_n_steps,
                max_new_tokens=text_retention.max_new_tokens,
            )
        )
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
        entry_trainer(
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
) -> GradLogger | None:
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
    return GradLogger(
        (gradient_comparison,),
        _gradient_probes(selected_probes),
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


if __name__ == "__main__":
    main()
