from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Union, cast

import hydra
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig

from speech_to_speech.callback import OOMDiagnostics, OnDeviceCodecMaterializer
from speech_to_speech.callback.logging import (
    AcousticEvaluation,
    FlowMatchingLogger,
    GradLogger,
    GradNormLogger,
    LossSummary,
    OutputsLogger,
    TaskSampleLogger,
    TextRetentionLogger,
)
from speech_to_speech.datamodule import DataModule
from speech_to_speech.datamodule.config import DataLoaderConfig, SpeechConfig
from speech_to_speech.datamodule.module import LoaderSpec
from speech_to_speech.datamodule.types import ModelBatch
from speech_to_speech.generation.evaluation import evaluate_autoregressive
from speech_to_speech.model.acoustic import AcousticType, FlowModel, RVQModel
from speech_to_speech.pl_module import SpeechToSpeechModule
from speech_to_speech.pl_module.composition import build
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import Runtime
from speech_to_speech.runtime.types import codec_sample_rate
from speech_to_speech.stage import ParameterGroup, apply_parameter_policy
from speech_to_speech.task import Task

if TYPE_CHECKING:
    from scripts._config import OverfitConfig

if __package__:
    from ._config import (
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
    from _config import (
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
    rt = Runtime(rt_config, audio_route=config.audio_route)
    codec = rt.codec
    task = Task(config.task)
    datamodule = DataModule(
        rt,
        {
            "train": LoaderSpec.speech(
                SpeechConfig(
                    codec=config.runtime.codec,
                    dataloader=DataLoaderConfig(batch_size=1, num_workers=0),
                    shape=config.data.shape,
                    encode_missing_codes=config.data.encode_missing_codes,
                    dataset=config.data,
                ),
                {task: 1.0},
                sample_index=config.data.sample_index,
            )
        },
    )

    torch.manual_seed(config.train.seed)
    acoustic_type, module, model = build(
        rt,
        config.pl_module,
        config.model,
        config.acoustic,
    )
    uses_acoustic_decoder = acoustic_type is not AcousticType.NONE
    evaluation: AcousticEvaluation | None = None
    repa_weight = (
        config.acoustic.repa.weight
        if isinstance(config, OverfitFlowConfig)
        else None
    )
    apply_parameter_policy(model, config.parameter_policy.spec())
    if config.data.encode_missing_codes is True:
        module.batch_materializer = OnDeviceCodecMaterializer(rt)
    if uses_acoustic_decoder and config.callbacks.evaluation.enabled:
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
        if config.data.encode_missing_codes is True:
            batch = module.materialize_batch(batch)
            if not isinstance(batch, ModelBatch):
                raise TypeError(
                    "acoustic evaluation requires a materialized ModelBatch."
                )
        evaluation_batch = cast(ModelBatch, batch)
        acoustic_model = cast(Union[FlowModel, RVQModel], model)
        evaluation = AcousticEvaluation(
            acoustic_model,
            evaluation_batch,
            acoustic_model.acoustic_codec,
            output_dir,
            every_n_steps=max(1, config.train.max_steps // 5),
            every_audio_seconds=config.callbacks.evaluation.every_audio_seconds,
            seeds=range(4),
        )
    summary = LossSummary()
    loss_pair: tuple[str, str] | None = None
    if acoustic_type is AcousticType.FLOW:
        loss_pair = (
            ("flow_matching", "repa")
            if repa_weight is not None
            else ("token", "flow_matching")
        )
    elif acoustic_type is AcousticType.RVQ:
        loss_pair = ("token", "rvq")
    callbacks = cast(
        list[Callback],
        [
            OutputsLogger(),
            TextRetentionLogger(
                {
                    "zh_en": {
                        "instruction": "Translate into English: 昨晚的暴雨导致三趟列车晚点。",
                        "reference": "Last night's heavy rain delayed three trains.",
                    },
                },
                every_n_steps=1,
                max_new_tokens=8,
            ),
            summary,
        ],
    )
    if not config.callbacks.performance.enabled:
        callbacks.insert(1, GradNormLogger())
    if config.callbacks.task_sample.enabled:
        callbacks.insert(
            2,
            TaskSampleLogger(
                [config.data.sample_index],
                every_n_steps=config.callbacks.task_sample.every_n_steps,
                loader_name="train",
                task=task,
                every_audio_seconds=config.callbacks.task_sample.every_audio_seconds,
            ),
        )
    if evaluation is not None:
        callbacks.append(evaluation)
    gradient = _gradient_logger(config, acoustic_type, loss_pair)
    if gradient is not None:
        callbacks.insert(
            1,
            gradient,
        )
    if uses_acoustic_decoder and acoustic_type is AcousticType.FLOW:
        callbacks.insert(1, FlowMatchingLogger(rt.flow_matching, every_n_steps=1))
    performance_callback = performance(config.callbacks.performance)
    if performance_callback is not None:
        callbacks.insert(0, performance_callback)
    callbacks.insert(1 if performance_callback is not None else 0, OOMDiagnostics())
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
        "parameter_policy": config.parameter_policy.name.value,
        "stage": config.stage.name.value,
        "sample_index": config.data.sample_index,
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
    loss_pair: tuple[str, str] | None,
) -> GradLogger | None:
    if acoustic_type is AcousticType.NONE or config.callbacks.performance.enabled:
        return None
    policy = config.parameter_policy.spec()
    if ParameterGroup.BACKBONE not in policy.trainable_groups:
        return None
    if loss_pair is None:
        raise RuntimeError("acoustic composition metadata is unavailable.")
    parameter_name = (
        "model.backbone.model.norm.weight"
        if policy.backbone_top_fraction is not None
        and policy.backbone_top_fraction < 1
        else "model.backbone.model.layers.0.self_attn.q_proj.weight"
    )
    return GradLogger(
        loss_pair,
        parameter_name,
        every_n_steps=1,
    )


if __name__ == "__main__":
    main()
