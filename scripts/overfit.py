from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, cast

import hydra
import torch
from anytrain.lightning import PerformanceCallback
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig

from speech_to_speech.callback import StageConfig as CallbackStageConfig
from speech_to_speech.callback import StageSwitcher
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
from speech_to_speech.datamodule import FixedDataModule, ModelBatch
from speech_to_speech.generation.batch import requests_from_batch
from speech_to_speech.model import (
    AcousticType,
    SpeechToSpeechFlowModel,
    SpeechToSpeechRVQModel,
)
from speech_to_speech.pl_module import SpeechToSpeechModule
from speech_to_speech.pl_module.composition import flow, rvq, token
from speech_to_speech.performance import TrainingFlops
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import init_runtime
from speech_to_speech.runtime.types import Codec
from speech_to_speech.stage import ParameterGroup
from speech_to_speech.task import Task

if TYPE_CHECKING:
    from scripts._config import OverfitConfig

if __package__:
    from ._config import (
        OverfitFlowConfig,
        OverfitTokenConfig,
        overfit as parse_config,
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
        OverfitFlowConfig,
        OverfitTokenConfig,
        overfit as parse_config,
    )
    from _entry import (
        acoustic_composition,
        performance,
        runtime_config as entry_runtime_config,
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
    rt_config = runtime_config(config)
    rt = init_runtime(rt_config)
    codec = rt.codec
    task = Task(config.task)
    datamodule = FixedDataModule(
        config.runtime.codec,
        rt,
        {task: 1.0},
        config.data.sample_index,
        dataset=config.data,
    )

    repa_weight = None
    torch.manual_seed(config.train.seed)
    uses_acoustic_decoder = bool(codec.acoustic_codebook_sizes)
    acoustic_type = _composition(
        config,
        uses_acoustic_decoder=uses_acoustic_decoder,
    )
    evaluation: AcousticEvaluation | None = None
    if isinstance(config, OverfitTokenConfig):
        module, model = token(rt, config.pl_module, config.model)
    elif isinstance(config, OverfitFlowConfig):
        module, model, repa_weight = flow(
            rt,
            config.pl_module,
            config.model,
            config.acoustic,
        )
    else:
        module, model = rvq(rt, config.pl_module, config.model, config.acoustic)
    if uses_acoustic_decoder and config.callbacks.evaluation.enabled:
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
        evaluation = AcousticEvaluation(
            cast(
                Union[SpeechToSpeechFlowModel, SpeechToSpeechRVQModel],
                model,
            ),
            batch,
            codec,
            output_dir,
            every_n_steps=max(1, config.train.max_steps // 5),
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
            StageSwitcher(
                CallbackStageConfig(
                    task_weights_by_stage=[{task: 1.0}],
                    epoch_milestones=[],
                    model_stages=[config.stage],
                )
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
    performance = _performance(config)
    if performance is not None:
        callbacks.insert(0, performance)
    trainer = build_trainer(config, output_dir, callbacks)
    trainer.fit(module, datamodule=datamodule)

    if not trainer.is_global_zero:
        return

    if evaluation is not None:
        _prepare_generation_module(module, _device(rt_config))
        generation = evaluate_generation(module, evaluation.batch, codec)
        (output_dir / "generation.json").write_text(
            json.dumps(generation, indent=2, sort_keys=True) + "\n"
        )

    acoustic_decoder_parameters = (
        sum(parameter.numel() for parameter in model.acoustic_flow.decoder.parameters())
        if isinstance(model, SpeechToSpeechFlowModel)
        else sum(parameter.numel() for parameter in model.acoustic_decoder.parameters())
        if isinstance(model, SpeechToSpeechRVQModel)
        else 0
    )
    result = {
        "task": task.value,
        "parameter_stage": config.stage.name.value,
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


@torch.no_grad()
def evaluate_generation(
    module: SpeechToSpeechModule,
    batch: ModelBatch,
    codec: Codec,
) -> dict[str, Any]:
    device = next(module.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = module.generate(
        requests_from_batch(batch),
        max_new_tokens=64,
        do_sample=False,
    )[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    audio = result["audio"]
    if audio is None:
        raise RuntimeError("acoustic task generation did not return audio.")
    features = audio["features"]
    waveform = audio["waveform"]
    if features is None:
        raise RuntimeError("LongCat generation did not return acoustic features.")
    if not bool(torch.isfinite(features).all() and torch.isfinite(waveform).all()):
        raise RuntimeError("generation returned non-finite acoustic output.")
    duration = waveform.numel() / codec.sample_rate
    return {
        "token_ids": result["response_ids"].detach().cpu().tolist(),
        "feature_shape": list(features.shape),
        "waveform_shape": list(waveform.shape),
        "duration_seconds": duration,
        "elapsed_seconds": elapsed,
        "rtf": elapsed / duration,
        "finite": True,
    }


def runtime_config(config: OverfitConfig) -> RuntimeConfig:
    return entry_runtime_config(config.runtime)


def _device(config: RuntimeConfig) -> torch.device | None:
    return None if config.device is None else torch.device(config.device)


def _performance(config: OverfitConfig) -> Callback | None:
    return performance(
        config.callbacks.performance,
        callback=PerformanceCallback,
        flops=TrainingFlops(),
    )


def _gradient_logger(
    config: OverfitConfig,
    acoustic_type: AcousticType | None,
    loss_pair: tuple[str, str] | None,
) -> GradLogger | None:
    if acoustic_type is None or config.callbacks.performance.enabled:
        return None
    stage = config.stage.spec()
    if ParameterGroup.BACKBONE not in stage.trainable_groups:
        return None
    if loss_pair is None:
        raise RuntimeError("acoustic composition metadata is unavailable.")
    parameter_name = (
        "model.backbone.model.norm.weight"
        if stage.backbone_top_fraction is not None and stage.backbone_top_fraction < 1
        else "model.backbone.model.layers.0.self_attn.q_proj.weight"
    )
    return GradLogger(
        loss_pair,
        parameter_name,
        every_n_steps=1,
    )


def _composition(
    config: OverfitConfig,
    *,
    uses_acoustic_decoder: bool,
) -> AcousticType | None:
    return acoustic_composition(
        config,
        token_type=OverfitTokenConfig,
        uses_acoustic_decoder=uses_acoustic_decoder,
    )


if __name__ == "__main__":
    main()
