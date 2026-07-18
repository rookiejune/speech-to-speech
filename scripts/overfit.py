from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Union, cast

import hydra
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from speech_to_speech.callback import StageConfig, StageSwitcher, WorldSizeContract
from speech_to_speech.callback.logging import (
    FlowMatchingLogger,
    GradLogger,
    GradNormLogger,
    OutputsLogger,
    SampleLogger,
    TextRetentionLogger,
)
from speech_to_speech.datamodule import ModelBatch
from speech_to_speech.generation.batch import requests_from_batch
from speech_to_speech.model import (
    AcousticType,
    SpeechToSpeechFlowModel,
    SpeechToSpeechRVQModel,
)
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.pl_module import SpeechToSpeechModule
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import init_runtime
from speech_to_speech.runtime.types import Codec
from speech_to_speech.task import Task

if __package__:
    from ._overfit_composition import flow, rvq, token
    from ._overfit_support import AcousticEvaluation, FixedDataModule, LossSummary
else:
    from _overfit_composition import flow, rvq, token
    from _overfit_support import AcousticEvaluation, FixedDataModule, LossSummary


@hydra.main(version_base=None, config_path="../configs", config_name="overfit")
def main(config: DictConfig) -> None:
    run(config)


def run(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    output_dir = Path(str(config.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(int(config.train.seed), workers=True)
    rt = init_runtime(runtime_config(config))
    codec = rt.codec
    flow_matching = rt.flow_matching
    task = Task(str(config.task))
    acoustic_type = AcousticType(str(config.acoustic.type))
    datamodule = FixedDataModule(
        str(config.codec.name),
        rt,
        {task: 1.0},
        int(config.data.sample_index),
        root=(
            None
            if config.data.root is None
            else Path(str(config.data.root)).expanduser()
        ),
        split=str(config.data.split),
    )

    repa_weight = None
    torch.manual_seed(int(config.train.seed))
    uses_acoustic_decoder = bool(codec.acoustic_codebook_sizes)
    module_config = ModuleConfig(
        learning_rate=float(config.optimizer.learning_rate),
        weight_decay=float(config.optimizer.weight_decay),
    )
    evaluation: AcousticEvaluation | None = None
    if not uses_acoustic_decoder:
        module, model = token(rt, module_config)
    elif acoustic_type is AcousticType.FLOW:
        module, model, repa_weight = flow(rt, module_config, config.acoustic)
    else:
        module, model = rvq(rt, module_config, config.acoustic)
    if uses_acoustic_decoder:
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
            every_n_steps=max(1, int(config.train.max_steps) // 5),
            seeds=range(4),
        )
    summary = LossSummary()
    if acoustic_type is AcousticType.FLOW:
        loss_pair = (
            ("flow_matching", "repa")
            if repa_weight is not None
            else ("token", "flow_matching")
        )
    else:
        loss_pair = ("token", "rvq")
    callbacks = cast(
        list[Callback],
        [
            OutputsLogger(),
            GradNormLogger(),
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
                StageConfig(
                    task_weights_by_stage=[{task: 1.0}],
                    epoch_milestones=[],
                )
            ),
            summary,
        ],
    )
    if bool(config.callbacks.sample.enabled):
        callbacks.insert(
            2,
            SampleLogger(
                [int(config.data.sample_index)],
                every_n_steps=int(config.callbacks.sample.every_n_steps),
            ),
        )
    if evaluation is not None:
        callbacks.append(evaluation)
    if uses_acoustic_decoder:
        callbacks.insert(
            1,
            GradLogger(
                loss_pair,
                "model.acoustic_flow.decoder.input.weight"
                if acoustic_type is AcousticType.FLOW
                else "model.acoustic_decoder.decoder.layers.0.self_attn.q_proj.weight",
                every_n_steps=1,
            ),
        )
    if uses_acoustic_decoder and acoustic_type is AcousticType.FLOW:
        callbacks.insert(1, FlowMatchingLogger(flow_matching, every_n_steps=1))
    trainer = build_trainer(config, output_dir, callbacks)
    trainer.fit(module, datamodule=datamodule)

    if not trainer.is_global_zero:
        return

    if uses_acoustic_decoder:
        if evaluation is None:
            raise RuntimeError("acoustic evaluation is unavailable.")
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
        "sample_index": int(config.data.sample_index),
        "max_steps": int(config.train.max_steps),
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
    config: DictConfig,
    output_dir: Path,
    callbacks: list[Callback],
) -> pl.Trainer:
    callbacks = [
        WorldSizeContract(int(config.trainer.expected_world_size)),
        *callbacks,
    ]
    return pl.Trainer(
        accelerator=str(config.trainer.accelerator),
        devices=config.trainer.devices,
        precision=cast(Any, str(config.trainer.precision)),
        max_steps=int(config.train.max_steps),
        max_epochs=int(config.trainer.max_epochs),
        default_root_dir=str(output_dir),
        logger=build_logger(config.logging, output_dir),
        callbacks=callbacks,
        log_every_n_steps=int(config.trainer.log_every_n_steps),
        enable_checkpointing=bool(config.trainer.enable_checkpointing),
        gradient_clip_val=float(config.trainer.gradient_clip_val),
        strategy=str(config.trainer.strategy),
        use_distributed_sampler=bool(config.trainer.use_distributed_sampler),
    )


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


def runtime_config(config: DictConfig) -> RuntimeConfig:
    audio_tokenizer = config.runtime.audio_tokenizer
    device = torch.device(str(config.runtime.device))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return RuntimeConfig(
        codec=str(config.codec.name),
        backbone=str(config.runtime.backbone),
        audio_tokenizer=(None if audio_tokenizer is None else str(audio_tokenizer)),
        device=str(device),
        dtype=str(config.runtime.dtype),
        attn_implementation=str(config.runtime.attn_implementation),
        flow_method=str(config.flow.method),
        flow_nfe=int(config.flow.nfe),
        flow_num_steps=int(config.flow.num_steps),
    )


def build_logger(config: DictConfig, output_dir: Path):
    name = str(config.name)
    if name == "tensorboard":
        return TensorBoardLogger(save_dir=str(output_dir), name="tensorboard")
    if name == "csv":
        return CSVLogger(save_dir=str(output_dir), name="csv")
    raise ValueError("logging.name must be tensorboard or csv.")


if __name__ == "__main__":
    main()
