from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

import hydra
import torch
from anytrain.lightning import DebugCallback
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from zhuyin.datasets.wmt19_tts import wmt19_tts_codec

from speech_to_speech.callback.logging import GradNormLogger
from speech_to_speech.codec_oracle import (
    DataModule as OracleDataModule,
)
from speech_to_speech.codec_oracle import (
    AcousticFlowScreening,
    Initialization,
    Logger as OracleLogger,
    Objective,
    SamplerEpochSetter,
    WorldSizeContract,
    codes as sample_codes,
    event,
    single_batch_loader,
    timed,
)
from speech_to_speech.model import SpeechToSpeechFlowModel
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import Runtime

TrainerPrecision = Literal[
    64,
    32,
    16,
    "64",
    "32",
    "16",
    "bf16",
    "64-true",
    "32-true",
    "16-true",
    "bf16-true",
    "16-mixed",
    "bf16-mixed",
    "transformer-engine",
    "transformer-engine-float16",
]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    run(config)


def run(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    pl.seed_everything(int(config.train.seed), workers=True)
    device = process_device()
    objective = Objective(str(config.codec.objective))
    initialization = Initialization(str(config.init.name))
    event(
        "run",
        "start",
        codec=str(config.codec.name),
        objective=objective,
        initialization=initialization,
    )
    output_dir = Path(str(config.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    codes = load_codes(config.data, config.codec)
    if objective is Objective.FLOW:
        module, metadata, codec = build_flow(config, codes, initialization, device)
    else:
        raise ValueError("codec screening entry only supports acoustic flow.")

    callback = OracleLogger(
        objective=objective,
        codec=codec,
        codes=codes,
        output_dir=output_dir,
        sample_rate=int(codec.sample_rate),
        seed=int(config.train.seed),
        sample_every_n_steps=int(config.logging.sample_every_n_steps),
        histogram_every_n_steps=int(config.logging.histogram_every_n_steps),
        save_audio=bool(config.logging.save_audio),
        metadata=metadata,
    )
    callbacks: list[Callback] = [
        callback,
        GradNormLogger(),
        WorldSizeContract(int(config.trainer.expected_world_size)),
        SamplerEpochSetter(),
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="step-{step}",
            save_last=True,
            save_top_k=0,
        ),
    ]
    if bool(config.logging.nonfinite_check):
        callbacks.append(DebugCallback())
    with timed("logger.build", logger=str(config.logging.logger)):
        logger = build_logger(config.logging, output_dir)
        logger.log_hyperparams(
            cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
        )
    trainer = pl.Trainer(
        accelerator=str(config.trainer.accelerator),
        devices=config.trainer.devices,
        precision=cast(TrainerPrecision, str(config.trainer.precision)),
        max_steps=int(config.train.max_steps),
        max_epochs=int(config.trainer.max_epochs),
        log_every_n_steps=int(config.trainer.log_every_n_steps),
        enable_checkpointing=bool(config.trainer.enable_checkpointing),
        gradient_clip_val=float(config.trainer.gradient_clip_val),
        default_root_dir=str(output_dir),
        logger=logger,
        callbacks=callbacks,
        strategy=str(config.trainer.strategy),
        use_distributed_sampler=bool(config.trainer.use_distributed_sampler),
    )
    with timed("trainer.fit", objective=objective):
        if bool(config.data.lba.enabled):
            trainer.fit(
                module,
                datamodule=OracleDataModule(
                    config.data,
                    config.codec,
                    output_dir=output_dir,
                    seed=int(config.train.seed),
                ),
            )
        else:
            trainer.fit(
                module,
                train_dataloaders=single_batch_loader(
                    codes,
                ),
            )


def load_codes(data: DictConfig, codec: DictConfig) -> Tensor:
    name = str(codec.name)
    with timed(
        "dataset.load",
        codec=name,
        split=str(data.split),
        sample_index=int(data.sample_index),
    ):
        dataset = wmt19_tts_codec(
            codec=name,
            root=path(data.root),
            split=str(data.split),
        )
        codes = sample_codes(dataset[int(data.sample_index)], codec=codec, data=data)
    event(
        "dataset.sample",
        "ready",
        codec=name,
        code_shape=list(codes.shape),
        code_min=int(codes.min()),
        code_max=int(codes.max()),
    )
    return codes


@torch.no_grad()
def build_flow(
    config: DictConfig,
    codes: Tensor,
    initialization: Initialization,
    device: torch.device,
) -> tuple[AcousticFlowScreening, dict[str, Any], Any]:
    if codes.size(-1) < 2:
        raise ValueError("flow screening requires semantic and acoustic codebooks.")
    runtime = Runtime(
        RuntimeConfig(
            codec=str(config.codec.name),
            backbone=str(config.runtime.backbone),
            audio_tokenizer=None,
            device=str(device),
            dtype=str(config.runtime.dtype),
            attn_implementation=str(config.runtime.attn_implementation),
            flow_method=str(config.flow.method),
            flow_nfe=int(config.flow.nfe),
            flow_num_steps=int(config.flow.num_steps),
        )
    )
    model = SpeechToSpeechFlowModel(runtime_snapshot=runtime)
    codec = runtime.codec
    semantic_codes = codes[:, 0]
    acoustic_codes = codes[:, 1:]
    with timed(
        "codec.dequantize_probe",
        codec=str(config.codec.name),
        code_shape=list(acoustic_codes.shape),
    ):
        target = codec.acoustic_codes_to_features(
            acoustic_codes.unsqueeze(0).to(device)
        ).float()
    mean, std = _feature_stats(
        target,
        enabled=bool(config.train.normalize_features),
    )
    codebook = codec.semantic_codebook.detach().float()
    module = AcousticFlowScreening(
        model,
        initialization=initialization,
        seed=int(config.train.seed),
        flow_runtime=runtime.flow_matching,
        learning_rate=float(config.optimizer.learning_rate),
        weight_decay=float(config.optimizer.weight_decay),
        target_mean=mean.cpu(),
        target_std=std.cpu(),
    )
    metadata = common_metadata(config, codes, codebook) | {
        "semantic_frames": int(semantic_codes.size(0)),
        "feature_dim": int(target.size(-1)),
        "feature_mean": float(target.mean()),
        "feature_std": float(target.std(correction=0)),
    }
    return module, metadata, codec


def common_metadata(
    config: DictConfig,
    codes: Tensor,
    codebook: Tensor,
) -> dict[str, Any]:
    return {
        "codec": str(config.codec.name),
        "objective": str(config.codec.objective),
        "initialization": str(config.init.name),
        "code_shape": list(codes.shape),
        "codebook_shape": list(codebook.shape),
        "codebook_mean": float(codebook.mean()),
        "codebook_std": float(codebook.std(correction=0)),
        "frame_rate": float(config.codec.frame_rate),
        "max_seconds": float(config.data.max_seconds),
    }


def _feature_stats(target: Tensor, *, enabled: bool) -> tuple[Tensor, Tensor]:
    if not enabled:
        shape = (1, 1, target.size(-1))
        return target.new_zeros(shape), target.new_ones(shape)
    mean = target.mean(dim=(0, 1), keepdim=True)
    std = target.std(dim=(0, 1), correction=0, keepdim=True).clamp_min(1e-5)
    return mean, std


def build_logger(config: DictConfig, output_dir: Path):
    name = str(config.logger)
    if name == "tensorboard":
        return TensorBoardLogger(save_dir=str(output_dir), name="tensorboard")
    if name == "csv":
        return CSVLogger(save_dir=str(output_dir), name="csv")
    raise ValueError("logging.logger must be tensorboard or csv.")


def process_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return device


def path(value: Any) -> Path | None:
    return None if value is None else Path(str(value)).expanduser()


if __name__ == "__main__":
    main()
