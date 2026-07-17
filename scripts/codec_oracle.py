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
from speech_to_speech.model import Config as ModelConfig
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
    objective = Objective(str(config.acoustic.objective))
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

    runtime = build_runtime(config, device)
    codec = runtime.codec
    codes = load_codes(config.data, config.codec, frame_rate=codec.frame_rate)
    if objective is Objective.FLOW:
        module, metadata = build_flow(
            config,
            codes,
            initialization,
            runtime,
            device,
        )
    else:
        raise ValueError("codec screening entry only supports acoustic flow.")

    callback = OracleLogger(
        objective=objective,
        codec=codec,
        codes=codes,
        output_dir=output_dir,
        sample_rate=int(codec.sample_rate),
        seed=int(config.train.seed),
        sample_every_n_steps=int(config.callbacks.oracle.sample_every_n_steps),
        histogram_every_n_steps=int(
            config.callbacks.oracle.histogram_every_n_steps
        ),
        save_audio=bool(config.callbacks.oracle.save_audio),
        metadata=metadata,
    )
    callbacks = training_callbacks(config, callback, output_dir)
    fit(
        config,
        module,
        codes,
        callbacks,
        output_dir,
        frame_rate=codec.frame_rate,
    )


def training_callbacks(
    config: DictConfig,
    oracle: Callback,
    output_dir: Path,
) -> list[Callback]:
    callbacks: list[Callback] = [
        oracle,
        WorldSizeContract(int(config.trainer.expected_world_size)),
    ]
    if bool(config.data.lba.enabled):
        callbacks.append(SamplerEpochSetter())
    if bool(config.callbacks.grad_norm.enabled):
        callbacks.append(
            GradNormLogger(
                every_n_steps=int(config.callbacks.grad_norm.every_n_steps),
            )
        )
    if bool(config.trainer.enable_checkpointing):
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                filename=str(config.callbacks.checkpoint.filename),
                save_last=bool(config.callbacks.checkpoint.save_last),
                save_top_k=int(config.callbacks.checkpoint.save_top_k),
            )
        )
    if bool(config.callbacks.nonfinite.enabled):
        callbacks.append(DebugCallback())
    return callbacks


def fit(
    config: DictConfig,
    module: AcousticFlowScreening,
    codes: Tensor,
    callbacks: list[Callback],
    output_dir: Path,
    *,
    frame_rate: float,
) -> None:
    with timed("logger.build", logger=str(config.logging.name)):
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
    with timed("trainer.fit", objective=Objective(str(config.acoustic.objective))):
        if bool(config.data.lba.enabled):
            trainer.fit(
                module,
                datamodule=OracleDataModule(
                    config.data,
                    config.codec,
                    frame_rate=frame_rate,
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


def load_codes(data: DictConfig, codec: DictConfig, *, frame_rate: float) -> Tensor:
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
        codes = sample_codes(
            dataset[int(data.sample_index)],
            codec=codec,
            data=data,
            frame_rate=frame_rate,
        )
    event(
        "dataset.sample",
        "ready",
        codec=name,
        code_shape=list(codes.shape),
        code_min=int(codes.min()),
        code_max=int(codes.max()),
    )
    return codes


def build_runtime(config: DictConfig, device: torch.device) -> Runtime:
    return Runtime(
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


@torch.no_grad()
def build_flow(
    config: DictConfig,
    codes: Tensor,
    initialization: Initialization,
    runtime: Runtime,
    device: torch.device,
) -> tuple[AcousticFlowScreening, dict[str, Any]]:
    if codes.size(-1) < 2:
        raise ValueError("flow screening requires semantic and acoustic codebooks.")
    model = SpeechToSpeechFlowModel(
        model_config(config.acoustic),
        runtime_snapshot=runtime,
    )
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
        enabled=bool(config.acoustic.normalize_features),
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
    metadata = common_metadata(
        config,
        codes,
        codebook,
        frame_rate=codec.frame_rate,
    ) | {
        "semantic_frames": int(semantic_codes.size(0)),
        "feature_dim": int(target.size(-1)),
        "feature_mean": float(target.mean()),
        "feature_std": float(target.std(correction=0)),
    }
    return module, metadata


def common_metadata(
    config: DictConfig,
    codes: Tensor,
    codebook: Tensor,
    *,
    frame_rate: float,
) -> dict[str, Any]:
    return {
        "codec": str(config.codec.name),
        "acoustic": str(config.acoustic.name),
        "objective": str(config.acoustic.objective),
        "initialization": str(config.init.name),
        "code_shape": list(codes.shape),
        "codebook_shape": list(codebook.shape),
        "codebook_mean": float(codebook.mean()),
        "codebook_std": float(codebook.std(correction=0)),
        "frame_rate": frame_rate,
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
    name = str(config.name)
    if name == "tensorboard":
        return TensorBoardLogger(save_dir=str(output_dir), name="tensorboard")
    if name == "csv":
        return CSVLogger(save_dir=str(output_dir), name="csv")
    raise ValueError("logging.name must be tensorboard or csv.")


def model_config(config: DictConfig) -> ModelConfig:
    dim = config.decoder.dim
    return ModelConfig(
        acoustic_decoder_dim=None if dim is None else int(dim),
        acoustic_decoder_layers=int(config.decoder.layers),
        acoustic_decoder_heads=int(config.decoder.heads),
        acoustic_decoder_ffn_ratio=int(config.decoder.ffn_ratio),
    )


def process_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return device


def path(value: Any) -> Path | None:
    return None if value is None else Path(str(value)).expanduser()


if __name__ == "__main__":
    main()
