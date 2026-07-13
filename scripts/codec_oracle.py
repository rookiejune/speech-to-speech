from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

import hydra
import torch
from anytrain.framework.flow_matching import ContinuousFlowRuntime, ODESampler
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
    FlowOracle,
    Initialization,
    Logger as OracleLogger,
    Objective,
    SamplerEpochSetter,
    TokenOracle,
    WorldSizeContract,
    codes as sample_codes,
    event,
    feature_stats,
    single_batch_loader,
    timed,
)

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
    codec = build_codec(config.codec, device=device)
    if objective is Objective.FLOW:
        module, metadata = build_flow(config, codec, codes, initialization)
    elif objective is Objective.TOKEN:
        module, metadata = build_token(config, codec, codes, initialization)
    else:
        raise AssertionError(f"unsupported objective: {objective}")

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
                    objective=objective,
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


def build_codec(config: DictConfig, *, device: torch.device) -> Any:
    name = str(config.name)
    with timed("codec.load", codec=name):
        if name == "longcat":
            from anytrain.codec.longcat import LongCat, LongCatDecoderName

            codec = LongCat.from_pretrained(
                cache_dir=path(config.cache_dir),
                decoder=cast(LongCatDecoderName, str(config.decoder)),
                device=device,
                local_files_only=bool(config.local_files_only),
            )
        elif name == "unicodec":
            from anytrain.codec.unicodec import UniCodec

            codec = UniCodec.from_pretrained(
                cache_dir=path(config.cache_dir),
                device=device,
                domain=str(config.domain),
                bandwidth_id=int(config.bandwidth_id),
                local_files_only=bool(config.local_files_only),
            )
        else:
            raise ValueError(f"unsupported codec oracle codec: {name}")
    return codec


@torch.no_grad()
def build_flow(
    config: DictConfig,
    codec: Any,
    codes: Tensor,
    initialization: Initialization,
) -> tuple[FlowOracle, dict[str, Any]]:
    if codes.size(-1) < 2:
        raise ValueError("flow oracle requires semantic and acoustic codebooks.")
    semantic_codes = codes[:, 0]
    acoustic_codes = codes[:, 1:]
    with timed(
        "codec.dequantize_probe",
        codec=str(config.codec.name),
        code_shape=list(acoustic_codes.shape),
    ):
        target = codec.acoustic_codes_to_features(
            acoustic_codes.unsqueeze(0).to(codec.device)
        ).float()
    mean, std = feature_stats(
        target,
        enabled=bool(config.train.normalize_features),
    )
    codebook = codec.semantic_codebook.detach().float()
    flow = ContinuousFlowRuntime(
        sampler=ODESampler(
            method=str(config.train.flow.method),
            nfe=int(config.train.flow.nfe),
            num_steps=int(config.train.flow.num_steps),
            return_intermediates=False,
        )
    )
    module = FlowOracle(
        codebook.cpu(),
        target.size(-1),
        initialization=initialization,
        seed=int(config.train.seed),
        dequantize=codec.acoustic_codes_to_features,
        flow_runtime=flow,
        learning_rate=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
        target_mean=mean.cpu(),
        target_std=std.cpu(),
    )
    metadata = common_metadata(config, codes, codebook) | {
        "semantic_frames": int(semantic_codes.size(0)),
        "feature_dim": int(target.size(-1)),
        "feature_mean": float(target.mean()),
        "feature_std": float(target.std(correction=0)),
    }
    return module, metadata


@torch.no_grad()
def build_token(
    config: DictConfig,
    codec: Any,
    codes: Tensor,
    initialization: Initialization,
) -> tuple[TokenOracle, dict[str, Any]]:
    if codes.size(-1) != 1:
        raise ValueError("unified token oracle requires exactly one codebook.")
    vocab_size = int(codec.codebook_sizes[0])
    ids = torch.arange(vocab_size, device=codec.device).view(1, vocab_size, 1)
    with timed("codec.codebook_extract", codec=str(config.codec.name), rows=vocab_size):
        codebook = codec.codes_to_features(ids)[0].detach().float()
    module = TokenOracle(
        codebook.cpu(),
        round(float(config.data.max_seconds) * float(config.codec.frame_rate)),
        initialization=initialization,
        seed=int(config.train.seed),
        layers=int(config.token.layers),
        heads=int(config.token.heads),
        feedforward_dim=int(config.token.feedforward_dim),
        dropout=float(config.token.dropout),
        learning_rate=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
    )
    return module, common_metadata(config, codes, codebook)


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
