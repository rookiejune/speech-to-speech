from __future__ import annotations

import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, cast

import hydra
import torch
from anytrain.lightning import DebugCallback, PerformanceCallback
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from omegaconf import DictConfig
from torch import Tensor
from transformers import AutoConfig
from zhuyin.datasets.wmt19_tts import wmt19_tts_codec

from speech_to_speech.callback.logging import GradNormLogger
from speech_to_speech.codec_oracle import (
    DataConfig as OracleDataConfig,
    DataModule as OracleDataModule,
)
from speech_to_speech.codec_oracle import (
    AcousticFlowScreening,
    AcousticFlowModel,
    AcousticRVQModel,
    AcousticRVQScreening,
    Initialization,
    Logger as OracleLogger,
    Objective,
    TrainingFlops,
    codes as sample_codes,
    event,
    single_batch_loader,
    timed,
)
from speech_to_speech.runtime import Runtime

if __package__:
    from ._config import (
        CodecOracleConfig,
        codec_oracle as parse_config,
    )
    from ._logging import build as build_logger
else:
    from _config import (
        CodecOracleConfig,
        codec_oracle as parse_config,
    )
    from _logging import build as build_logger

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


@hydra.main(version_base=None, config_path="../configs", config_name="codec_oracle")
def main(config: DictConfig) -> None:
    run(parse_config(config))


def run(config: CodecOracleConfig) -> None:
    pl.seed_everything(config.train.seed, workers=True)
    device = process_device(config.runtime.device)
    objective = config.codec_oracle.objective
    initialization = config.codec_oracle.initialization
    event(
        "run",
        "start",
        codec=config.runtime.codec,
        objective=objective,
        initialization=initialization,
    )
    output_dir = Path(config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = build_runtime(config, device)
    codec = runtime.codec
    data = config.codec_oracle.data
    codes = load_codes(data, config.runtime.codec, frame_rate=codec.frame_rate)
    if objective is Objective.FLOW:
        module, metadata = build_flow(
            config,
            codes,
            initialization,
            runtime,
            device,
        )
    elif objective is Objective.RVQ:
        module, metadata = build_rvq(
            config,
            codes,
            initialization,
            runtime,
            device,
        )
    else:
        raise AssertionError(f"unsupported objective: {objective}")

    callback = OracleLogger(
        objective=objective,
        codec=codec,
        codes=codes,
        output_dir=output_dir,
        sample_rate=int(codec.sample_rate),
        seed=config.train.seed,
        sample_every_n_steps=config.callbacks.oracle.sample_every_n_steps,
        histogram_every_n_steps=config.callbacks.oracle.histogram_every_n_steps,
        save_audio=config.callbacks.oracle.save_audio,
        metadata=metadata,
    )
    callbacks = training_callbacks(config, callback, output_dir)
    fit(
        config,
        module,
        codes,
        callbacks,
        output_dir,
        data=data,
        frame_rate=codec.frame_rate,
    )


def training_callbacks(
    config: CodecOracleConfig,
    oracle: Callback,
    output_dir: Path,
) -> list[Callback]:
    callbacks: list[Callback] = []
    performance = config.callbacks.performance
    if performance.enabled:
        callbacks.append(
            PerformanceCallback(
                model_flops_per_batch=TrainingFlops(),
                hardware_peak_flops=performance.hardware_peak_flops,
                log_every_n_steps=performance.log_every_n_steps,
                warmup_steps=performance.warmup_steps,
                measure_window_steps=performance.measure_window_steps,
                sync_cuda=performance.sync_cuda,
                sync_distributed=performance.sync_distributed,
            )
        )
    callbacks.append(oracle)
    if config.callbacks.grad_norm.enabled and not performance.enabled:
        callbacks.append(
            GradNormLogger(
                every_n_steps=config.callbacks.grad_norm.every_n_steps,
            )
        )
    if config.trainer.enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                filename=config.callbacks.checkpoint.filename,
                save_last=config.callbacks.checkpoint.save_last,
                save_top_k=config.callbacks.checkpoint.save_top_k,
                every_n_train_steps=(config.callbacks.checkpoint.every_n_train_steps),
                auto_insert_metric_name=False,
            )
        )
    if config.callbacks.nonfinite.enabled:
        callbacks.append(DebugCallback())
    return callbacks


def fit(
    config: CodecOracleConfig,
    module: AcousticFlowScreening | AcousticRVQScreening,
    codes: Tensor,
    callbacks: list[Callback],
    output_dir: Path,
    *,
    data: OracleDataConfig,
    frame_rate: float,
) -> None:
    with timed("logger.build", logger=config.logging.name):
        logger = build_logger(config.logging)
        logger.log_hyperparams(asdict(config))
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=cast(TrainerPrecision, config.trainer.precision),
        max_steps=config.train.max_steps,
        max_epochs=config.trainer.max_epochs,
        log_every_n_steps=config.trainer.log_every_n_steps,
        enable_checkpointing=config.trainer.enable_checkpointing,
        gradient_clip_val=config.trainer.gradient_clip_val,
        default_root_dir=str(output_dir),
        logger=logger,
        callbacks=callbacks,
        strategy=config.trainer.strategy,
        use_distributed_sampler=config.trainer.use_distributed_sampler,
    )
    with timed("trainer.fit", objective=config.codec_oracle.objective):
        if data.lba.enabled:
            trainer.fit(
                module,
                datamodule=OracleDataModule(
                    data,
                    config.runtime.codec,
                    frame_rate=frame_rate,
                    output_dir=output_dir,
                ),
            )
        else:
            trainer.fit(
                module,
                train_dataloaders=single_batch_loader(
                    codes,
                ),
            )


def load_codes(data: OracleDataConfig, codec: str, *, frame_rate: float) -> Tensor:
    with timed(
        "dataset.load",
        codec=codec,
        split=data.split,
        sample_index=data.sample_index,
    ):
        dataset = wmt19_tts_codec(
            codec=codec,
            root=path(data.root),
            split=data.split,
        )
        codes = sample_codes(
            dataset[data.sample_index],
            codec=codec,
            data=data,
            frame_rate=frame_rate,
        )
    event(
        "dataset.sample",
        "ready",
        codec=codec,
        code_shape=list(codes.shape),
        code_min=int(codes.min()),
        code_max=int(codes.max()),
    )
    return codes


def build_runtime(config: CodecOracleConfig, device: torch.device) -> Runtime:
    return Runtime(replace(config.runtime, device=str(device)))


@torch.no_grad()
def build_flow(
    config: CodecOracleConfig,
    codes: Tensor,
    initialization: Initialization,
    runtime: Runtime,
    device: torch.device,
) -> tuple[AcousticFlowScreening, dict[str, Any]]:
    if codes.size(-1) < 2:
        raise ValueError("flow screening requires semantic and acoustic codebooks.")
    model = AcousticFlowModel(
        adapter=config.model.semantic_audio_adapter,
        runtime=runtime,
        condition_dim=condition_dim(config),
        flow_runtime=runtime.flow_matching,
        decoder=config.codec_oracle.decoder,
        device=device,
        dtype=model_dtype(config.runtime.dtype),
    )
    codec = runtime.codec
    semantic_codes = codes[:, 0]
    acoustic_codes = codes[:, 1:]
    with timed(
        "codec.dequantize_probe",
        codec=config.runtime.codec,
        code_shape=list(acoustic_codes.shape),
    ):
        target = codec.acoustic_codes_to_features(
            acoustic_codes.unsqueeze(0).to(device)
        ).float()
    mean, std = _feature_stats(
        target,
        enabled=config.codec_oracle.normalize_features,
    )
    codebook = codec.semantic_codebook.detach().float()
    module = AcousticFlowScreening(
        model,
        initialization=initialization,
        seed=config.train.seed,
        flow_runtime=runtime.flow_matching,
        learning_rate=config.codec_oracle.learning_rate,
        weight_decay=config.codec_oracle.weight_decay,
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


@torch.no_grad()
def build_rvq(
    config: CodecOracleConfig,
    codes: Tensor,
    initialization: Initialization,
    runtime: Runtime,
    device: torch.device,
) -> tuple[AcousticRVQScreening, dict[str, Any]]:
    codec = runtime.codec
    acoustic_sizes = codec.acoustic_codebook_sizes
    expected_codebooks = 1 + len(acoustic_sizes)
    if not acoustic_sizes:
        raise ValueError("RVQ screening requires acoustic codebooks.")
    if codes.size(-1) != expected_codebooks:
        raise ValueError(
            "RVQ screening prepared codes must match the runtime codec: "
            f"{codes.size(-1)} != {expected_codebooks}."
        )
    model = AcousticRVQModel(
        adapter=config.model.semantic_audio_adapter,
        runtime=runtime,
        condition_dim=condition_dim(config),
        decoder=config.codec_oracle.decoder,
        device=device,
        dtype=model_dtype(config.runtime.dtype),
    )
    codebook = codec.semantic_codebook.detach().float()
    module = AcousticRVQScreening(
        model,
        initialization=initialization,
        seed=config.train.seed,
        learning_rate=config.codec_oracle.learning_rate,
        weight_decay=config.codec_oracle.weight_decay,
    )
    metadata = common_metadata(
        config,
        codes,
        codebook,
        frame_rate=codec.frame_rate,
    ) | {
        "semantic_frames": int(codes.size(0)),
        "acoustic_codebooks": len(acoustic_sizes),
        "acoustic_codebook_sizes": list(acoustic_sizes),
    }
    return module, metadata


def common_metadata(
    config: CodecOracleConfig,
    codes: Tensor,
    codebook: Tensor,
    *,
    frame_rate: float,
) -> dict[str, Any]:
    return {
        "codec": config.runtime.codec,
        "objective": config.codec_oracle.objective.value,
        "initialization": config.codec_oracle.initialization.value,
        "code_shape": list(codes.shape),
        "codebook_shape": list(codebook.shape),
        "codebook_mean": float(codebook.mean()),
        "codebook_std": float(codebook.std(correction=0)),
        "frame_rate": frame_rate,
        "max_seconds": config.codec_oracle.data.max_seconds,
    }


def _feature_stats(target: Tensor, *, enabled: bool) -> tuple[Tensor, Tensor]:
    if not enabled:
        shape = (1, 1, target.size(-1))
        return target.new_zeros(shape), target.new_ones(shape)
    mean = target.mean(dim=(0, 1), keepdim=True)
    std = target.std(dim=(0, 1), correction=0, keepdim=True).clamp_min(1e-5)
    return mean, std


class _BackboneConfig(Protocol):
    hidden_size: int


def condition_dim(config: CodecOracleConfig) -> int:
    if config.model.toy is not None:
        return config.model.toy.hidden_size
    loaded = AutoConfig.from_pretrained(config.runtime.backbone)
    backbone = cast(_BackboneConfig, cast(object, loaded))
    hidden_size = backbone.hidden_size
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
        raise TypeError("backbone hidden_size must be an integer.")
    if hidden_size <= 0:
        raise ValueError("backbone hidden_size must be positive.")
    return hidden_size


def model_dtype(value: Optional[str]) -> torch.dtype:
    if value is None:
        return torch.get_default_dtype()
    try:
        result = getattr(torch, value)
    except AttributeError as error:
        raise ValueError(f"unknown torch dtype: {value}") from error
    if not isinstance(result, torch.dtype) or not result.is_floating_point:
        raise ValueError(f"oracle model dtype must be floating point: {value}")
    return result


def process_device(configured: Optional[str]) -> torch.device:
    requested = torch.device("cuda" if configured is None else configured)
    if requested.type != "cuda":
        raise ValueError("codec oracle requires runtime.device to be cuda.")
    device = (
        requested
        if requested.index is not None
        else torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    )
    torch.cuda.set_device(device)
    return device


def path(value: Optional[str]) -> Path | None:
    return None if value is None else Path(value).expanduser()


if __name__ == "__main__":
    main()
