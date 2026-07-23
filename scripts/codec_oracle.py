from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional, cast

import hydra
from anytrain.lightning import DebugCallback, PerformanceCallback
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from omegaconf import DictConfig
from torch import Tensor
from zhuyin.datasets.wmt19_tts import wmt19_tts_codec

from speech_to_speech.callback.logging import GradNormLogger
from speech_to_speech.codec_oracle import (
    DataConfig as OracleDataConfig,
    DataModule as OracleDataModule,
)
from speech_to_speech.codec_oracle import (
    AcousticRVQScreening,
    AcousticFlowScreening,
    Initialization,
    Logger as OracleLogger,
    Objective,
    TrainingFlops,
    codes as sample_codes,
    event,
    single_batch_loader,
    timed,
    training_item,
)
from speech_to_speech.codec_oracle.factory import (
    build_flow,
    build_rvq,
    build_runtime,
    process_device,
)
from speech_to_speech.runtime.types import AudioTokenizer

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
    fixed_sample = training_item(codes, audio_tokenizer=runtime.audio_tokenizer)
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
        semantic_tokens=fixed_sample["semantic_tokens"],
        semantic_token_spans=fixed_sample["semantic_token_spans"],
    )
    callbacks = training_callbacks(config, callback, output_dir)
    fit(
        config,
        module,
        codes,
        callbacks,
        output_dir,
        data=data,
        audio_tokenizer=runtime.audio_tokenizer,
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
    audio_tokenizer: AudioTokenizer,
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
                    audio_tokenizer=audio_tokenizer,
                    frame_rate=frame_rate,
                    output_dir=output_dir,
                ),
            )
        else:
            trainer.fit(
                module,
                train_dataloaders=single_batch_loader(
                    codes,
                    audio_tokenizer,
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


def path(value: Optional[str]) -> Path | None:
    return None if value is None else Path(value).expanduser()


if __name__ == "__main__":
    main()
