"""Standalone Hydra entry for Kimi-style aligned MIMO pretraining."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset

from speech_to_speech.datamodule.mimo import (
    MimoDataModule,
    MimoDatasetConfig,
)
from speech_to_speech.datamodule.mimo.factory import create_dataset, task_weights
from speech_to_speech.mimo import MimoSample
from speech_to_speech.model.mimo_toy import create_toy_mimo_model
from speech_to_speech.model.mimo_factory import build_mimo_model, derive_mimo_vocab
from speech_to_speech.pl_module.mimo import MimoModule
from speech_to_speech.runtime import runtime_for_sequence_layout

if __package__:
    from ._config.mimo import (
        MimoTrainConfig,
        PreparedMimoDataConfig,
        parse as parse_config,
    )
else:
    from _config.mimo import (  # type: ignore[no-redef]
        MimoTrainConfig,
        PreparedMimoDataConfig,
        parse as parse_config,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="mimo_train")
def main(config: DictConfig) -> None:
    run(parse_config(config))


def run(config: MimoTrainConfig | DictConfig) -> dict[str, Any]:
    """Run one MIMO experiment and return the persisted metric summary."""

    parsed = parse_config(config) if isinstance(config, DictConfig) else config
    if not isinstance(parsed, MimoTrainConfig):
        raise TypeError("mimo_train.run expects MimoTrainConfig or DictConfig.")
    output_dir = Path(parsed.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(parsed.train.seed, workers=True)
    torch.manual_seed(parsed.train.seed)

    model, runtime = build_model(parsed)
    data_config = _data_for_model(parsed, runtime)
    train_dataset = _dataset(data_config, data_config.factory, data_config.kwargs)
    validation_dataset = None
    if data_config.validation_factory is not None:
        validation_dataset = _dataset(
            data_config,
            data_config.validation_factory,
            data_config.validation_kwargs,
        )
    datamodule = MimoDataModule(
        train_dataset,
        dataloader=parsed.dataloader,
        text_pad_token_id=data_config.text_pad_token_id,
        audio_pad_token_id=data_config.audio_pad_token_id,
        validation_dataset=validation_dataset,
    )
    module = MimoModule(model=model, optim=parsed.optim)
    callbacks = mimo_callbacks(parsed)
    trainer = pl.Trainer(
        accelerator=parsed.trainer.accelerator,
        devices=parsed.trainer.devices,
        strategy=parsed.trainer.strategy,
        use_distributed_sampler=parsed.trainer.use_distributed_sampler,
        precision=cast(Any, parsed.trainer.precision),
        max_epochs=parsed.trainer.max_epochs,
        max_steps=parsed.train.max_steps,
        logger=_logger(parsed),
        callbacks=callbacks,
        default_root_dir=str(output_dir),
        log_every_n_steps=parsed.trainer.log_every_n_steps,
        enable_checkpointing=parsed.trainer.enable_checkpointing,
        gradient_clip_val=parsed.trainer.gradient_clip_val,
        num_sanity_val_steps=0,
    )
    if validation_dataset is None:
        # Lightning treats a user-defined ``val_dataloader`` returning None as
        # an invalid loader.  The prepared MIMO datamodule intentionally has
        # no validation path unless configured, so pass its train loader
        # directly in that case.
        trainer.fit(
            module,
            train_dataloaders=datamodule.train_dataloader(),
            ckpt_path=parsed.train.ckpt_path,
        )
    else:
        trainer.fit(module, datamodule=datamodule, ckpt_path=parsed.train.ckpt_path)

    summary: dict[str, Any] = {
        "run_name": parsed.run_name,
        "mode": "mimo",
        "global_step": int(trainer.global_step),
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
        "runtime": None if runtime is None else runtime.codec_name,
    }
    if trainer.is_global_zero:
        (output_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    return summary


def build_model(config: MimoTrainConfig) -> tuple[nn.Module, Any | None]:
    """Build a runtime model, or a tiny local model for CPU smoke tests."""

    if bool(getattr(config.model, "toy", False)):
        kwargs = config.data.kwargs
        return (
            create_toy_mimo_model(
                text_vocab_size=int(kwargs.get("text_vocab_size", 128)),
                audio_vocab_size=int(kwargs.get("audio_vocab_size", 256)) + 3,
                hidden_size=int(
                    getattr(config.model, "audio_embedding_dim", None) or 32
                ),
            ),
            None,
        )
    runtime_config = config.runtime
    runtime = runtime_for_sequence_layout(
        runtime_config,
        config.audio_sequence_layout,
    )
    return build_mimo_model(runtime, config.model), runtime


def _runtime_data_config(
    config: MimoTrainConfig,
    runtime: Any | None,
) -> PreparedMimoDataConfig:
    if not config.data.derive_special_tokens:
        return config.data
    if runtime is None:
        if config.model.toy:
            return config.data
        raise ValueError("data.derive_special_tokens requires a runtime-backed model.")
    vocab = derive_mimo_vocab(runtime, config.model)
    return replace(
        config.data,
        special=vocab.special_tokens(
            audio_delay_tokens=config.data.audio_delay_tokens,
        ),
        text_pad_token_id=vocab.text_blank,
        audio_pad_token_id=vocab.audio_blank,
    )


def _data_for_model(
    config: MimoTrainConfig,
    runtime: Any | None,
) -> PreparedMimoDataConfig:
    if not config.model.toy:
        return _runtime_data_config(config, runtime)
    # The production experiment intentionally replaces the entry's dataset
    # factory.  A CLI ``model.toy=true`` smoke override must switch both sides
    # of that contract and discard only the production manifest path.
    toy_fields = {
        "samples",
        "text_tokens",
        "audio_tokens",
        "text_vocab_size",
        "audio_vocab_size",
        "feature_dim",
    }
    return replace(
        config.data,
        factory="speech_to_speech.datamodule.mimo.dataset:ToyMimoSegmentDataset",
        kwargs={
            key: value for key, value in config.data.kwargs.items() if key in toy_fields
        },
        derive_special_tokens=False,
    )


def _dataset(
    config: PreparedMimoDataConfig,
    factory_path: str,
    kwargs: dict[str, Any],
) -> Dataset[MimoSample]:
    return create_dataset(
        factory_path,
        kwargs,
        kind=config.kind,
        special=config.special,
        config=MimoDatasetConfig(
            samples_per_epoch=config.samples_per_epoch,
            seed=config.seed,
            max_sequence_length=config.max_sequence_length,
            task_weights=task_weights(config.task_weights),
        ),
    )


def mimo_callbacks(config: MimoTrainConfig) -> list[Callback]:
    """Return callbacks safe for MimoBatch/MimoModule.

    Single-stream callbacks intentionally do not appear here: they expect
    ``ModelBatch``, ``Task`` and ``evaluate_text`` contracts that a MIMO run
    does not implement.
    """

    checkpoint = config.callbacks.checkpoint
    if not checkpoint.enabled or not config.trainer.enable_checkpointing:
        return []
    return [
        ModelCheckpoint(
            dirpath=str(Path(config.output_dir) / "checkpoints"),
            filename=checkpoint.filename,
            save_last=checkpoint.save_last,
            save_top_k=checkpoint.save_top_k,
            every_n_train_steps=checkpoint.every_n_train_steps,
            auto_insert_metric_name=False,
            enable_version_counter=False,
        )
    ]


def _logger(config: MimoTrainConfig):
    if config.logging.name == "csv":
        return CSVLogger(
            save_dir=config.logging.save_dir,
            name=config.logging.run_name,
        )
    if config.logging.name == "tensorboard":
        return TensorBoardLogger(
            save_dir=config.logging.save_dir,
            name=config.logging.run_name,
        )
    raise ValueError("logging.name must be csv or tensorboard.")




if __name__ == "__main__":
    main()


__all__ = [
    "build_model",
    "main",
    "mimo_callbacks",
    "run",
]
