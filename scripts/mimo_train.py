"""Standalone Hydra entry for Kimi-style aligned MIMO pretraining."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import hydra
import torch
from anytrain.lightning import ModelCheckpoint, ObservationCallback
from anytrain.lightning.schedule import ScheduleRuntime
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import Dataset

from speech_to_speech.callback import OOMDiagnostics, build_unit_schedule
from speech_to_speech.callback.logging import LossSummary
from speech_to_speech.datamodule.mimo import (
    MimoDataModule,
    MimoDatasetConfig,
)
from speech_to_speech.datamodule.mimo.factory import create_dataset, task_weights
from speech_to_speech.mimo import MIMO_IGNORE_INDEX, MimoSample
from speech_to_speech.model.toy import create_toy_mimo_model
from speech_to_speech.model.mimo_factory import (
    MimoVocab,
    build_mimo_model,
    derive_mimo_vocab,
)
from speech_to_speech.pl_module.mimo import MimoModule
from speech_to_speech.runtime import runtime_for_sequence_layout
from speech_to_speech.training.composition import build_logger, create_trainer

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

    model, runtime, vocab = build_model(parsed)
    data_config = _data_for_model(parsed, vocab)
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
    schedule_runtime = build_unit_schedule(parsed.optim.schedule)
    module = MimoModule(
        model=model,
        optim=parsed.optim,
        schedule_runtime=schedule_runtime,
        checkpoint_metadata=_checkpoint_metadata(parsed, data_config, vocab),
    )
    summary = LossSummary()
    callbacks = mimo_callbacks(parsed, summary, schedule_runtime)
    trainer = create_trainer(
        parsed,
        output_dir,
        callbacks,
        logger=build_logger(parsed.logging),
        factory=pl.Trainer,
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

    result: dict[str, Any] = {
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
        "metrics": summary.report(),
    }
    if trainer.is_global_zero:
        (output_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    return result


def build_model(
    config: MimoTrainConfig,
) -> tuple[nn.Module, Any | None, MimoVocab | None]:
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
            None,
        )
    runtime_config = config.runtime
    runtime = runtime_for_sequence_layout(
        runtime_config,
        config.audio_sequence_layout,
    )
    vocab = derive_mimo_vocab(runtime, config.model)
    return build_mimo_model(runtime, config.model, vocab=vocab), runtime, vocab


def _runtime_data_config(
    config: MimoTrainConfig,
    vocab: MimoVocab | None,
) -> PreparedMimoDataConfig:
    if not config.data.derive_special_tokens:
        return config.data
    if vocab is None:
        if config.model.toy:
            return config.data
        raise ValueError("data.derive_special_tokens requires a runtime-backed model.")
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
    vocab: MimoVocab | None,
) -> PreparedMimoDataConfig:
    if not config.model.toy:
        return _runtime_data_config(config, vocab)
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


def mimo_callbacks(
    config: MimoTrainConfig,
    summary: LossSummary | None = None,
    schedule_runtime: ScheduleRuntime | None = None,
) -> list[Callback]:
    """Return callbacks safe for MimoBatch/MimoModule.

    Single-stream callbacks intentionally do not appear here: they expect
    ``ModelBatch``, ``Task`` and ``evaluate_text`` contracts that a MIMO run
    does not implement.
    """

    resolved_summary = LossSummary() if summary is None else summary
    resolved_schedule = (
        build_unit_schedule(config.optim.schedule)
        if schedule_runtime is None
        else schedule_runtime
    )
    callbacks: list[Callback] = [OOMDiagnostics()]
    callbacks.extend(resolved_schedule.callbacks())
    callbacks.extend(
        (
            resolved_summary,
            ObservationCallback(every_n_steps=config.trainer.log_every_n_steps),
        )
    )
    checkpoint = config.callbacks.checkpoint
    if checkpoint.enabled and config.trainer.enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(Path(config.output_dir) / "checkpoints"),
                filename=checkpoint.filename,
                save_last=checkpoint.save_last,
                save_top_k=checkpoint.save_top_k,
                every_n_train_steps=checkpoint.every_n_train_steps,
                auto_insert_metric_name=False,
                enable_version_counter=False,
            )
        )
    return callbacks


def _checkpoint_metadata(
    config: MimoTrainConfig,
    data: PreparedMimoDataConfig,
    vocab: MimoVocab | None,
) -> dict[str, object]:
    dataset = MimoDatasetConfig(
        samples_per_epoch=data.samples_per_epoch,
        seed=data.seed,
        max_sequence_length=data.max_sequence_length,
        task_weights=task_weights(data.task_weights),
    )
    return {
        "factory_config": asdict(config.model),
        "vocab": None if vocab is None else asdict(vocab),
        "data": {
            "special": asdict(data.special),
            "text_pad_token_id": data.text_pad_token_id,
            "audio_pad_token_id": data.audio_pad_token_id,
            "audio_delay_tokens": data.audio_delay_tokens,
            "task_weights": {
                task.value: weight
                for task, weight in dataset.resolved_task_weights.items()
            },
            "ignore_index": MIMO_IGNORE_INDEX,
        },
    }




if __name__ == "__main__":
    main()


__all__ = [
    "build_model",
    "main",
    "mimo_callbacks",
    "run",
]
