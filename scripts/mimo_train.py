"""Standalone Hydra entry for Kimi-style aligned MIMO pretraining."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence, cast

import hydra
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import Dataset

from speech_to_speech.datamodule.mimo import (
    MimoDataModule,
    MimoDatasetConfig,
    MimoSample,
    MimoSegment,
    MimoTask,
    MimoTaskDataset,
)
from speech_to_speech.model import MimoModel, MimoModelConfig
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
    train_dataset = build_dataset(data_config, data_config.factory, data_config.kwargs)
    validation_dataset = None
    if data_config.validation_factory is not None:
        validation_dataset = build_dataset(
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
        return _toy_model(config), None
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


def build_dataset(
    config: PreparedMimoDataConfig,
    factory_path: str,
    kwargs: Mapping[str, Any],
) -> Dataset[MimoSample]:
    """Instantiate a configured prepared dataset and normalize its contract."""

    factory = import_factory(factory_path)
    value = factory(**dict(kwargs))
    if config.kind == "samples":
        return _sample_dataset(value)
    if config.kind != "segments":
        raise ValueError("data.kind must be 'segments' or 'samples'.")
    source = _segment_source(value)
    weights = (
        {MimoTask(key): float(weight) for key, weight in config.task_weights.items()}
        if config.task_weights
        else None
    )
    return MimoTaskDataset(
        source,
        config.special,
        config=MimoDatasetConfig(
            samples_per_epoch=config.samples_per_epoch,
            seed=config.seed,
            max_sequence_length=config.max_sequence_length,
            task_weights=weights,
        ),
    )


def import_factory(path: str) -> Callable[..., Any]:
    if not isinstance(path, str) or not path:
        raise ValueError("dataset factory must be a non-empty import path.")
    if ":" in path:
        module_name, attribute = path.split(":", 1)
    else:
        module_name, _, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise ValueError("dataset factory must use module:attribute or module.attribute.")
    value = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(value):
        raise TypeError(f"dataset factory {path!r} is not callable.")
    return cast(Callable[..., Any], value)


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


def _segment_source(value: object) -> Any:
    if isinstance(value, Dataset):
        return value
    if isinstance(value, Sequence):
        values = tuple(value)
        if not values or any(not isinstance(item, MimoSegment) for item in values):
            raise TypeError("segment factory sequences must contain MimoSegment values.")
        return _SequenceDataset(values)
    raise TypeError("segment factory must return a Dataset or sequence of MimoSegment.")


def _sample_dataset(value: object) -> Dataset[MimoSample]:
    if isinstance(value, Dataset):
        return cast(Dataset[MimoSample], value)
    if isinstance(value, Sequence):
        values = tuple(value)
        if not values or any(not isinstance(item, MimoSample) for item in values):
            raise TypeError("sample factory sequences must contain MimoSample values.")
        return _SequenceDataset(values)
    raise TypeError("sample factory must return a Dataset or sequence of MimoSample.")


class _SequenceDataset(Dataset[Any]):
    def __init__(self, values: Sequence[Any]) -> None:
        self.values = tuple(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Any:
        return self.values[index]


class _ToyBody(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, *, inputs_embeds: Tensor, **_: object) -> object:
        hidden = self.norm(torch.tanh(self.proj(inputs_embeds)))
        return SimpleNamespace(
            last_hidden_state=(hidden, hidden),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


def _toy_model(config: MimoTrainConfig) -> MimoModel:
    # These dimensions intentionally match ToyMimoSegmentDataset defaults and
    # are overridable through ``data.kwargs`` for a small deterministic smoke.
    kwargs = config.data.kwargs
    text_vocab = int(kwargs.get("text_vocab_size", 128))
    audio_vocab = int(kwargs.get("audio_vocab_size", 256)) + 3
    hidden = int(getattr(config.model, "audio_embedding_dim", None) or 32)
    text_embedding = nn.Embedding(text_vocab, hidden)
    audio_embedding = nn.Embedding(audio_vocab, hidden)
    return MimoModel(
        _ToyBody(hidden),
        text_embedding=text_embedding,
        audio_embedding=audio_embedding,
        text_readout=__import__(
            "speech_to_speech.runtime.types", fromlist=["BackboneReadout"]
        ).BackboneReadout("last_hidden_state[0]"),
        audio_readout=__import__(
            "speech_to_speech.runtime.types", fromlist=["BackboneReadout"]
        ).BackboneReadout("last_hidden_state[1]"),
        config=MimoModelConfig(supports_cache_position=False),
    )


if __name__ == "__main__":
    main()


__all__ = [
    "build_dataset",
    "build_model",
    "import_factory",
    "main",
    "mimo_callbacks",
    "run",
]
