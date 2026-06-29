"""Real-data smoke runner for the speech-to-speech training stack.

This module composes Hydra configs, prepares or loads the LongCat BPE
tokenizer from anydataset speech pairs, and runs a short Lightning training
loop. It only owns top-level orchestration; sample schema rules live in
`datamodule`, model assembly lives in `model`, and training behavior lives in
`pl_module`.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, get_origin, get_type_hints

from hydra import compose, initialize_config_dir
from lightning.pytorch import Trainer, seed_everything
from omegaconf import DictConfig, ListConfig, OmegaConf

from .config import (
    BPEConfig,
    DataLoaderConfig,
    DataModuleConfig,
    DatasetFactoryConfig,
    LBAConfig,
    LoRAConfig,
    ModelConfig,
    SpeechToSpeechConfig,
    TaskConfig,
    TaskWeightsConfig,
    TrainConfig,
    TrainerConfig,
)
from .dataset import dataset_metadata, training_dataset
from .datamodule import SpeechToSpeechDataModule
from .datamodule.example import speech_pair_from_sample
from .model.DiT.model import DiT
from .model.orchestrator import Orchestrator, dit_config
from .pl_module import SpeechToSpeechModule
from .runtime import longcat_codec, prepare_longcat_tokenizer, qwen3_tokenizer
from .types import SpeechPair


def load_config(
    config_name: str = "config",
    *,
    overrides: Sequence[str] = (),
    config_dir: str | Path = "configs",
) -> SpeechToSpeechConfig:
    root = Path(config_dir)
    if not root.is_absolute():
        root = Path.cwd() / root
    with initialize_config_dir(config_dir=str(root), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    payload = OmegaConf.to_container(cfg, resolve=True)
    return _speech_to_speech_config(_mapping(payload, "config"))


def run_smoke(
    config_name: str = "config",
    *,
    overrides: Sequence[str] = (),
    config_dir: str | Path = "configs",
    max_steps: int | None = None,
    default_root_dir: str | Path = "outputs/smoke",
    accelerator: str | None = None,
    devices: int | str | None = None,
    enable_progress_bar: bool = True,
) -> Trainer:
    config = load_config(config_name, overrides=overrides, config_dir=config_dir)
    if max_steps is not None:
        config = replace(config, train=replace(config.train, max_steps=max_steps))

    seed_everything(config.train.seed, workers=True)
    tokenizer = qwen3_tokenizer(config.model)
    bpe = prepare_longcat_tokenizer(
        partial(_speech_pairs, config.datamodule.dataset_factory),
        datasets=dataset_metadata(config.datamodule.dataset_factory),
        config=config.bpe,
    )
    acoustic_training = config.train.acoustic_loss_weight > 0.0
    model = Orchestrator(
        dit=DiT(dit_config()) if acoustic_training else None,
        model_config=config.model,
        bpe_config=config.bpe,
        tokenizer=tokenizer,
        bpe_vocab_size=bpe.vocab_size,
    )
    module = SpeechToSpeechModule(
        model,
        config.train,
        bpe=bpe if acoustic_training else None,
        acoustic_feature_extractor=longcat_codec() if acoustic_training else None,
    )
    datamodule = SpeechToSpeechDataModule(
        config.datamodule,
        config.tasks,
        model.embed_tokens,
        tokenizer=tokenizer,
        bpe_tokenizer=bpe,
        bpe=config.bpe,
    )

    trainer_kwargs: dict[str, object] = {
        "default_root_dir": str(default_root_dir),
        "max_steps": config.train.max_steps,
        "accelerator": accelerator or _accelerator(config.train),
        "precision": config.train.precision,
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "enable_progress_bar": enable_progress_bar,
    }
    if devices is not None:
        trainer_kwargs["devices"] = devices
    else:
        trainer_kwargs["devices"] = 1

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(module, datamodule=datamodule)
    return trainer


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    trainer = run_smoke(
        args.config_name,
        overrides=args.overrides,
        config_dir=args.config_dir,
        max_steps=args.max_steps,
        default_root_dir=args.default_root_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        enable_progress_bar=not args.no_progress_bar,
    )
    print(f"smoke finished: global_step={trainer.global_step}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real-data speech-to-speech smoke test.")
    parser.add_argument("config_name", nargs="?", default="config")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--max-steps", type=int, help="Override train.max_steps.")
    parser.add_argument(
        "--default-root-dir",
        default="outputs/smoke",
        help="Lightning default_root_dir for smoke outputs.",
    )
    parser.add_argument("--accelerator", help="Override Lightning accelerator.")
    parser.add_argument("--devices", type=_devices, help="Override Lightning devices.")
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable Lightning progress output.",
    )
    return parser.parse_args(argv)


def _speech_pairs(config: DatasetFactoryConfig) -> Iterable[SpeechPair]:
    for sample in training_dataset(config):
        yield speech_pair_from_sample(sample)


def _accelerator(train: TrainConfig) -> str:
    if train.device in {"cuda", "gpu"}:
        return "cuda"
    return train.device


def _speech_to_speech_config(data: Mapping[str, object]) -> SpeechToSpeechConfig:
    return SpeechToSpeechConfig(
        datamodule=_datamodule_config(_optional_mapping(data, "datamodule")),
        bpe=_from_mapping(BPEConfig, _optional_mapping(data, "bpe")),
        tasks=_task_config(_optional_mapping(data, "tasks")),
        model=_model_config(_optional_mapping(data, "model")),
        train=_from_mapping(TrainConfig, _optional_mapping(data, "train")),
        trainer=_from_mapping(TrainerConfig, _optional_mapping(data, "trainer")),
    )


def _datamodule_config(data: Mapping[str, object] | None) -> DataModuleConfig:
    data = data or {}
    return DataModuleConfig(
        dataset_factory=_from_mapping(
            DatasetFactoryConfig,
            _optional_mapping(data, "dataset_factory"),
        ),
        dataloader=_from_mapping(DataLoaderConfig, _optional_mapping(data, "dataloader")),
        lba=_from_mapping(LBAConfig, _optional_mapping(data, "lba")),
    )


def _model_config(data: Mapping[str, object] | None) -> ModelConfig:
    data = data or {}
    return ModelConfig(
        **_dataclass_kwargs(ModelConfig, data, skip={"lora"}),
        lora=_from_mapping(LoRAConfig, _optional_mapping(data, "lora")),
    )


def _task_config(data: Mapping[str, object] | None) -> TaskConfig:
    data = data or {}
    return TaskConfig(
        **_dataclass_kwargs(TaskConfig, data, skip={"weights"}),
        weights=_from_mapping(TaskWeightsConfig, _optional_mapping(data, "weights")),
    )


def _from_mapping(cls: type[Any], data: Mapping[str, object] | None) -> Any:
    data = data or {}
    return cls(**_dataclass_kwargs(cls, data))


def _dataclass_kwargs(
    cls: type[Any],
    data: Mapping[str, object],
    *,
    skip: set[str] | None = None,
) -> dict[str, object]:
    if not is_dataclass(cls):
        raise TypeError("config target must be a dataclass type.")
    skip = skip or set()
    hints = get_type_hints(cls)
    allowed = {field.name for field in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise KeyError(f"unknown {cls.__name__} field(s): {names}")
    kwargs: dict[str, object] = {}
    for name in allowed:
        if name in skip or name not in data:
            continue
        kwargs[name] = _coerce_value(data[name], hints[name], name)
    return kwargs


def _coerce_value(value: object, annotation: object, name: str) -> object:
    if isinstance(value, ListConfig):
        value = list(value)
    if isinstance(value, DictConfig):
        value = dict(value)
    origin = get_origin(annotation)
    if origin is tuple:
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            raise TypeError(f"{name} must be a sequence.")
        return tuple(value)
    return value


def _optional_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object] | None:
    value = data.get(key)
    if value is None:
        return None
    return _mapping(value, f"config.{key}")


def _mapping(data: object, path: str) -> Mapping[str, object]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    return data


def _devices(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
