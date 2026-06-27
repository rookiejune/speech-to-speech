"""Real-data smoke runner for the speech-to-speech training stack.

This module loads the project YAML config, prepares or loads the LongCat BPE
tokenizer from anydataset speech pairs, and runs a short Lightning training
loop. It only owns top-level orchestration; sample schema rules live in
`datamodule`, model assembly lives in `model`, and training behavior lives in
`pl_module`.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from anydataset import AnyDataset, MultipleAnyDataset, WeightedRandomStrategy
from lightning.pytorch import Trainer, seed_everything

from .config import (
    BPEConfig,
    DataConfig,
    LoRAConfig,
    ModelConfig,
    SpeechToSpeechConfig,
    TaskConfig,
    TrainConfig,
)
from .datamodule import SpeechToSpeechDataModule
from .datamodule.example import speech_pair_from_sample
from .model.orchestrator import Orchestrator
from .pl_module import SpeechToSpeechModule
from .runtime import prepare_longcat_tokenizer, qwen3_tokenizer
from .types import SpeechPair


def load_config(path: str | Path) -> SpeechToSpeechConfig:
    payload = _load_yaml(path)
    data = _mapping(payload, "config")
    return SpeechToSpeechConfig(
        data=_data_config(_required_mapping(data, "data")),
        bpe=_bpe_config(_optional_mapping(data, "bpe")),
        tasks=_task_config(_optional_mapping(data, "tasks")),
        model=_model_config(_optional_mapping(data, "model")),
        train=_train_config(_optional_mapping(data, "train")),
    )


def run_smoke(
    config_path: str | Path,
    *,
    max_steps: int | None = None,
    default_root_dir: str | Path = "outputs/smoke",
    accelerator: str | None = None,
    devices: int | str | None = None,
    enable_progress_bar: bool = True,
) -> Trainer:
    config = load_config(config_path)
    if max_steps is not None:
        config = replace(config, train=replace(config.train, max_steps=max_steps))

    seed_everything(config.train.seed, workers=True)
    tokenizer = qwen3_tokenizer(config.model)
    bpe = prepare_longcat_tokenizer(
        _speech_pairs(config.data),
        datasets=config.data.datasets,
        config=config.bpe,
    )
    model = Orchestrator(
        model_config=config.model,
        bpe_config=config.bpe,
        tokenizer=tokenizer,
        bpe_vocab_size=bpe.vocab_size,
    )
    module = SpeechToSpeechModule(model, config.train)
    datamodule = SpeechToSpeechDataModule(
        config.data,
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
        args.config,
        max_steps=args.max_steps,
        default_root_dir=args.default_root_dir,
        accelerator=args.accelerator,
        devices=args.devices,
        enable_progress_bar=not args.no_progress_bar,
    )
    print(f"smoke finished: global_step={trainer.global_step}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real-data speech-to-speech smoke test.")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/qwen3_smoke.yaml",
        help="Path to the speech-to-speech YAML config.",
    )
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


def _speech_pairs(data: DataConfig) -> Iterable[SpeechPair]:
    source = _dataset(data)
    for sample in source:
        yield speech_pair_from_sample(sample)


def _dataset(data: DataConfig) -> AnyDataset | MultipleAnyDataset:
    datasets = tuple(AnyDataset(dataset, cache_root=data.cache_root) for dataset in data.datasets)
    if len(datasets) == 1:
        return datasets[0]
    return MultipleAnyDataset(datasets, strategy=WeightedRandomStrategy())


def _accelerator(train: TrainConfig) -> str:
    if train.device in {"cuda", "gpu"}:
        return "cuda"
    return train.device


def _load_yaml(path: str | Path) -> object:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to load speech-to-speech configs.") from error

    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _data_config(data: Mapping[str, object]) -> DataConfig:
    return DataConfig(
        datasets=_str_tuple(data, "datasets", required=True),
        cache_root=_optional_str(data, "cache_root"),
        lba_log_dir=_optional_str(data, "lba_log_dir"),
        batch_size=_int(data, "batch_size", DataConfig(datasets=("unused",)).batch_size),
        num_workers=_int(data, "num_workers", DataConfig(datasets=("unused",)).num_workers),
        pin_memory=_bool(data, "pin_memory", DataConfig(datasets=("unused",)).pin_memory),
        drop_last=_bool(data, "drop_last", DataConfig(datasets=("unused",)).drop_last),
    )


def _bpe_config(data: Mapping[str, object] | None) -> BPEConfig:
    defaults = BPEConfig()
    data = data or {}
    return BPEConfig(
        cache_dir_env=_str(data, "cache_dir_env", defaults.cache_dir_env),
        codec_name=_str(data, "codec_name", defaults.codec_name),
        vocab_size=_int(data, "vocab_size", defaults.vocab_size),
        max_piece_frames=_int(data, "max_piece_frames", defaults.max_piece_frames),
    )


def _task_config(data: Mapping[str, object] | None) -> TaskConfig:
    defaults = TaskConfig()
    data = data or {}
    return TaskConfig(
        enabled=_str_tuple(data, "enabled", default=defaults.enabled),
    )


def _model_config(data: Mapping[str, object] | None) -> ModelConfig:
    defaults = ModelConfig()
    data = data or {}
    return ModelConfig(
        model_name_or_path=_str(data, "model_name_or_path", defaults.model_name_or_path),
        trust_remote_code=_bool(data, "trust_remote_code", defaults.trust_remote_code),
        load_in_4bit=_bool(data, "load_in_4bit", defaults.load_in_4bit),
        train_text_embedding=_bool(data, "train_text_embedding", defaults.train_text_embedding),
        train_audio_embedding=_bool(data, "train_audio_embedding", defaults.train_audio_embedding),
        train_audio_special_tokens=_bool(
            data,
            "train_audio_special_tokens",
            defaults.train_audio_special_tokens,
        ),
        train_backbone=_bool(data, "train_backbone", defaults.train_backbone),
        train_dit=_bool(data, "train_dit", defaults.train_dit),
        lora=_lora_config(_optional_mapping(data, "lora")),
    )


def _lora_config(data: Mapping[str, object] | None) -> LoRAConfig:
    defaults = LoRAConfig()
    data = data or {}
    return LoRAConfig(
        enabled=_bool(data, "enabled", defaults.enabled),
        rank=_int(data, "rank", defaults.rank),
        alpha=_int(data, "alpha", defaults.alpha),
        dropout=_float(data, "dropout", defaults.dropout),
        targets=_str_tuple(data, "targets", default=defaults.targets),
    )


def _train_config(data: Mapping[str, object] | None) -> TrainConfig:
    defaults = TrainConfig()
    data = data or {}
    return TrainConfig(
        max_steps=_int(data, "max_steps", defaults.max_steps),
        learning_rate=_float(data, "learning_rate", defaults.learning_rate),
        optimizer_preset=_str(data, "optimizer_preset", defaults.optimizer_preset),
        optimizer=_str(data, "optimizer", defaults.optimizer),
        weight_decay=_optional_float(data, "weight_decay", defaults.weight_decay),
        schedule=_str(data, "schedule", defaults.schedule),
        warmup_steps=_int(data, "warmup_steps", defaults.warmup_steps),
        stable_steps=_optional_int(data, "stable_steps", defaults.stable_steps),
        decay_steps=_optional_int(data, "decay_steps", defaults.decay_steps),
        min_lr_ratio=_float(data, "min_lr_ratio", defaults.min_lr_ratio),
        seed=_int(data, "seed", defaults.seed),
        device=_str(data, "device", defaults.device),
        precision=_str(data, "precision", defaults.precision),
    )


def _required_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    if key not in data:
        raise KeyError(f"config.{key} is required.")
    return _mapping(data[key], f"config.{key}")


def _optional_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object] | None:
    value = data.get(key)
    if value is None:
        return None
    return _mapping(value, f"config.{key}")


def _mapping(data: object, path: str) -> Mapping[str, object]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    return data


def _str(data: Mapping[str, object], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string.")
    return value


def _optional_str(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string or null.")
    return value


def _str_tuple(
    data: Mapping[str, object],
    key: str,
    *,
    required: bool = False,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if key not in data:
        if required:
            raise KeyError(f"{key} is required.")
        return default
    value = data[key]
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{key} must be a sequence of strings.")
    if not all(isinstance(item, str) for item in value):
        raise TypeError(f"{key} must contain only strings.")
    return tuple(value)


def _bool(data: Mapping[str, object], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise TypeError(f"{key} must be a boolean.")
    return value


def _int(data: Mapping[str, object], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer.")
    return value


def _optional_int(data: Mapping[str, object], key: str, default: int | None) -> int | None:
    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer or null.")
    return value


def _float(data: Mapping[str, object], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key} must be a number.")
    return float(value)


def _optional_float(
    data: Mapping[str, object],
    key: str,
    default: float | None,
) -> float | None:
    value = data.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key} must be a number or null.")
    return float(value)


def _devices(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
