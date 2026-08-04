"""Structured configuration for the standalone MIMO training entry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from speech_to_speech.datamodule.config import (
    DataLoaderConfig,
    DataLoaderCostsConfig,
)
from speech_to_speech.mimo import MimoSpecialTokens
from speech_to_speech.model.mimo_factory import MimoFactoryConfig
from speech_to_speech.pl_module.optim import Config as OptimConfig
from speech_to_speech.runtime import (
    AudioSequenceLayout,
    BackboneInitialization,
    BackboneType,
    Config as RuntimeConfig,
    migrate_config_fields,
)


@dataclass
class MimoTrainValues:
    seed: int = 0
    max_steps: int = 1
    ckpt_path: str | None = None


@dataclass
class MimoTrainerConfig:
    accelerator: str = "cpu"
    devices: int | str = 1
    strategy: str = "auto"
    use_distributed_sampler: bool = True
    precision: str = "32-true"
    max_epochs: int = -1
    log_every_n_steps: int = 1
    enable_checkpointing: bool = False
    gradient_clip_val: float = 1.0


@dataclass
class MimoLoggingConfig:
    name: str = "csv"
    save_dir: str = "outputs"
    run_name: str = "mimo"


@dataclass
class MimoCheckpointConfig:
    enabled: bool = False
    every_n_train_steps: int = 1000
    save_last: bool = True
    save_top_k: int = -1
    filename: str = "step-{step:08d}"


@dataclass
class MimoCallbacksConfig:
    checkpoint: MimoCheckpointConfig = field(default_factory=MimoCheckpointConfig)


@dataclass
class PreparedMimoDataConfig:
    """A factory boundary for prepared segments or samples.

    ``factory`` is an import path in ``module:attribute`` or dotted form.  It
    must return a map-style Dataset (or a sequence) of ``MimoSegment`` values
    when ``kind=segments``; ``kind=samples`` accepts ``MimoSample`` values.
    Raw corpus parsing and codec materialization intentionally stay outside
    this entry.
    """

    factory: str = "speech_to_speech.datamodule.mimo.dataset:ToyMimoSegmentDataset"
    kwargs: dict[str, Any] = field(default_factory=dict)
    kind: str = "segments"
    samples_per_epoch: int | None = None
    seed: int = 0
    max_sequence_length: int | None = None
    task_weights: dict[str, float] = field(default_factory=dict)
    special: MimoSpecialTokens = field(
        default_factory=lambda: MimoSpecialTokens(
            text_bos=1,
            text_eos=2,
            text_blank=0,
            audio_bos=1,
            audio_eos=2,
            audio_blank=0,
            audio_delay_tokens=0,
        )
    )
    text_pad_token_id: int = 0
    audio_pad_token_id: int = 0
    derive_special_tokens: bool = False
    audio_delay_tokens: int = 0
    validation_factory: str | None = None
    validation_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class MimoTrainConfig:
    run_name: str = "mimo"
    repo_output_root: str = "outputs"
    output_subdir: str = "mimo"
    output_dir: str = "outputs/mimo"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    audio_sequence_layout: AudioSequenceLayout = AudioSequenceLayout.SEMANTIC
    model: MimoFactoryConfig = field(default_factory=MimoFactoryConfig)
    data: PreparedMimoDataConfig = field(default_factory=PreparedMimoDataConfig)
    dataloader: DataLoaderConfig = field(
        default_factory=lambda: DataLoaderConfig(batch_size=1, num_workers=0)
    )
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: MimoTrainValues = field(default_factory=MimoTrainValues)
    trainer: MimoTrainerConfig = field(default_factory=MimoTrainerConfig)
    logging: MimoLoggingConfig = field(default_factory=MimoLoggingConfig)
    callbacks: MimoCallbacksConfig = field(default_factory=MimoCallbacksConfig)


def parse(config: DictConfig | Mapping[str, Any]) -> MimoTrainConfig:
    """Resolve a Hydra config into validated Python dataclasses."""

    raw = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else dict(config)
    if not isinstance(raw, Mapping):
        raise TypeError("MIMO train config must resolve to a mapping.")
    runtime = _runtime(_mapping(raw, "runtime"))
    model = MimoFactoryConfig(**_mapping(raw, "model"))
    data = _data(_mapping(raw, "data"))
    loader = _loader(_mapping(raw, "dataloader"))
    train = MimoTrainValues(**_mapping(raw, "train"))
    trainer = MimoTrainerConfig(**_mapping(raw, "trainer"))
    logging = MimoLoggingConfig(**_mapping(raw, "logging"))
    callbacks = _callbacks(_mapping(raw, "callbacks"))
    optim_fields = _mapping(raw, "optim")
    # The shared SFT preset also carries a unit schedule for single-stream
    # staged training.  MimoModule currently consumes the base optimizer
    # fields; reject nothing, but keep the schedule out of that constructor.
    optim_fields.pop("schedule", None)
    result = MimoTrainConfig(
        run_name=str(raw.get("run_name", "mimo")),
        repo_output_root=str(raw.get("repo_output_root", "outputs")),
        output_subdir=str(raw.get("output_subdir", "mimo")),
        output_dir=str(raw.get("output_dir", "outputs/mimo")),
        runtime=runtime,
        audio_sequence_layout=AudioSequenceLayout(str(raw.get("audio_sequence_layout", "semantic"))),
        model=model,
        data=data,
        dataloader=loader,
        optim=OptimConfig(**optim_fields),
        train=train,
        trainer=trainer,
        logging=logging,
        callbacks=callbacks,
    )
    validate(result)
    return result


def validate(config: MimoTrainConfig) -> None:
    if not config.run_name or not config.output_subdir:
        raise ValueError("MIMO run_name and output_subdir must be non-empty.")
    output = Path(config.output_subdir)
    if output.is_absolute() or ".." in output.parts:
        raise ValueError("output_subdir must be relative and must not contain '..'.")
    expected = Path(config.repo_output_root).expanduser() / output
    if Path(config.output_dir).expanduser() != expected:
        raise ValueError("output_dir must equal repo_output_root/output_subdir.")
    if config.train.seed < 0 or config.train.max_steps <= 0:
        raise ValueError("train.seed must be non-negative and max_steps positive.")
    if config.dataloader.costs.enabled:
        raise ValueError("MIMO prepared data does not support cost-planned batches.")
    if config.data.kind not in {"segments", "samples"}:
        raise ValueError("data.kind must be 'segments' or 'samples'.")
    if not isinstance(config.data.derive_special_tokens, bool):
        raise TypeError("data.derive_special_tokens must be a boolean.")
    if config.data.audio_delay_tokens < 0:
        raise ValueError("data.audio_delay_tokens must be non-negative.")


def _runtime(value: Mapping[str, Any]) -> RuntimeConfig:
    fields = dict(value)
    migrate_config_fields(fields)
    fields["backbone_type"] = BackboneType(str(fields.get("backbone_type", "hf_causal_lm")))
    fields["backbone_initialization"] = BackboneInitialization(
        str(fields.get("backbone_initialization", "pretrained"))
    )
    return RuntimeConfig(**fields)


def _data(value: Mapping[str, Any]) -> PreparedMimoDataConfig:
    fields = dict(value)
    special = fields.get("special")
    if not isinstance(special, MimoSpecialTokens):
        fields["special"] = MimoSpecialTokens(**_mapping_value(special, "data.special"))
    weights = fields.get("task_weights", {})
    fields["task_weights"] = {str(key): float(val) for key, val in dict(weights).items()}
    return PreparedMimoDataConfig(**fields)


def _loader(value: Mapping[str, Any]) -> DataLoaderConfig:
    fields = dict(value)
    costs = fields.get("costs", {})
    if not isinstance(costs, DataLoaderCostsConfig):
        fields["costs"] = DataLoaderCostsConfig(**_mapping_value(costs, "dataloader.costs"))
    return DataLoaderConfig(**fields)


def _callbacks(value: Mapping[str, Any]) -> MimoCallbacksConfig:
    checkpoint = value.get("checkpoint", {})
    return MimoCallbacksConfig(
        checkpoint=(
            checkpoint
            if isinstance(checkpoint, MimoCheckpointConfig)
            else MimoCheckpointConfig(**_mapping_value(checkpoint, "callbacks.checkpoint"))
        )
    )


def _mapping(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    nested = value.get(key, {})
    return _mapping_value(nested, key)


def _mapping_value(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


__all__ = [
    "MimoCallbacksConfig",
    "MimoCheckpointConfig",
    "MimoLoggingConfig",
    "MimoTrainConfig",
    "MimoTrainValues",
    "MimoTrainerConfig",
    "PreparedMimoDataConfig",
    "parse",
    "validate",
]
