"""Shared training-entry configuration contracts and validation."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Union

from anydataset.types import AudioView
from omegaconf import MISSING

from speech_to_speech.datamodule.config import SpeechConfig
from speech_to_speech.model import Config as ModelConfig
from speech_to_speech.model.acoustic import (
    AcousticNoneConfig,
    AcousticType,
    FlowConfig,
    RepaConfig,
    RVQConfig,
)
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.runtime import (
    AudioSequenceLayout,
    BackboneInitialization,
)
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.training.parameter_policy import (
    ParameterGroup,
    ParameterPolicyConfig,
    ParameterPolicyName,
)
from speech_to_speech.callback.schedule import SUPPORTED_UNIT_NAMES
from speech_to_speech.pl_module.optim import Config as OptimBaseConfig


@dataclass
class TokenModelConfig(ModelConfig):
    acoustic: AcousticNoneConfig = field(default_factory=AcousticNoneConfig)


@dataclass
class FlowModelConfig(ModelConfig):
    acoustic: FlowConfig = field(default_factory=FlowConfig)


@dataclass
class RVQModelConfig(ModelConfig):
    acoustic: RVQConfig = field(default_factory=RVQConfig)


EntryModelConfig = Union[TokenModelConfig, FlowModelConfig, RVQModelConfig]


@dataclass
class TrainConfig:
    seed: int = MISSING
    max_steps: int = MISSING


@dataclass
class TrainerConfig:
    accelerator: str = MISSING
    devices: Union[int, str] = MISSING
    strategy: str = MISSING
    use_distributed_sampler: bool = MISSING
    precision: str = MISSING
    max_epochs: int = MISSING
    log_every_n_steps: int = MISSING
    enable_checkpointing: bool = MISSING
    gradient_clip_val: float = MISSING


@dataclass
class LoggingConfig:
    name: str = MISSING
    save_dir: str = MISSING
    run_name: str = MISSING
    version: Optional[Union[int, str]] = None


@dataclass
class PerformanceConfig:
    enabled: bool = MISSING
    hardware_peak_flops: Optional[float] = MISSING
    log_every_n_steps: int = MISSING
    warmup_steps: int = MISSING
    measure_window_steps: int = MISSING
    sync_cuda: bool = MISSING
    sync_distributed: bool = MISSING
    stop_after_measurement: bool = False


@dataclass
class GradientProbeConfig:
    parameters: list[str] = field(default_factory=list)
    match: str = "exact"
    trainable_only: bool = True


@dataclass
class GradientTargetConfig:
    loss: str = MISSING
    group: str = "batch"


@dataclass
class GradientComparisonConfig:
    left: GradientTargetConfig = field(default_factory=GradientTargetConfig)
    right: GradientTargetConfig = field(default_factory=GradientTargetConfig)


@dataclass
class UnitScheduleCurveConfig:
    type: str = "constant"
    value: Optional[float] = None
    start: Optional[float] = None
    end: Optional[float] = None


@dataclass
class UnitSchedulePhaseConfig:
    name: str = MISSING
    duration: float = MISSING
    lr: UnitScheduleCurveConfig = field(default_factory=UnitScheduleCurveConfig)


@dataclass
class UnitScheduleConfig:
    unit: str = "tokens"
    log_every_n_units: Optional[float] = None
    measure_window_batches: int = 100
    sync_cuda: bool = True
    sync_distributed: bool = True
    allow_external_lr_changes: bool = False
    stop_at: Optional[float] = None
    stop_at_end: bool = False
    phases: list[UnitSchedulePhaseConfig] = field(default_factory=list)


@dataclass
class OptimConfig(OptimBaseConfig):
    schedule: UnitScheduleConfig = field(default_factory=UnitScheduleConfig)


@dataclass
class TextProbeConfig:
    instruction: str = MISSING
    reference: str = MISSING


@dataclass
class TextRetentionCallbackConfig:
    enabled: bool = False
    every_n_steps: int = 10_000
    max_new_tokens: int = 128
    probes: dict[str, TextProbeConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        positive_integer(
            self.every_n_steps,
            "text retention every_n_steps",
        )
        positive_integer(
            self.max_new_tokens,
            "text retention max_new_tokens",
        )
        if not self.enabled:
            return
        if not self.probes:
            raise ValueError("enabled text retention requires at least one probe.")
        for name, probe in self.probes.items():
            non_empty_string(name, "text retention probe name")
            non_empty_string(
                probe.instruction,
                f"text retention probe {name!r} instruction",
            )
            non_empty_string(
                probe.reference,
                f"text retention probe {name!r} reference",
            )


class _TrainValues(Protocol):
    @property
    def seed(self) -> int: ...

    @property
    def max_steps(self) -> int: ...


class _Callbacks(Protocol):
    @property
    def performance(self) -> PerformanceConfig: ...

    @property
    def parameter_policy(self) -> ParameterPolicyConfig: ...


class _EntryConfig(Protocol):
    @property
    def repo_output_root(self) -> str: ...

    @property
    def output_subdir(self) -> str: ...

    @property
    def output_dir(self) -> str: ...

    @property
    def model(self) -> EntryModelConfig: ...

    @property
    def runtime(self) -> RuntimeConfig: ...

    @property
    def audio_sequence_layout(self) -> AudioSequenceLayout: ...

    @property
    def datamodule(self) -> SpeechConfig: ...

    @property
    def pl_module(self) -> ModuleConfig: ...

    @property
    def optim(self) -> OptimConfig: ...

    @property
    def train(self) -> _TrainValues: ...

    @property
    def trainer(self) -> TrainerConfig: ...

    @property
    def callbacks(self) -> _Callbacks: ...


def validate_training(config: _EntryConfig) -> None:
    non_negative_integer(config.train.seed, "train.seed")
    if config.datamodule.streaming.enabled or config.datamodule.source.enabled:
        source_toy = (
            config.datamodule.source.enabled
            and config.datamodule.source.mode == "toy"
        )
        if source_toy and config.train.max_steps != -1:
            positive_integer(config.train.max_steps, "train.max_steps")
        elif not source_toy and config.train.max_steps != -1:
            raise ValueError(
                "streaming synthesis requires train.max_steps=-1 so only the "
                "sealed logical epoch can stop training."
            )
    else:
        positive_integer(config.train.max_steps, "train.max_steps")
    positive_integer(
        config.trainer.log_every_n_steps,
        "trainer.log_every_n_steps",
    )
    _validate_optim(config.optim)
    _validate_performance(config.callbacks.performance)
    _validate_output(config)
    _validate_audio_sequence_layout(config)
    _validate_backbone_initialization(config)
    _validate_lora(config)


def optional_positive_number(value: object, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None.")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def non_negative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")


def validate_gradient_probes(
    probes: dict[str, GradientProbeConfig],
    path: str,
) -> None:
    if not isinstance(probes, dict):
        raise TypeError(f"{path} must be a mapping.")
    for name, probe in probes.items():
        non_empty_string(name, f"{path} probe name")
        if probe.match not in {"exact", "regex"}:
            raise ValueError(f"{path}.{name}.match must be 'exact' or 'regex'.")
        if not isinstance(probe.trainable_only, bool):
            raise TypeError(f"{path}.{name}.trainable_only must be a boolean.")
        if not probe.parameters:
            raise ValueError(f"{path}.{name}.parameters must not be empty.")
        for index, parameter in enumerate(probe.parameters):
            non_empty_string(
                parameter,
                f"{path}.{name}.parameters[{index}]",
            )


def validate_gradient_comparisons(
    comparisons: list[GradientComparisonConfig],
    path: str,
) -> None:
    if not isinstance(comparisons, list):
        raise TypeError(f"{path} must be a list.")
    for index, comparison in enumerate(comparisons):
        for side, target in (("left", comparison.left), ("right", comparison.right)):
            target_path = f"{path}[{index}].{side}"
            non_empty_string(target.loss, f"{target_path}.loss")
            non_empty_string(target.group, f"{target_path}.group")


def _validate_performance(config: PerformanceConfig) -> None:
    positive_integer(
        config.log_every_n_steps,
        "callbacks.performance.log_every_n_steps",
    )
    non_negative_integer(
        config.warmup_steps,
        "callbacks.performance.warmup_steps",
    )
    positive_integer(
        config.measure_window_steps,
        "callbacks.performance.measure_window_steps",
    )
    if not isinstance(config.stop_after_measurement, bool):
        raise TypeError(
            "callbacks.performance.stop_after_measurement must be a boolean."
        )


def _validate_output(config: _EntryConfig) -> None:
    subdir = Path(config.output_subdir)
    if subdir == Path(".") or subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError(
            "output_subdir must be a non-empty relative path without '..'."
        )
    expected = Path(config.repo_output_root).expanduser() / subdir
    if Path(config.output_dir).expanduser() != expected:
        raise ValueError("output_dir must equal repo_output_root/output_subdir.")


def _validate_audio_sequence_layout(config: _EntryConfig) -> None:
    acoustic = AcousticType(config.model.acoustic.type)
    if (
        config.audio_sequence_layout is AudioSequenceLayout.FLATTENED
        and acoustic is not AcousticType.NONE
    ):
        raise ValueError(
            "audio_sequence_layout=flattened requires model/acoustic=none "
            "because full codec codes are trained as sequence tokens."
        )
    if (
        config.runtime.audio_output.acoustic_generator_artifact is not None
        and config.audio_sequence_layout is AudioSequenceLayout.FLATTENED
    ):
        raise ValueError(
            "runtime.audio_output.acoustic_generator_artifact requires "
            "audio_sequence_layout=semantic."
        )
    if (
        config.runtime.audio_output.acoustic_generator_artifact is not None
        and acoustic is not AcousticType.NONE
    ):
        raise ValueError(
            "runtime.audio_output.acoustic_generator_artifact requires "
            "model/acoustic=none."
        )
    if (
        config.runtime.audio_output.audio_view is AudioView.BICODEC
        and acoustic is not AcousticType.NONE
    ):
        raise ValueError(
            "BiCodec global units require model/acoustic=none and "
            "audio_sequence_layout=flattened."
        )
    if (
        acoustic is AcousticType.NONE
        and config.runtime.audio_output.audio_view is AudioView.LONGCAT
        and config.audio_sequence_layout is AudioSequenceLayout.SEMANTIC
        and config.runtime.audio_output.acoustic_generator_artifact is None
    ):
        raise ValueError(
            "audio_sequence_layout=semantic with model/acoustic=none requires "
            "runtime.audio_output.acoustic_generator_artifact; use "
            "audio_sequence_layout=flattened "
            "for token-only training."
        )


def _validate_backbone_initialization(config: _EntryConfig) -> None:
    if config.runtime.backbone_initialization is not BackboneInitialization.RANDOM:
        return
    if config.model.toy is not None:
        raise ValueError(
            "runtime.backbone_initialization=random cannot be combined with model.toy."
        )
    policy = config.callbacks.parameter_policy.spec()
    if (
        ParameterGroup.BACKBONE not in policy.trainable_groups
        or (
            policy.backbone_top_fraction is not None
            and policy.backbone_top_fraction < 1
        )
    ):
        raise ValueError(
            "random backbone initialization requires a fully trainable backbone; "
            "select callbacks.parameter_policy=full."
        )


def _validate_lora(config: _EntryConfig) -> None:
    enabled = config.model.lora is not None
    selected = config.callbacks.parameter_policy.name is ParameterPolicyName.LORA
    if enabled != selected:
        raise ValueError(
            "model/lora and callbacks.parameter_policy=lora must be selected "
            "together."
        )
    if enabled and config.model.lora is not None and config.model.lora.inference_mode:
        raise ValueError("training requires model.lora.inference_mode=false.")
    if enabled and config.callbacks.performance.enabled:
        warnings.warn(
            "LoRA performance metrics use approximate FLOPs: the provider counts the "
            "wrapped backbone's dense base projections with the conventional full-training "
            "backward multiplier and omits the low-rank adapter projections. Use MFU for "
            "smoke and relative diagnostics, not exact accounting.",
            UserWarning,
            stacklevel=3,
        )
    if enabled and config.optim.name == "muon":
        init = config.model.lora.init_lora_weights if config.model.lora is not None else None
        if not isinstance(init, str) or not init.startswith("pissa"):
            raise ValueError(
                "optim.name=muon with LoRA requires model.lora.init_lora_weights "
                "to be a pissa initialization (for example 'pissa')."
            )


def _validate_optim(config: OptimConfig) -> None:
    if config.name not in {"adamw", "muon"}:
        raise ValueError("optim.name must be adamw or muon.")
    _positive_number(config.learning_rate, "optim.learning_rate")
    _non_negative_number(config.weight_decay, "optim.weight_decay")
    _validate_unit_schedule(config.schedule)


def _validate_unit_schedule(config: UnitScheduleConfig) -> None:
    if config.unit not in SUPPORTED_UNIT_NAMES:
        raise ValueError(
            "optim.schedule.unit must be one of "
            + ", ".join(sorted(SUPPORTED_UNIT_NAMES))
        )
    if config.log_every_n_units is not None:
        _positive_number(config.log_every_n_units, "optim.schedule.log_every_n_units")
    positive_integer(
        config.measure_window_batches,
        "optim.schedule.measure_window_batches",
    )
    if not isinstance(config.sync_cuda, bool):
        raise TypeError("optim.schedule.sync_cuda must be a boolean.")
    if not isinstance(config.sync_distributed, bool):
        raise TypeError("optim.schedule.sync_distributed must be a boolean.")
    if not isinstance(config.allow_external_lr_changes, bool):
        raise TypeError(
            "optim.schedule.allow_external_lr_changes must be a boolean."
        )
    if config.stop_at is not None:
        _positive_number(config.stop_at, "optim.schedule.stop_at")
    if not isinstance(config.stop_at_end, bool):
        raise TypeError("optim.schedule.stop_at_end must be a boolean.")
    if not config.phases:
        raise ValueError("optim.schedule requires phases.")
    seen: set[str] = set()
    for index, phase in enumerate(config.phases):
        path = f"optim.schedule.phases[{index}]"
        non_empty_string(phase.name, f"{path}.name")
        if phase.name in seen:
            raise ValueError(f"duplicate optim.schedule phase {phase.name!r}.")
        seen.add(phase.name)
        _positive_number(phase.duration, f"{path}.duration")
        _validate_schedule_curve(phase.lr, f"{path}.lr")


def _validate_schedule_curve(config: UnitScheduleCurveConfig, path: str) -> None:
    if config.type not in {"constant", "linear", "cosine"}:
        raise ValueError(f"{path}.type must be constant, linear, or cosine.")
    if config.value is not None:
        _non_negative_number(config.value, f"{path}.value")
    if config.start is not None:
        _non_negative_number(config.start, f"{path}.start")
    if config.end is not None:
        _non_negative_number(config.end, f"{path}.end")


def _positive_number(value: object, name: str) -> None:
    number = _number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive.")


def _non_negative_number(value: object, name: str) -> None:
    number = _number(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative.")


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


__all__ = [
    "AcousticNoneConfig",
    "FlowConfig",
    "FlowModelConfig",
    "GradientComparisonConfig",
    "GradientProbeConfig",
    "GradientTargetConfig",
    "LoggingConfig",
    "OptimConfig",
    "PerformanceConfig",
    "RVQConfig",
    "RVQModelConfig",
    "RepaConfig",
    "TextProbeConfig",
    "TextRetentionCallbackConfig",
    "TokenModelConfig",
    "TrainConfig",
    "TrainerConfig",
    "UnitScheduleConfig",
    "UnitScheduleCurveConfig",
    "UnitSchedulePhaseConfig",
    "non_empty_string",
    "non_negative_integer",
    "optional_positive_number",
    "positive_integer",
    "validate_training",
    "validate_gradient_comparisons",
    "validate_gradient_probes",
]
