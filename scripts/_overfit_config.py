from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type, Union

from omegaconf import MISSING, DictConfig

from speech_to_speech.datamodule.config import SpeechConfig
from speech_to_speech.model import Config as ModelConfig
from speech_to_speech.model.acoustic import AcousticType
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.runtime import AudioSequenceLayout, Config as RuntimeConfig
from speech_to_speech.parameter_policy import ParameterPolicyConfig

if __package__:
    from ._config_common import (
        FlowModelConfig,
        LoggingConfig,
        PerformanceConfig,
        RVQModelConfig,
        TextRetentionCallbackConfig,
        TokenModelConfig,
        TrainConfig,
        TrainerConfig,
        non_empty_string,
        non_negative_integer,
        positive_integer,
        validate_training,
    )
    from ._config_normalization import parse, peft_lora, prepare
else:
    from _config_common import (
        FlowModelConfig,
        LoggingConfig,
        PerformanceConfig,
        RVQModelConfig,
        TextRetentionCallbackConfig,
        TokenModelConfig,
        TrainConfig,
        TrainerConfig,
        non_empty_string,
        non_negative_integer,
        positive_integer,
        validate_training,
    )
    from _config_normalization import parse, peft_lora, prepare


@dataclass
class TaskSampleCallbackConfig:
    enabled: bool = MISSING
    every_n_steps: int = MISSING


@dataclass
class EvaluationCallbackConfig:
    enabled: bool = True


@dataclass
class GradientProbeConfig:
    parameters: list[str] = field(default_factory=list)
    match: str = "exact"
    trainable_only: bool = True


@dataclass
class GradientProbeCallbackConfig:
    enabled: bool = MISSING
    every_n_steps: int = MISSING
    probes: dict[str, GradientProbeConfig] = field(default_factory=dict)
    partial_probes: dict[str, GradientProbeConfig] = field(default_factory=dict)


@dataclass
class FlowMatchingCallbackConfig:
    enabled: bool = MISSING
    every_n_steps: int = MISSING


@dataclass
class OverfitCallbacksConfig:
    parameter_policy: ParameterPolicyConfig = field(
        default_factory=ParameterPolicyConfig
    )
    task_sample: TaskSampleCallbackConfig = field(
        default_factory=TaskSampleCallbackConfig
    )
    evaluation: EvaluationCallbackConfig = field(
        default_factory=EvaluationCallbackConfig
    )
    text_retention: TextRetentionCallbackConfig = field(
        default_factory=TextRetentionCallbackConfig
    )
    gradient_probe: GradientProbeCallbackConfig = field(
        default_factory=GradientProbeCallbackConfig
    )
    flow_matching: FlowMatchingCallbackConfig = field(
        default_factory=FlowMatchingCallbackConfig
    )
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


@dataclass
class _OverfitConfig:
    task: str = MISSING
    sample_index: int = MISSING
    run_name: str = MISSING
    repo_output_root: str = MISSING
    output_subdir: str = MISSING
    output_dir: str = MISSING
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    audio_sequence_layout: AudioSequenceLayout = MISSING
    datamodule: SpeechConfig = MISSING
    pl_module: ModuleConfig = field(default_factory=ModuleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    callbacks: OverfitCallbacksConfig = field(default_factory=OverfitCallbacksConfig)


@dataclass
class OverfitTokenConfig(_OverfitConfig):
    model: TokenModelConfig = field(default_factory=TokenModelConfig)


@dataclass
class OverfitFlowConfig(_OverfitConfig):
    model: FlowModelConfig = field(default_factory=FlowModelConfig)


@dataclass
class OverfitRVQConfig(_OverfitConfig):
    model: RVQModelConfig = field(default_factory=RVQModelConfig)


OverfitConfig = Union[OverfitTokenConfig, OverfitFlowConfig, OverfitRVQConfig]


def overfit(config: DictConfig) -> OverfitConfig:
    config = prepare(config)
    lora = peft_lora(config)
    schema: Type[OverfitConfig]
    acoustic = AcousticType(str(config.model.acoustic.type))
    if acoustic is AcousticType.NONE:
        schema = OverfitTokenConfig
    elif acoustic is AcousticType.FLOW:
        schema = OverfitFlowConfig
    else:
        schema = OverfitRVQConfig
    result = parse(config, schema)
    result.model.lora = lora
    validate_training(result)
    non_negative_integer(result.sample_index, "sample_index")
    _validate_callbacks(result.callbacks)
    if result.callbacks.performance.enabled and result.callbacks.task_sample.enabled:
        raise ValueError(
            "overfit performance requires callbacks.task_sample.enabled=false "
            "because task sample generation cannot be excluded from distributed "
            "step timing."
        )
    return result


def _validate_callbacks(config: OverfitCallbacksConfig) -> None:
    positive_integer(
        config.task_sample.every_n_steps,
        "callbacks.task_sample.every_n_steps",
    )
    config.text_retention.validate()
    positive_integer(
        config.gradient_probe.every_n_steps,
        "callbacks.gradient_probe.every_n_steps",
    )
    _validate_gradient_probes(
        config.gradient_probe.probes,
        "callbacks.gradient_probe.probes",
    )
    _validate_gradient_probes(
        config.gradient_probe.partial_probes,
        "callbacks.gradient_probe.partial_probes",
    )
    if config.gradient_probe.enabled and not config.gradient_probe.probes:
        raise ValueError("enabled gradient probe requires at least one probe.")
    positive_integer(
        config.flow_matching.every_n_steps,
        "callbacks.flow_matching.every_n_steps",
    )


def _validate_gradient_probes(
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


__all__ = [
    "EvaluationCallbackConfig",
    "FlowMatchingCallbackConfig",
    "GradientProbeCallbackConfig",
    "GradientProbeConfig",
    "OverfitCallbacksConfig",
    "OverfitConfig",
    "OverfitFlowConfig",
    "OverfitRVQConfig",
    "OverfitTokenConfig",
    "TaskSampleCallbackConfig",
    "overfit",
]
