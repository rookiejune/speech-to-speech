from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Union

from anydataset.types import AudioView
from omegaconf import MISSING

from speech_to_speech.audio_route import Config as AudioRouteConfig
from speech_to_speech.datamodule.config import SpeechConfig
from speech_to_speech.datamodule.dataset.speech import DatasetName
from speech_to_speech.model import Config as ModelConfig
from speech_to_speech.model.acoustic import AcousticType, DecoderConfig
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.runtime import AudioRepresentation, BackboneInitialization
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import validate_audio_route
from speech_to_speech.stage import (
    ParameterGroup,
    ParameterPolicyConfig,
    ParameterPolicyName,
)


@dataclass
class RepaConfig:
    weight: Optional[float] = None
    teacher_checkpoint: str = "microsoft/wavlm-base"
    teacher_layer: int = 9
    student_layer: Optional[int] = None


@dataclass
class FlowConfig:
    type: str = AcousticType.FLOW.value
    name: str = MISSING
    init_artifact: Optional[str] = None
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    repa: RepaConfig = field(default_factory=RepaConfig)

    def __post_init__(self) -> None:
        _validate_init_artifact(self.init_artifact)


@dataclass
class RVQConfig:
    type: str = AcousticType.RVQ.value
    name: str = MISSING
    init_artifact: Optional[str] = None
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    def __post_init__(self) -> None:
        _validate_init_artifact(self.init_artifact)


@dataclass
class AcousticNoneConfig:
    type: str = AcousticType.NONE.value
    name: str = "token"


AcousticConfig = Union[AcousticNoneConfig, FlowConfig, RVQConfig]


def _validate_init_artifact(value: Optional[str]) -> None:
    if value is not None and not value:
        raise ValueError("acoustic init_artifact must not be empty.")


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


@dataclass
class PerformanceConfig:
    enabled: bool = MISSING
    hardware_peak_flops: Optional[float] = MISSING
    log_every_n_steps: int = MISSING
    warmup_steps: int = MISSING
    measure_window_steps: int = MISSING
    sync_cuda: bool = MISSING
    sync_distributed: bool = MISSING


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


class _EntryConfig(Protocol):
    @property
    def repo_output_root(self) -> str: ...

    @property
    def output_subdir(self) -> str: ...

    @property
    def output_dir(self) -> str: ...

    @property
    def model(self) -> ModelConfig: ...

    @property
    def runtime(self) -> RuntimeConfig: ...

    @property
    def audio_route(self) -> AudioRouteConfig: ...

    @property
    def data(self) -> SpeechConfig: ...

    @property
    def parameter_policy(self) -> ParameterPolicyConfig: ...

    @property
    def pl_module(self) -> ModuleConfig: ...

    @property
    def acoustic(self) -> AcousticConfig: ...

    @property
    def train(self) -> _TrainValues: ...

    @property
    def trainer(self) -> TrainerConfig: ...

    @property
    def callbacks(self) -> _Callbacks: ...


def validate_training(config: _EntryConfig) -> None:
    non_negative_integer(config.train.seed, "train.seed")
    positive_integer(config.train.max_steps, "train.max_steps")
    positive_integer(
        config.trainer.log_every_n_steps,
        "trainer.log_every_n_steps",
    )
    _validate_performance(config.callbacks.performance)
    _validate_output(config)
    _validate_audio_representation(config)
    _validate_audio_route(config)
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


def _validate_output(config: _EntryConfig) -> None:
    subdir = Path(config.output_subdir)
    if subdir == Path(".") or subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError(
            "output_subdir must be a non-empty relative path without '..'."
        )
    expected = Path(config.repo_output_root).expanduser() / subdir
    if Path(config.output_dir).expanduser() != expected:
        raise ValueError("output_dir must equal repo_output_root/output_subdir.")


def _validate_audio_representation(config: _EntryConfig) -> None:
    acoustic = AcousticType(config.acoustic.type)
    if (
        config.runtime.audio_representation
        is AudioRepresentation.FULL_CODEC_SEQUENCE
        and acoustic is not AcousticType.NONE
    ):
        raise ValueError(
            "runtime.audio_representation=full_codec_sequence requires "
            "model/acoustic=none because codec codes are trained as tokens."
        )
    if (
        config.runtime.semantic_codec_artifact is not None
        and acoustic is not AcousticType.NONE
    ):
        raise ValueError(
            "runtime.semantic_codec_artifact requires model/acoustic=none."
        )
    if config.runtime.audio_view is AudioView.BICODEC and acoustic is not AcousticType.NONE:
        raise ValueError(
            "BiCodec fixed-length acoustic units require model/acoustic=none; "
            "use a semantic codec artifact or full_codec_sequence."
        )
    if (
        acoustic is AcousticType.NONE
        and config.runtime.audio_view in {AudioView.LONGCAT, AudioView.BICODEC}
        and config.runtime.audio_representation is AudioRepresentation.DECOUPLED
        and config.runtime.semantic_codec_artifact is None
    ):
        raise ValueError(
            "decoupled model/acoustic=none requires runtime.semantic_codec_artifact; "
            "use a full_codec_sequence runtime for token-only training."
        )


def _validate_audio_route(config: _EntryConfig) -> None:
    validate_audio_route(config.runtime, config.audio_route)
    if config.runtime.audio_view is not AudioView.BICODEC:
        return
    if config.runtime.semantic_codec_artifact is not None:
        raise ValueError(
            "BiCodec audio routes decode structured codes and must not configure "
            "a semantic codec artifact."
        )
    if config.data.dataset.name is not DatasetName.QWEN_TTS_SPEAKER:
        raise ValueError("BiCodec audio routes currently require qwen_tts_speaker data.")


def _validate_backbone_initialization(config: _EntryConfig) -> None:
    if config.runtime.backbone_initialization is not BackboneInitialization.RANDOM:
        return
    if config.model.toy is not None:
        raise ValueError(
            "runtime.backbone_initialization=random cannot be combined with model.toy."
        )
    policy = config.parameter_policy.spec()
    if (
        ParameterGroup.BACKBONE not in policy.trainable_groups
        or (
            policy.backbone_top_fraction is not None
            and policy.backbone_top_fraction < 1
        )
    ):
        raise ValueError(
            "random backbone initialization requires a fully trainable backbone; "
            "select parameter_policy=full."
        )


def _validate_lora(config: _EntryConfig) -> None:
    enabled = config.model.lora is not None
    selected = config.parameter_policy.name is ParameterPolicyName.LORA
    if enabled != selected:
        raise ValueError(
            "model/lora and parameter_policy=lora must be selected together."
        )
    if enabled and config.model.lora is not None and config.model.lora.inference_mode:
        raise ValueError("training requires model.lora.inference_mode=false.")
    if enabled and config.callbacks.performance.enabled:
        raise ValueError(
            "LoRA training FLOPs are not supported by the current performance provider; "
            "set callbacks.performance.enabled=false."
        )
    if enabled and config.pl_module.optimizer == "muon":
        init = config.model.lora.init_lora_weights if config.model.lora is not None else None
        if not isinstance(init, str) or not init.startswith("pissa"):
            raise ValueError(
                "pl_module.optimizer=muon with LoRA requires model.lora.init_lora_weights "
                "to be a pissa initialization (for example 'pissa')."
            )


__all__ = [
    "AcousticNoneConfig",
    "FlowConfig",
    "LoggingConfig",
    "PerformanceConfig",
    "RVQConfig",
    "RepaConfig",
    "TextProbeConfig",
    "TextRetentionCallbackConfig",
    "TrainConfig",
    "TrainerConfig",
    "non_empty_string",
    "non_negative_integer",
    "optional_positive_number",
    "positive_integer",
    "validate_training",
]
