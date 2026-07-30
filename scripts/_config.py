from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type, Union

from anydataset.types import AudioView
from omegaconf import MISSING, DictConfig

from speech_to_speech.datamodule.config import SpeechConfig
from speech_to_speech.datamodule.dataset import DatasetConfig, DatasetName
from speech_to_speech.datamodule.joint import LoaderSchedule
from speech_to_speech.datamodule.text import (
    TextConfig as TextDataConfig,
)
from speech_to_speech.datamodule.types import DataShape
from speech_to_speech.audio_route import (
    Config as AudioRouteConfig,
)
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
    StageConfig,
)
from speech_to_speech.task import Task

if __package__:
    from ._config_normalization import parse as _parse
    from ._config_normalization import prepare as _prepare
else:
    from _config_normalization import parse as _parse
    from _config_normalization import prepare as _prepare


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


def _validate_init_artifact(value: Optional[str]) -> None:
    if value is not None and not value:
        raise ValueError("acoustic init_artifact must not be empty.")


@dataclass
class FixedDataConfig(DatasetConfig):
    sample_index: int = MISSING
    shape: DataShape = DataShape.PAIR
    encode_missing_codes: bool = False


@dataclass
class TrainConfig:
    seed: int = MISSING
    max_steps: int = MISSING


@dataclass
class ResumableTrainConfig(TrainConfig):
    ckpt_path: Optional[str] = None


@dataclass
class ValidationConfig:
    enabled: bool = False
    loader: str = "tts"
    split_label: str = "dev"
    every_n_steps: int = 1000
    sanity_steps: int = -1


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
class TaskSampleCallbackConfig:
    enabled: bool = MISSING
    every_n_steps: int = MISSING
    every_audio_seconds: Optional[float] = None


@dataclass
class TaskSamplePanelConfig:
    split: str = "train"
    loader: str = MISSING
    task: str = MISSING
    indices: list[int] = field(default_factory=list)


@dataclass
class StagedTaskSampleCallbackConfig:
    enabled: bool = False
    every_n_steps: int = 10_000
    every_audio_seconds: Optional[float] = None
    panels: list[TaskSamplePanelConfig] = field(default_factory=list)
    seed: int = 0
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False
    use_cache: bool = True


@dataclass
class TextProbeConfig:
    instruction: str = MISSING
    reference: str = MISSING


@dataclass
class TextRetentionCallbackConfig:
    enabled: bool = False
    every_n_steps: int = 10_000
    every_audio_seconds: Optional[float] = None
    max_new_tokens: int = 128
    probes: dict[str, TextProbeConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if (
            isinstance(self.every_n_steps, bool)
            or not isinstance(self.every_n_steps, int)
            or self.every_n_steps <= 0
        ):
            raise ValueError("text retention every_n_steps must be a positive integer.")
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens <= 0
        ):
            raise ValueError("text retention max_new_tokens must be a positive integer.")
        _validate_optional_positive_number(
            self.every_audio_seconds,
            "text retention every_audio_seconds",
        )
        if not self.enabled:
            return
        if not self.probes:
            raise ValueError("enabled text retention requires at least one probe.")
        for name, probe in self.probes.items():
            if not isinstance(name, str) or not name:
                raise TypeError(
                    "text retention probe names must be non-empty strings."
                )
            if not isinstance(probe.instruction, str) or not probe.instruction:
                raise TypeError(
                    f"text retention probe {name!r} instruction must be a non-empty string."
                )
            if not isinstance(probe.reference, str) or not probe.reference:
                raise TypeError(
                    f"text retention probe {name!r} reference must be a non-empty string."
                )


@dataclass
class EvaluationCallbackConfig:
    enabled: bool = True
    every_audio_seconds: Optional[float] = None


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
class GradNormCallbackConfig:
    enabled: bool = MISSING
    every_n_steps: int = MISSING
    every_audio_seconds: Optional[float] = None


@dataclass
class NonfiniteCallbackConfig:
    enabled: bool = MISSING


@dataclass
class CheckpointCallbackConfig:
    filename: str = MISSING
    save_last: bool = MISSING
    save_top_k: int = MISSING
    every_n_train_steps: int = MISSING


@dataclass
class OverfitCallbacksConfig:
    task_sample: TaskSampleCallbackConfig = field(
        default_factory=TaskSampleCallbackConfig
    )
    evaluation: EvaluationCallbackConfig = field(
        default_factory=EvaluationCallbackConfig
    )
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


@dataclass
class StagedCallbacksConfig:
    task_sample: StagedTaskSampleCallbackConfig = field(
        default_factory=StagedTaskSampleCallbackConfig
    )
    text_retention: TextRetentionCallbackConfig = field(
        default_factory=TextRetentionCallbackConfig
    )
    grad_norm: GradNormCallbackConfig = field(default_factory=GradNormCallbackConfig)
    checkpoint: CheckpointCallbackConfig = field(
        default_factory=CheckpointCallbackConfig
    )
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


@dataclass
class _OverfitConfig:
    task: str = MISSING
    stage: StageConfig = field(default_factory=StageConfig)
    parameter_policy: ParameterPolicyConfig = field(
        default_factory=ParameterPolicyConfig
    )
    run_name: str = MISSING
    repo_output_root: str = MISSING
    output_subdir: str = MISSING
    output_dir: str = MISSING
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    audio_route: AudioRouteConfig = MISSING
    data: FixedDataConfig = field(default_factory=FixedDataConfig)
    pl_module: ModuleConfig = field(default_factory=ModuleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    callbacks: OverfitCallbacksConfig = field(default_factory=OverfitCallbacksConfig)


@dataclass
class OverfitTokenConfig(_OverfitConfig):
    acoustic: AcousticNoneConfig = field(default_factory=AcousticNoneConfig)


@dataclass
class OverfitFlowConfig(_OverfitConfig):
    acoustic: FlowConfig = field(default_factory=FlowConfig)


@dataclass
class OverfitRVQConfig(_OverfitConfig):
    acoustic: RVQConfig = field(default_factory=RVQConfig)


OverfitConfig = Union[OverfitTokenConfig, OverfitFlowConfig, OverfitRVQConfig]


@dataclass
class _StagedTrainConfig:
    stage: StageConfig = field(default_factory=StageConfig)
    parameter_policy: ParameterPolicyConfig = field(
        default_factory=ParameterPolicyConfig
    )
    run_name: str = MISSING
    repo_output_root: str = MISSING
    output_subdir: str = MISSING
    output_dir: str = MISSING
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    audio_route: AudioRouteConfig = MISSING
    data: SpeechConfig = MISSING
    text_data: TextDataConfig = MISSING
    pl_module: ModuleConfig = field(default_factory=ModuleConfig)
    train: ResumableTrainConfig = field(default_factory=ResumableTrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    callbacks: StagedCallbacksConfig = field(default_factory=StagedCallbacksConfig)


@dataclass
class StagedTrainTokenConfig(_StagedTrainConfig):
    acoustic: AcousticNoneConfig = field(default_factory=AcousticNoneConfig)


@dataclass
class StagedTrainFlowConfig(_StagedTrainConfig):
    acoustic: FlowConfig = field(default_factory=FlowConfig)


@dataclass
class StagedTrainRVQConfig(_StagedTrainConfig):
    acoustic: RVQConfig = field(default_factory=RVQConfig)


StagedTrainConfig = Union[
    StagedTrainTokenConfig,
    StagedTrainFlowConfig,
    StagedTrainRVQConfig,
]


def overfit(config: DictConfig) -> OverfitConfig:
    config = _prepare(config)
    schema: Type[OverfitConfig]
    acoustic = AcousticType(str(config.acoustic.type))
    if acoustic is AcousticType.NONE:
        schema = OverfitTokenConfig
    elif acoustic is AcousticType.FLOW:
        schema = OverfitFlowConfig
    else:
        schema = OverfitRVQConfig
    result = _parse(config, schema)
    _validate_training(result)
    if (
        result.callbacks.performance.enabled
        and result.callbacks.task_sample.enabled
    ):
        raise ValueError(
            "overfit performance requires callbacks.task_sample.enabled=false "
            "because task sample generation cannot be excluded from distributed "
            "step timing."
        )
    return result


def train(config: DictConfig) -> StagedTrainConfig:
    config = _prepare(config)
    schema: Type[StagedTrainConfig]
    acoustic = AcousticType(str(config.acoustic.type))
    if acoustic is AcousticType.NONE:
        schema = StagedTrainTokenConfig
    elif acoustic is AcousticType.FLOW:
        schema = StagedTrainFlowConfig
    else:
        schema = StagedTrainRVQConfig
    result = _parse(config, schema)
    _validate_training(result)
    if not result.stage.loaders:
        raise ValueError("formal train requires stage.loaders.")
    _validate_loader_schedule(result)
    _validate_validation(result)
    _validate_task_samples(result)
    return result


def _validate_training(config: Union[OverfitConfig, StagedTrainConfig]) -> None:
    _validate_output(config)
    _validate_audio_representation(config)
    _validate_audio_route(config)
    _validate_backbone_initialization(config)
    _validate_lora(config)


def _validate_task_samples(config: StagedTrainConfig) -> None:
    callback = config.callbacks.task_sample
    if (
        isinstance(callback.every_n_steps, bool)
        or not isinstance(callback.every_n_steps, int)
        or callback.every_n_steps <= 0
    ):
        raise ValueError("task sample every_n_steps must be a positive integer.")
    if not callback.enabled:
        return
    if isinstance(callback.seed, bool) or not isinstance(callback.seed, int):
        raise TypeError("task sample seed must be an integer.")
    if callback.seed < 0:
        raise ValueError("task sample seed must be non-negative.")
    if not callback.panels:
        raise ValueError("enabled staged task samples require panels.")
    seen: set[tuple[str, str, str, int]] = set()
    for panel in callback.panels:
        if panel.split not in {"train", "validation"}:
            raise ValueError("task sample split must be 'train' or 'validation'.")
        loader_name = panel.loader
        if not isinstance(loader_name, str) or not loader_name:
            raise TypeError("task sample loader names must be non-empty strings.")
        if loader_name not in config.stage.loaders:
            raise ValueError(
                f"task sample callback references unknown loader {loader_name!r}."
            )
        loader = config.stage.loaders[loader_name]
        text_loader = loader.is_text
        try:
            task = Task(panel.task)
        except ValueError as error:
            raise ValueError(f"unknown task sample task {panel.task!r}.") from error
        if loader.tasks.get(task, 0.0) <= 0:
            raise ValueError(
                f"task sample task {task.value!r} is not active in loader "
                f"{loader_name!r}."
            )
        if not panel.indices:
            raise ValueError(
                f"task sample indices for loader {loader_name!r} must not be empty."
            )
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in panel.indices
        ):
            raise ValueError(
                f"task sample indices for loader {loader_name!r} must be non-negative integers."
            )
        for index in panel.indices:
            key = (panel.split, loader_name, task.value, index)
            if key in seen:
                raise ValueError(f"duplicate task sample panel row: {key!r}.")
            seen.add(key)
        if panel.split == "validation":
            if text_loader:
                raise ValueError(
                    "validation task sample panels require speech loaders."
                )
            if not config.validation.enabled:
                raise ValueError(
                    "validation task sample panels require validation.enabled=true."
                )


def _validate_optional_positive_number(value: object, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None.")
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _validate_loader_schedule(config: StagedTrainConfig) -> None:
    loaders = config.stage.loaders
    if len(loaders) > 1 and config.trainer.strategy in {
        "ddp",
        "ddp_find_unused_parameters_false",
    }:
        raise ValueError(
            "multi-loader gradient accumulation requires DDP unused-parameter "
            "detection; select trainer=ddp instead of a static DDP strategy."
        )
    LoaderSchedule(
        config.stage.loader_weights(),
        accumulate_grad_batches=config.stage.accumulate_grad_batches,
    )


def _validate_validation(config: StagedTrainConfig) -> None:
    validation = config.validation
    if not isinstance(validation.enabled, bool):
        raise TypeError("validation enabled must be a boolean.")
    if not isinstance(validation.loader, str) or not validation.loader:
        raise TypeError("validation loader must be a non-empty string.")
    if not isinstance(validation.split_label, str) or not validation.split_label:
        raise TypeError("validation split_label must be a non-empty string.")
    if (
        isinstance(validation.every_n_steps, bool)
        or not isinstance(validation.every_n_steps, int)
        or validation.every_n_steps <= 0
    ):
        raise TypeError("validation every_n_steps must be a positive integer.")
    if (
        isinstance(validation.sanity_steps, bool)
        or not isinstance(validation.sanity_steps, int)
        or validation.sanity_steps < -1
    ):
        raise TypeError("validation sanity_steps must be -1 or non-negative.")
    if not validation.enabled:
        return
    if validation.loader not in config.stage.loaders:
        raise ValueError(f"unknown validation loader {validation.loader!r}.")
    if config.stage.loaders[validation.loader].is_text:
        raise ValueError("validation loader must be a speech loader.")
    dataset = config.data.dataset
    if dataset.split_manifest is None:
        raise ValueError("enabled validation requires data.dataset.split_manifest.")
    if validation.split_label == dataset.split_label:
        raise ValueError("validation split_label must differ from the train split_label.")


def _validate_output(
    config: Union[_OverfitConfig, _StagedTrainConfig],
) -> None:
    subdir = Path(config.output_subdir)
    if subdir == Path(".") or subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError(
            "output_subdir must be a non-empty relative path without '..'."
        )
    expected = Path(config.repo_output_root).expanduser() / subdir
    if Path(config.output_dir).expanduser() != expected:
        raise ValueError("output_dir must equal repo_output_root/output_subdir.")


def _validate_audio_representation(
    config: Union[OverfitConfig, StagedTrainConfig],
) -> None:
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


def _validate_audio_route(
    config: Union[OverfitConfig, StagedTrainConfig],
) -> None:
    route = config.audio_route
    validate_audio_route(config.runtime, route)
    if config.runtime.audio_view is AudioView.BICODEC:
        if config.runtime.semantic_codec_artifact is not None:
            raise ValueError(
                "BiCodec audio routes decode structured codes and must not configure "
                "a semantic codec artifact."
            )
        dataset = (
            config.data
            if isinstance(config, _OverfitConfig)
            else config.data.dataset
        )
        if dataset.name is not DatasetName.QWEN_TTS_SPEAKER:
            raise ValueError(
                "BiCodec audio routes currently require qwen_tts_speaker data."
            )
        return


def _validate_backbone_initialization(
    config: Union[OverfitConfig, StagedTrainConfig],
) -> None:
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


def _validate_lora(config: Union[OverfitConfig, StagedTrainConfig]) -> None:
    enabled = config.model.lora.enabled
    selected = config.parameter_policy.name is ParameterPolicyName.LORA
    if enabled != selected:
        raise ValueError(
            "model.lora.enabled and parameter_policy=lora must be selected together."
        )
    if enabled and config.callbacks.performance.enabled:
        raise ValueError(
            "LoRA training FLOPs are not supported by the current performance provider; "
            "set callbacks.performance.enabled=false."
        )
