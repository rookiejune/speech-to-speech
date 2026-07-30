from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type, Union

from omegaconf import MISSING, DictConfig

from speech_to_speech.audio_route import Config as AudioRouteConfig
from speech_to_speech.datamodule.config import SpeechConfig
from speech_to_speech.datamodule.joint import LoaderSchedule
from speech_to_speech.datamodule.text import TextConfig as TextDataConfig
from speech_to_speech.model import Config as ModelConfig
from speech_to_speech.model.acoustic import AcousticType
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.stage import ParameterPolicyConfig, StageConfig
from speech_to_speech.task import Task

if __package__:
    from ._config_common import (
        AcousticNoneConfig,
        FlowConfig,
        GradNormCallbackConfig,
        LoggingConfig,
        PerformanceConfig,
        RVQConfig,
        TextProbeConfig,
        TextRetentionCallbackConfig,
        TrainConfig,
        TrainerConfig,
        non_negative_integer,
        optional_positive_number,
        positive_integer,
        validate_training,
    )
    from ._config_normalization import parse, peft_lora, prepare
else:
    from _config_common import (
        AcousticNoneConfig,
        FlowConfig,
        GradNormCallbackConfig,
        LoggingConfig,
        PerformanceConfig,
        RVQConfig,
        TextProbeConfig,
        TextRetentionCallbackConfig,
        TrainConfig,
        TrainerConfig,
        non_negative_integer,
        optional_positive_number,
        positive_integer,
        validate_training,
    )
    from _config_normalization import parse, peft_lora, prepare


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
class TaskSamplePanelConfig:
    split: str = "train"
    loader: str = MISSING
    task: str = MISSING
    indices: list[int] = field(default_factory=list)


@dataclass
class StagedTaskSampleCallbackConfig:
    enabled: bool = False
    every_n_steps: int = 10_000
    panels: list[TaskSamplePanelConfig] = field(default_factory=list)
    seed: int = 0
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False
    use_cache: bool = True


@dataclass
class CheckpointCallbackConfig:
    filename: str = MISSING
    save_last: bool = MISSING
    save_top_k: int = MISSING
    every_n_train_steps: int = MISSING


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


def train(config: DictConfig) -> StagedTrainConfig:
    config = prepare(config)
    lora = peft_lora(config)
    schema: Type[StagedTrainConfig]
    acoustic = AcousticType(str(config.acoustic.type))
    if acoustic is AcousticType.NONE:
        schema = StagedTrainTokenConfig
    elif acoustic is AcousticType.FLOW:
        schema = StagedTrainFlowConfig
    else:
        schema = StagedTrainRVQConfig
    result = parse(config, schema)
    result.model.lora = lora
    validate_training(result)
    if result.callbacks.performance.enabled and result.callbacks.task_sample.enabled:
        raise ValueError(
            "train performance requires callbacks.task_sample.enabled=false "
            "because task sample generation cannot be excluded from distributed "
            "step timing."
        )
    if not result.stage.loaders:
        raise ValueError("formal train requires stage.loaders.")
    _validate_loader_schedule(result)
    _validate_validation(result)
    _validate_task_samples(result)
    _validate_callback_cadences(result.callbacks)
    return result


def _validate_task_samples(config: StagedTrainConfig) -> None:
    callback = config.callbacks.task_sample
    positive_integer(
        callback.every_n_steps,
        "callbacks.task_sample.every_n_steps",
    )
    non_negative_integer(callback.seed, "callbacks.task_sample.seed")
    positive_integer(
        callback.max_new_tokens,
        "callbacks.task_sample.max_new_tokens",
    )
    if not callback.enabled:
        return
    if not callback.panels:
        raise ValueError("enabled staged task samples require panels.")
    seen: set[tuple[str, int]] = set()
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
                f"task sample indices for loader {loader_name!r} must be "
                "non-negative integers."
            )
        for index in panel.indices:
            key = (task.value, index)
            if key in seen:
                raise ValueError(
                    "duplicate task sample tag "
                    f"sample/{task.value}/{index}; split/loader are fetch "
                    "coordinates and must not collide on task+index."
                )
            seen.add(key)
        if panel.split == "validation":
            if loader.is_text:
                raise ValueError(
                    "validation task sample panels require speech loaders."
                )
            if not config.validation.enabled:
                raise ValueError(
                    "validation task sample panels require validation.enabled=true."
                )


def _validate_callback_cadences(config: StagedCallbacksConfig) -> None:
    positive_integer(
        config.grad_norm.every_n_steps,
        "callbacks.grad_norm.every_n_steps",
    )
    positive_integer(
        config.checkpoint.every_n_train_steps,
        "callbacks.checkpoint.every_n_train_steps",
    )


def _validate_loader_schedule(config: StagedTrainConfig) -> None:
    if len(config.stage.loaders) > 1 and config.trainer.strategy in {
        "ddp",
        "ddp_find_unused_parameters_false",
    }:
        raise ValueError(
            "multi-loader gradient accumulation requires DDP unused-parameter "
            "detection; select trainer=staged_ddp instead of a static DDP strategy."
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
        raise ValueError(
            "validation split_label must differ from the train split_label."
        )


__all__ = [
    "CheckpointCallbackConfig",
    "ResumableTrainConfig",
    "StagedCallbacksConfig",
    "StagedTaskSampleCallbackConfig",
    "StagedTrainConfig",
    "StagedTrainFlowConfig",
    "StagedTrainRVQConfig",
    "StagedTrainTokenConfig",
    "TaskSamplePanelConfig",
    "TextProbeConfig",
    "ValidationConfig",
    "train",
]
