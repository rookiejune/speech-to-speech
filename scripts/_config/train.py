from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, Union

from omegaconf import MISSING, DictConfig

from speech_to_speech.datamodule.config import SpeechConfig
from speech_to_speech.datamodule.loader import (
    LoaderPlanConfig,
    LoaderSchedule,
    LoaderStepMode,
)
from speech_to_speech.datamodule.dataset.text import (
    TextConfig as TextDataConfig,
    TextDatasetName,
)
from speech_to_speech.model.acoustic import AcousticType
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.runtime import AudioSequenceLayout, Config as RuntimeConfig
from speech_to_speech.training.parameter_policy import ParameterPolicyConfig
from speech_to_speech.task import PredictionModality, Task

from speech_to_speech.training.config import (
    FlowModelConfig,
    GradientComparisonConfig,
    GradientProbeConfig,
    LoggingConfig,
    OptimConfig,
    PerformanceConfig,
    RVQModelConfig,
    TextProbeConfig,
    TextRetentionCallbackConfig,
    TokenModelConfig,
    TrainConfig,
    TrainerConfig,
    non_negative_integer,
    positive_integer,
    validate_gradient_comparisons,
    validate_gradient_probes,
    validate_training,
)
from .normalization import parse, peft_lora, prepare


@dataclass
class ResumableTrainConfig(TrainConfig):
    ckpt_path: Optional[str] = None
    auto_resume: bool = False


@dataclass
class ValidationConfig:
    enabled: bool = False
    loader: str = "tts"
    split_label: str = "dev"
    text_split: str = "validation"
    max_samples: int = 1000
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
class SynthesisSampleCallbackConfig:
    enabled: bool = False
    every_n_steps: int = 100
    loader: str = "s2st"
    indices: list[int] = field(default_factory=list)


@dataclass
class CheckpointCallbackConfig:
    filename: str = MISSING
    save_last: bool = MISSING
    save_top_k: int = MISSING
    every_n_train_steps: int = MISSING


@dataclass
class GradientProbeCallbackConfig:
    enabled: bool = False
    every_n_steps: int = 10_000
    probes: dict[str, GradientProbeConfig] = field(default_factory=dict)
    comparisons: list[GradientComparisonConfig] = field(default_factory=list)


@dataclass
class StagedCallbacksConfig:
    parameter_policy: ParameterPolicyConfig = field(
        default_factory=ParameterPolicyConfig
    )
    task_sample: StagedTaskSampleCallbackConfig = field(
        default_factory=StagedTaskSampleCallbackConfig
    )
    synthesis_sample: SynthesisSampleCallbackConfig = field(
        default_factory=SynthesisSampleCallbackConfig
    )
    text_retention: TextRetentionCallbackConfig = field(
        default_factory=TextRetentionCallbackConfig
    )
    gradient_probe: GradientProbeCallbackConfig = field(
        default_factory=GradientProbeCallbackConfig
    )
    checkpoint: CheckpointCallbackConfig = field(
        default_factory=CheckpointCallbackConfig
    )
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


@dataclass
class _StagedTrainConfig:
    run_name: str = MISSING
    repo_output_root: str = MISSING
    output_subdir: str = MISSING
    output_dir: str = MISSING
    loader_plan: LoaderPlanConfig = field(default_factory=LoaderPlanConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    audio_sequence_layout: AudioSequenceLayout = MISSING
    datamodule: SpeechConfig = MISSING
    text_datamodule: TextDataConfig = MISSING
    pl_module: ModuleConfig = field(default_factory=ModuleConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: ResumableTrainConfig = field(default_factory=ResumableTrainConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    callbacks: StagedCallbacksConfig = field(default_factory=StagedCallbacksConfig)


@dataclass
class StagedTrainTokenConfig(_StagedTrainConfig):
    model: TokenModelConfig = field(default_factory=TokenModelConfig)


@dataclass
class StagedTrainFlowConfig(_StagedTrainConfig):
    model: FlowModelConfig = field(default_factory=FlowModelConfig)


@dataclass
class StagedTrainRVQConfig(_StagedTrainConfig):
    model: RVQModelConfig = field(default_factory=RVQModelConfig)


StagedTrainConfig = Union[
    StagedTrainTokenConfig,
    StagedTrainFlowConfig,
    StagedTrainRVQConfig,
]


def train(config: DictConfig) -> StagedTrainConfig:
    config = prepare(config)
    lora = peft_lora(config)
    schema: Type[StagedTrainConfig]
    acoustic = AcousticType(str(config.model.acoustic.type))
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
    if not result.loader_plan.loaders:
        raise ValueError("formal train requires loader_plan.loaders.")
    _validate_loader_schedule(result)
    _validate_validation(result)
    _validate_task_samples(result)
    _validate_synthesis_samples(result)
    _validate_streaming(result)
    _validate_callback_cadences(result.callbacks)
    _validate_gradient_probe(result)
    return result


def _validate_streaming(config: StagedTrainConfig) -> None:
    if not isinstance(config.train.auto_resume, bool):
        raise TypeError("train.auto_resume must be a boolean.")
    if not config.datamodule.streaming.enabled:
        return
    if config.trainer.max_epochs != 1:
        raise ValueError(
            "streaming synthesis requires trainer.max_epochs=1; the one epoch "
            "continues across checkpoints until the dataset is sealed."
        )
    if config.trainer.use_distributed_sampler:
        raise ValueError(
            "streaming synthesis requires trainer.use_distributed_sampler=false."
        )
    if not config.trainer.enable_checkpointing:
        raise ValueError("streaming synthesis requires checkpointing for resume.")
    if not config.callbacks.checkpoint.save_last:
        raise ValueError("streaming synthesis requires checkpoint.save_last=true.")
    if not config.train.auto_resume and config.train.ckpt_path is None:
        raise ValueError(
            "streaming synthesis requires train.auto_resume=true or an explicit "
            "train.ckpt_path."
        )
    if len(config.loader_plan.loaders) != 1:
        raise ValueError("streaming synthesis requires exactly one training loader.")
    loader = next(iter(config.loader_plan.loaders.values()))
    if loader.is_text:
        raise ValueError("streaming synthesis requires one speech loader.")
    active_tasks = {task for task, weight in loader.tasks.items() if weight > 0}
    if active_tasks != {Task.S2ST}:
        raise ValueError(
            "streaming synthesis requires exactly the s2st task so every one of "
            "the 2N teacher samples contributes one direct translation example."
        )
    if loader.prediction_modality is not PredictionModality.PARALLEL:
        raise ValueError(
            "streaming s2st requires prediction=parallel so the teacher-generated "
            "target text and target audio are both backbone labels."
        )
    if config.loader_plan.accumulate_grad_batches != 1:
        raise ValueError(
            "streaming synthesis initially requires accumulate_grad_batches=1 "
            "for an unambiguous optimizer-boundary cursor."
        )
    if config.validation.enabled:
        raise ValueError(
            "streaming synthesis validation must run from a sealed immutable dataset."
        )
    if config.callbacks.task_sample.enabled:
        raise ValueError(
            "streaming synthesis uses callbacks.synthesis_sample for artifacts; "
            "task_sample requires fixed samples available at fit start."
        )


def _validate_synthesis_samples(config: StagedTrainConfig) -> None:
    callback = config.callbacks.synthesis_sample
    positive_integer(
        callback.every_n_steps,
        "callbacks.synthesis_sample.every_n_steps",
    )
    if not isinstance(callback.loader, str) or not callback.loader:
        raise TypeError("callbacks.synthesis_sample.loader must be a non-empty string.")
    if not callback.enabled:
        return
    if not config.datamodule.streaming.enabled:
        raise ValueError(
            "callbacks.synthesis_sample requires datamodule.streaming.enabled=true."
        )
    if callback.loader not in config.loader_plan.loaders:
        raise ValueError(
            "callbacks.synthesis_sample references unknown loader "
            f"{callback.loader!r}."
        )
    if not callback.indices:
        raise ValueError("enabled synthesis sample logging requires indices.")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in callback.indices
    ):
        raise ValueError(
            "callbacks.synthesis_sample.indices must be non-negative integers."
        )
    if config.callbacks.performance.enabled:
        raise ValueError(
            "train performance requires callbacks.synthesis_sample.enabled=false."
        )


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
        if loader_name not in config.loader_plan.loaders:
            raise ValueError(
                f"task sample callback references unknown loader {loader_name!r}."
            )
        loader = config.loader_plan.loaders[loader_name]
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
        config.gradient_probe.every_n_steps,
        "callbacks.gradient_probe.every_n_steps",
    )
    positive_integer(
        config.checkpoint.every_n_train_steps,
        "callbacks.checkpoint.every_n_train_steps",
    )


def _validate_gradient_probe(config: StagedTrainConfig) -> None:
    callback = config.callbacks.gradient_probe
    validate_gradient_probes(
        callback.probes,
        "callbacks.gradient_probe.probes",
    )
    validate_gradient_comparisons(
        callback.comparisons,
        "callbacks.gradient_probe.comparisons",
    )
    if not callback.enabled:
        return
    if not callback.probes:
        raise ValueError("enabled gradient probe requires at least one probe.")
    if not callback.comparisons:
        raise ValueError("enabled gradient probe requires at least one comparison.")
    allowed_sources = {"batch", *config.loader_plan.loaders}
    for comparison in callback.comparisons:
        for target in (comparison.left, comparison.right):
            if target.group not in allowed_sources:
                raise ValueError(
                    f"gradient comparison references unknown group {target.group!r}."
                )
            if target.group != "batch" and config.loader_plan.mode is not LoaderStepMode.FUSED_JOINT:
                raise ValueError(
                    "gradient non-batch comparisons require "
                    "loader_plan.step_mode=fused_joint."
                )


def _validate_loader_schedule(config: StagedTrainConfig) -> None:
    if len(config.loader_plan.loaders) > 1 and config.trainer.use_distributed_sampler:
        raise ValueError(
            "multi-loader staged training requires trainer.use_distributed_sampler=false; "
            "select trainer=staged_static_ddp or trainer=staged_ddp."
        )
    if config.loader_plan.mode is LoaderStepMode.SERIAL_JOINT and not _uses_unused_parameter_detection(config):
        raise ValueError(
            "loader_plan.step_mode=serial_joint requires DDP unused-parameter "
            "detection; select trainer=staged_ddp / "
            "ddp_find_unused_parameters_true."
        )
    LoaderSchedule(
        config.loader_plan.loader_weights(),
        accumulate_grad_batches=config.loader_plan.accumulate_grad_batches,
        fuse_loaders_per_step=config.loader_plan.fuse_loaders_per_step,
        step_mode=config.loader_plan.step_mode,
    )
    required: set[Task] = set()
    for loader in config.loader_plan.loaders.values():
        required.update(task for task, weight in loader.tasks.items() if weight > 0)
    configured = config.datamodule.tasks
    missing = (
        []
        if configured is None
        else sorted(task.value for task in required if task not in configured)
    )
    if missing:
        raise KeyError(
            "datamodule.tasks must declare every positive-weight loader_plan task; "
            "missing: " + ", ".join(missing)
        )


def _uses_unused_parameter_detection(config: StagedTrainConfig) -> bool:
    return config.trainer.strategy == "ddp_find_unused_parameters_true"


def _validate_validation(config: StagedTrainConfig) -> None:
    validation = config.validation
    if not isinstance(validation.enabled, bool):
        raise TypeError("validation enabled must be a boolean.")
    if not isinstance(validation.loader, str) or not validation.loader:
        raise TypeError("validation loader must be a non-empty string.")
    if not isinstance(validation.split_label, str) or not validation.split_label:
        raise TypeError("validation split_label must be a non-empty string.")
    if not isinstance(validation.text_split, str) or not validation.text_split:
        raise TypeError("validation text_split must be a non-empty string.")
    positive_integer(validation.max_samples, "validation.max_samples")
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
    if validation.loader not in config.loader_plan.loaders:
        raise ValueError(f"unknown validation loader {validation.loader!r}.")
    loader = config.loader_plan.loaders[validation.loader]
    if loader.is_text:
        if loader.tasks != {Task.MT: 1.0}:
            raise ValueError("text validation loader must contain only MT.")
        if config.text_datamodule.dataset.name is not TextDatasetName.WMT19:
            raise ValueError("MT text validation requires the WMT19 dataset.")
        return
    dataset = config.datamodule.dataset
    if dataset.split_manifest is None:
        raise ValueError(
            "enabled validation requires datamodule.dataset.split_manifest."
        )
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
