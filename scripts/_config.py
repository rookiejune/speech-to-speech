from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type, TypeVar, Union, cast

from anydataset.types import AudioView
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

from speech_to_speech.datamodule.config import SpeechConfig
from speech_to_speech.datamodule.dataset import DatasetConfig, DatasetName
from speech_to_speech.datamodule.text import (
    TextConfig as TextDataConfig,
    TextDatasetName,
)
from speech_to_speech.datamodule.types import DataShape
from speech_to_speech.model import AdapterType
from speech_to_speech.model import Config as ModelConfig
from speech_to_speech.model.acoustic import AcousticType, DecoderConfig
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.runtime import AudioRepresentation, Config as RuntimeConfig
from speech_to_speech.stage import (
    ParameterGroup,
    ParameterPolicyConfig,
    ParameterPolicyName,
    StageConfig,
    StageName,
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
class StagedTaskSampleCallbackConfig:
    enabled: bool = False
    every_n_steps: int = 10_000
    every_audio_seconds: Optional[float] = None
    indices_by_loader: dict[str, list[int]] = field(default_factory=dict)
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False
    use_cache: bool = True


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


ConfigT = TypeVar("ConfigT")


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
    _validate_output(result)
    _validate_audio_representation(result)
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
    _validate_output(result)
    _validate_audio_representation(result)
    if not result.stage.loaders:
        raise ValueError("formal train requires stage.loaders.")
    _validate_task_samples(result)
    _validate_validation(result)
    return result


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
    if not callback.indices_by_loader:
        raise ValueError(
            "enabled staged task samples require indices_by_loader."
        )
    for loader_name, indices in callback.indices_by_loader.items():
        if not isinstance(loader_name, str) or not loader_name:
            raise TypeError("task sample loader names must be non-empty strings.")
        if loader_name not in config.stage.loaders:
            raise ValueError(
                f"task sample callback references unknown loader {loader_name!r}."
            )
        if not indices:
            raise ValueError(
                f"task sample indices for loader {loader_name!r} must not be empty."
            )
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        ):
            raise ValueError(
                f"task sample indices for loader {loader_name!r} must be non-negative integers."
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


def _prepare(config: DictConfig) -> DictConfig:
    result = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(config)))
    OmegaConf.resolve(result)
    for key in (
        "semantic_audio_adapter",
        "semantic_audio_output_adapter",
    ):
        value = result.model[key]
        if value is not None:
            raw = str(value)
            result.model[key] = (
                AdapterType[raw].name
                if raw in AdapterType.__members__
                else AdapterType(raw).name
            )
    _normalize_dataset(result.get("data"))
    _normalize_dataset(result.get("data", {}).get("dataset"))
    _normalize_data_shape(result.get("data"))
    _normalize_text_dataset(result.get("text_data", {}).get("dataset"))
    runtime = result.get("runtime")
    if runtime is not None:
        representation = runtime.get("audio_representation")
        if representation is not None:
            raw = str(representation)
            runtime.audio_representation = (
                AudioRepresentation[raw].name
                if raw in AudioRepresentation.__members__
                else AudioRepresentation(raw).name
            )
    stage = result.get("stage")
    if stage is not None:
        name = stage.get("name")
        if name is not None:
            raw = str(name)
            stage.name = (
                StageName[raw].name
                if raw in StageName.__members__
                else StageName(raw).name
            )
    policy = result.get("parameter_policy")
    if policy is not None:
        name = policy.get("name")
        if name is not None:
            raw = str(name)
            policy.name = (
                ParameterPolicyName[raw].name
                if raw in ParameterPolicyName.__members__
                else ParameterPolicyName(raw).name
            )
        for key in ("trainable_groups", "frozen_groups"):
            groups = policy.get(key)
            if groups is None:
                continue
            policy[key] = [
                ParameterGroup[str(group)].name
                if str(group) in ParameterGroup.__members__
                else ParameterGroup(str(group)).name
                for group in groups
            ]
    return result


def _normalize_dataset(value: object) -> None:
    if not isinstance(value, DictConfig):
        return
    dataset = value.get("name")
    if dataset is None:
        return
    raw = str(dataset)
    value.name = (
        DatasetName[raw].name
        if raw in DatasetName.__members__
        else DatasetName(raw).name
    )


def _normalize_data_shape(value: object) -> None:
    if not isinstance(value, DictConfig):
        return
    shape = value.get("shape")
    if shape is None:
        return
    raw = str(shape)
    value.shape = (
        DataShape[raw].name if raw in DataShape.__members__ else DataShape(raw).name
    )


def _normalize_text_dataset(value: object) -> None:
    if not isinstance(value, DictConfig):
        return
    dataset = value.get("name")
    if dataset is None:
        return
    raw = str(dataset)
    value.name = (
        TextDatasetName[raw].name
        if raw in TextDatasetName.__members__
        else TextDatasetName(raw).name
    )


def _parse(config: DictConfig, schema: Type[ConfigT]) -> ConfigT:
    structured = OmegaConf.structured(schema)
    _writable(structured)
    merged = OmegaConf.merge(structured, config)
    OmegaConf.resolve(merged)
    return cast(ConfigT, OmegaConf.to_object(merged))


def _writable(config: Union[DictConfig, ListConfig]) -> None:
    OmegaConf.set_readonly(config, False)
    nodes = (
        (config._get_node(key) for key in config.keys())
        if isinstance(config, DictConfig)
        else (config._get_node(index) for index in range(len(config)))
    )
    for node in nodes:
        if isinstance(node, (DictConfig, ListConfig)):
            _writable(node)
