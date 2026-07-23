from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type, TypeVar, Union, cast

from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

from speech_to_speech.codec_oracle import Config as OracleConfig
from speech_to_speech.codec_oracle import Initialization, Objective
from speech_to_speech.datamodule import (
    DatasetConfig,
    DatasetName,
    LBAConfig,
    TextDatasetConfig,
    TextDatasetName,
)
from speech_to_speech.model import AcousticType, AdapterType, DecoderConfig
from speech_to_speech.model import Config as ModelConfig
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.stage import ParameterGroup, StageConfig, StageName


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
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    repa: RepaConfig = field(default_factory=RepaConfig)


@dataclass
class RVQConfig:
    type: str = AcousticType.RVQ.value
    name: str = MISSING
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


@dataclass
class AcousticNoneConfig:
    type: str = AcousticType.NONE.value
    name: str = "token"


@dataclass
class FixedDataConfig(DatasetConfig):
    sample_index: int = MISSING


@dataclass
class TrainConfig:
    seed: int = MISSING
    max_steps: int = MISSING


@dataclass
class TrainDataLoaderConfig:
    batch_size: int = MISSING
    num_workers: int = MISSING
    pin_memory: bool = False
    persistent_workers: bool = False
    lba: LBAConfig = field(default_factory=LBAConfig)


@dataclass
class SpeechDataConfig:
    codec: str = MISSING
    dataloader: TrainDataLoaderConfig = field(default_factory=TrainDataLoaderConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)


@dataclass
class TextDataConfig:
    dataloader: TrainDataLoaderConfig = field(default_factory=TrainDataLoaderConfig)
    dataset: TextDatasetConfig = field(default_factory=TextDatasetConfig)


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


@dataclass
class EvaluationCallbackConfig:
    enabled: bool = True


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
    grad_norm: GradNormCallbackConfig = field(default_factory=GradNormCallbackConfig)
    checkpoint: CheckpointCallbackConfig = field(
        default_factory=CheckpointCallbackConfig
    )
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


@dataclass
class _OverfitConfig:
    task: str = MISSING
    stage: StageConfig = field(default_factory=StageConfig)
    parameter_stage: StageName = MISSING
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
    run_name: str = MISSING
    repo_output_root: str = MISSING
    output_subdir: str = MISSING
    output_dir: str = MISSING
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: SpeechDataConfig = field(default_factory=SpeechDataConfig)
    text_data: TextDataConfig = field(default_factory=TextDataConfig)
    pl_module: ModuleConfig = field(default_factory=ModuleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
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


@dataclass
class OracleCallbackConfig:
    sample_every_n_steps: int = MISSING
    histogram_every_n_steps: int = MISSING
    save_audio: bool = MISSING


@dataclass
class OracleCallbacksConfig:
    oracle: OracleCallbackConfig = field(default_factory=OracleCallbackConfig)
    grad_norm: GradNormCallbackConfig = field(default_factory=GradNormCallbackConfig)
    nonfinite: NonfiniteCallbackConfig = field(default_factory=NonfiniteCallbackConfig)
    checkpoint: CheckpointCallbackConfig = field(
        default_factory=CheckpointCallbackConfig
    )
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


@dataclass
class CodecOracleConfig:
    repo_output_root: str = MISSING
    output_subdir: str = MISSING
    output_dir: str = MISSING
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    codec_oracle: OracleConfig = field(default_factory=OracleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    callbacks: OracleCallbacksConfig = field(default_factory=OracleCallbacksConfig)


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
    if not result.stage.loaders:
        raise ValueError("formal train requires stage.loaders.")
    return result


def codec_oracle(config: DictConfig) -> CodecOracleConfig:
    result = _parse(_prepare(config), CodecOracleConfig)
    _validate_output(result)
    if result.runtime.audio_tokenizer is not None:
        raise ValueError("codec oracle runtime.audio_tokenizer must be null.")
    return result


def _validate_output(
    config: Union[_OverfitConfig, _StagedTrainConfig, CodecOracleConfig],
) -> None:
    subdir = Path(config.output_subdir)
    if subdir == Path(".") or subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError(
            "output_subdir must be a non-empty relative path without '..'."
        )
    expected = Path(config.repo_output_root).expanduser() / subdir
    if Path(config.output_dir).expanduser() != expected:
        raise ValueError("output_dir must equal repo_output_root/output_subdir.")


def _prepare(config: DictConfig) -> DictConfig:
    result = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(config)))
    initialization = result.get("codec_oracle", {}).get("initialization")
    normalized_initialization = None
    if initialization is not None:
        raw = str(initialization)
        normalized_initialization = (
            Initialization[raw]
            if raw in Initialization.__members__
            else Initialization(raw)
        )
        result.codec_oracle.initialization = normalized_initialization.value
    objective = result.get("codec_oracle", {}).get("objective")
    normalized_objective = None
    if objective is not None:
        raw = str(objective)
        normalized_objective = (
            Objective[raw] if raw in Objective.__members__ else Objective(raw)
        )
        result.codec_oracle.objective = normalized_objective.value
    OmegaConf.resolve(result)
    for key in (
        "semantic_audio_adapter",
        "semantic_audio_output_adapter",
        "acoustic_prompt_adapter",
    ):
        value = result.model[key]
        if value is not None:
            raw = str(value)
            result.model[key] = (
                AdapterType[raw].name
                if raw in AdapterType.__members__
                else AdapterType(raw).name
            )
    if normalized_initialization is not None:
        result.codec_oracle.initialization = normalized_initialization.name
    if normalized_objective is not None:
        result.codec_oracle.objective = normalized_objective.name
    _normalize_dataset(result.get("data"))
    _normalize_dataset(result.get("data", {}).get("dataset"))
    _normalize_text_dataset(result.get("text_data", {}).get("dataset"))
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
        for key in ("trainable_groups", "frozen_groups"):
            groups = stage.get(key)
            if groups is None:
                continue
            stage[key] = [
                ParameterGroup[str(group)].name
                if str(group) in ParameterGroup.__members__
                else ParameterGroup(str(group)).name
                for group in groups
            ]
    parameter_stage = result.get("parameter_stage")
    if parameter_stage is not None:
        raw = str(parameter_stage)
        stage_name = StageName[raw] if raw in StageName.__members__ else StageName(raw)
        result.parameter_stage = stage_name.name
        if stage is not None:
            if result.stage.name != stage_name.name:
                raise ValueError(
                    "parameter_stage must match stage.name; select stage "
                    "with the Hydra config group, e.g. stage=stage_1."
                )
    elif stage is not None and "task" in result:
        result.parameter_stage = result.stage.name
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
