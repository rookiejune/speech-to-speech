from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar, Union, cast

from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

from speech_to_speech.codec_oracle import Config as OracleConfig
from speech_to_speech.model import AcousticType, AdapterType, DecoderConfig
from speech_to_speech.model import Config as ModelConfig
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.runtime import Config as RuntimeConfig


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
class FixedDataConfig:
    root: Optional[str] = None
    split: str = MISSING
    sample_index: int = MISSING


@dataclass
class TrainConfig:
    seed: int = MISSING
    max_steps: int = MISSING


@dataclass
class TrainerConfig:
    accelerator: str = MISSING
    devices: int = MISSING
    strategy: str = MISSING
    expected_world_size: int = MISSING
    use_distributed_sampler: bool = MISSING
    precision: str = MISSING
    max_epochs: int = MISSING
    log_every_n_steps: int = MISSING
    enable_checkpointing: bool = MISSING
    gradient_clip_val: float = MISSING


@dataclass
class LoggingConfig:
    name: str = MISSING


@dataclass
class SampleCallbackConfig:
    enabled: bool = MISSING
    every_n_steps: int = MISSING


@dataclass
class OverfitCallbacksConfig:
    sample: SampleCallbackConfig = field(default_factory=SampleCallbackConfig)


@dataclass
class _OverfitConfig:
    task: str = MISSING
    run_name: str = MISSING
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
    pass


@dataclass
class OverfitFlowConfig(_OverfitConfig):
    acoustic: FlowConfig = field(default_factory=FlowConfig)


@dataclass
class OverfitRVQConfig(_OverfitConfig):
    acoustic: RVQConfig = field(default_factory=RVQConfig)


OverfitConfig = Union[OverfitTokenConfig, OverfitFlowConfig, OverfitRVQConfig]


@dataclass
class OracleCallbackConfig:
    sample_every_n_steps: int = MISSING
    histogram_every_n_steps: int = MISSING
    save_audio: bool = MISSING


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
class OracleCallbacksConfig:
    oracle: OracleCallbackConfig = field(default_factory=OracleCallbackConfig)
    grad_norm: GradNormCallbackConfig = field(default_factory=GradNormCallbackConfig)
    nonfinite: NonfiniteCallbackConfig = field(default_factory=NonfiniteCallbackConfig)
    checkpoint: CheckpointCallbackConfig = field(
        default_factory=CheckpointCallbackConfig
    )


@dataclass
class CodecOracleConfig:
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
    if "acoustic" not in config:
        schema = OverfitTokenConfig
    else:
        acoustic = AcousticType(str(config.acoustic.type))
        if acoustic is AcousticType.FLOW:
            schema = OverfitFlowConfig
        else:
            schema = OverfitRVQConfig
    return _parse(config, schema)


def codec_oracle(config: DictConfig) -> CodecOracleConfig:
    result = _parse(_prepare(config), CodecOracleConfig)
    if result.runtime.audio_tokenizer is not None:
        raise ValueError("codec oracle runtime.audio_tokenizer must be null.")
    return result


def _prepare(config: DictConfig) -> DictConfig:
    result = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(config)))
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
    initialization = result.get("codec_oracle", {}).get("initialization")
    if initialization is not None:
        result.codec_oracle.initialization = str(initialization).upper()
    return result


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
