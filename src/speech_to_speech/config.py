from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum, auto
from pathlib import Path

from .types.datamodule import Task, TaskFamily


@dataclass(frozen=True)
class DatasetFactoryConfig:
    name: str = "wmt19_tts_longcat"


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = 1
    num_workers: int = 8
    pin_memory: bool = False
    drop_last: bool = False


@dataclass(frozen=True)
class LBAConfig:
    enabled: bool = False
    log_dir: str | Path | None = None


@dataclass(frozen=True)
class DataModuleConfig:
    dataset_factory: DatasetFactoryConfig = field(default_factory=DatasetFactoryConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    lba: LBAConfig = field(default_factory=LBAConfig)


@dataclass(frozen=True)
class BPEConfig:
    cache_dir_env: str = "BPE_CACHE_DIR"
    codec_name: str = "longcat"
    vocab_size: int = 10_000
    min_frequency: int = 0
    max_token_length: int | None = None
    codebook_sizes: tuple[int, ...] = (8192,)

    @property
    def artifact_name(self) -> str:
        vocab = (
            f"{self.vocab_size // 1000}k" if self.vocab_size % 1000 == 0 else str(self.vocab_size)
        )
        codebooks = "x".join(str(size) for size in self.codebook_sizes)
        maxlen = "none" if self.max_token_length is None else str(self.max_token_length)
        return f"vocab_{vocab}_minfreq_{self.min_frequency}_maxlen_{maxlen}_codes_{codebooks}"

    def artifact_path(self, cache_dir: str | Path) -> Path:
        return Path(cache_dir) / self.codec_name / self.artifact_name


@dataclass(frozen=True)
class TaskWeightsConfig:
    source_ar: float = 1.0
    target_ar: float = 1.0
    source_to_target: float = 1.0
    target_to_source: float = 1.0

    def weight(self, family: TaskFamily) -> float:
        match family:
            case TaskFamily.SOURCE_AR:
                return self.source_ar
            case TaskFamily.TARGET_AR:
                return self.target_ar
            case TaskFamily.SOURCE_TO_TARGET:
                return self.source_to_target
            case TaskFamily.TARGET_TO_SOURCE:
                return self.target_to_source


@dataclass(frozen=True)
class TaskConfig:
    enabled: tuple[str, ...] = field(
        default_factory=lambda: (
            Task.AUTOREGRESSION.value,
            Task.TRANSLATION.value,
        )
    )
    weights: TaskWeightsConfig = field(default_factory=TaskWeightsConfig)


@dataclass(frozen=True)
class LoRAConfig:
    enabled: bool = True
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


class AcousticAttentionMode(StrEnum):
    CAUSAL = auto()
    BIDIRECTIONAL = auto()


class AcousticConditionSource(StrEnum):
    QWEN_HIDDEN = auto()
    TARGET_AUDIO_EMBEDDING = auto()


class AudioEmbeddingType(StrEnum):
    LOOKUP = auto()
    SEMANTIC_COMPOSITION = auto()

    @property
    def requires_bpe(self) -> bool:
        return self is AudioEmbeddingType.SEMANTIC_COMPOSITION


class ModelTrainMode(StrEnum):
    DEFAULT = auto()
    ACOUSTIC_ONLY = auto()


class AdapterType(StrEnum):
    IDENTITY = auto()
    LINEAR = auto()
    QWEN_MLP = auto()


@dataclass(frozen=True)
class AdapterConfig:
    type: AdapterType = AdapterType.IDENTITY
    in_features: int | None = None
    out_features: int | None = None


@dataclass(frozen=True)
class DiTModelConfig:
    hidden_size: int = 1024
    num_hidden_layers: int = 8
    intermediate_size: int = 3072
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    norm_time: bool = False
    norm_hidden: bool = False
    norm_acoustic: bool = False


@dataclass(frozen=True)
class ConditionEncoderConfig:
    enabled: bool = False
    num_hidden_layers: int = 1
    intermediate_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None


@dataclass(frozen=True)
class QwenBackboneConfig:
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    trust_remote_code: bool = False
    load_in_4bit: bool = True
    train: bool = False
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass(frozen=True)
class TokenSpaceConfig:
    train_text_embedding: bool = False
    train_audio_embedding: bool = True
    train_audio_special_tokens: bool = True
    input_adapter: AdapterConfig = field(default_factory=AdapterConfig)
    output_adapter: AdapterConfig = field(default_factory=AdapterConfig)
    audio_embedding_type: AudioEmbeddingType = AudioEmbeddingType.LOOKUP
    semantic_codebook_size: int = 8192
    semantic_rope_base: float = 10_000.0
    semantic_shift_rank: int = 16


@dataclass(frozen=True)
class AcousticDecoderConfig:
    enabled: bool = False
    train: bool = True
    attention_mode: AcousticAttentionMode = AcousticAttentionMode.CAUSAL
    condition_dropout: float = 0.0
    condition_source: AcousticConditionSource = AcousticConditionSource.QWEN_HIDDEN
    condition_adapter: AdapterConfig = field(default_factory=AdapterConfig)
    condition_encoder: ConditionEncoderConfig = field(default_factory=ConditionEncoderConfig)
    dit: DiTModelConfig = field(default_factory=DiTModelConfig)


@dataclass(frozen=True)
class ModelConfig:
    train_mode: ModelTrainMode = ModelTrainMode.DEFAULT
    backbone: QwenBackboneConfig = field(default_factory=QwenBackboneConfig)
    token_space: TokenSpaceConfig = field(default_factory=TokenSpaceConfig)
    acoustic: AcousticDecoderConfig = field(default_factory=AcousticDecoderConfig)


def with_acoustic_decoder(
    config: ModelConfig,
    *,
    enabled: bool = True,
    train: bool | None = None,
    dit: DiTModelConfig | None = None,
) -> ModelConfig:
    acoustic = config.acoustic
    if train is not None:
        acoustic = replace(acoustic, train=train)
    if dit is not None:
        acoustic = replace(acoustic, dit=dit)
    acoustic = replace(acoustic, enabled=enabled)
    return replace(config, acoustic=acoustic)


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 5_000_000
    learning_rate: float = 1e-4
    adamw_learning_rate: float | None = None
    muon_learning_rate: float | None = None
    semantic_loss_weight: float = 1.0
    stop_loss_weight: float = 1.0
    acoustic_loss_weight: float = 0.0
    optimizer_preset: str = "pretrain"
    optimizer: str = "muon"
    weight_decay: float | None = None
    schedule: str = "warmup_cosine"
    warmup_ratio: float = 0.01
    stable_steps: int | None = None
    decay_steps: int | None = None
    min_lr_ratio: float = 0.1
    seed: int = 0
    device: str = "auto"
    precision: str = "bf16-mixed"


@dataclass(frozen=True)
class CheckpointCallbackConfig:
    enabled: bool = True
    every_n_steps: int = 10_000
    save_top_k: int = 2
    monitor: str = "loss"
    mode: str = "min"
    save_last: bool = True
    filename: str = "{step:08d}"


@dataclass(frozen=True)
class LearningRateMonitorCallbackConfig:
    enabled: bool = True
    logging_interval: str | None = "step"


@dataclass(frozen=True)
class GenerationCallbackConfig:
    enabled: bool = True
    every_n_steps: int | None = 5_000
    sample_index: int = 0
    flow_steps: int = 32
    chunk_size: int | None = None
    left_context_chunks: int | None = None
    guidance_scale: float = 1.0
    acoustic_sampler: str = "serial"
    preview_tokens: int = 1024
    max_audio_samples: int | None = 320_000


@dataclass(frozen=True)
class TrainerCallbacksConfig:
    checkpoint: CheckpointCallbackConfig = field(default_factory=CheckpointCallbackConfig)
    learning_rate_monitor: LearningRateMonitorCallbackConfig = field(
        default_factory=LearningRateMonitorCallbackConfig
    )
    generation: GenerationCallbackConfig = field(default_factory=GenerationCallbackConfig)


@dataclass(frozen=True)
class TrainerConfig:
    name: str = "default"
    default_root_dir: str | Path | None = None
    accelerator: str | None = None
    devices: int | str = 1
    strategy: str = "auto"
    gradient_clip_val: float | None = 1.0
    gradient_clip_algorithm: str | None = "norm"
    ckpt_path: str | Path | None = None
    log_every_n_steps: int | None = None
    callbacks: TrainerCallbacksConfig = field(default_factory=TrainerCallbacksConfig)
    enable_model_summary: bool = True
    enable_progress_bar: bool = True


@dataclass(frozen=True)
class SpeechToSpeechConfig:
    datamodule: DataModuleConfig = field(default_factory=DataModuleConfig)
    bpe: BPEConfig = field(default_factory=BPEConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
