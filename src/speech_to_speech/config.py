from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .types import Task, TaskFamily


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


@dataclass(frozen=True)
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    trust_remote_code: bool = False
    load_in_4bit: bool = True
    train_text_embedding: bool = False
    train_audio_embedding: bool = True
    train_audio_special_tokens: bool = True
    train_backbone: bool = False
    train_dit: bool = True
    acoustic_condition_dropout: float = 0.0
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 5_000_000
    learning_rate: float = 1e-4
    adamw_learning_rate: float | None = None
    muon_learning_rate: float | None = None
    acoustic_loss_weight: float = 0.0
    optimizer_preset: str = "pretrain"
    optimizer: str = "muon"
    weight_decay: float | None = None
    schedule: str = "warmup_cosine"
    warmup_steps: int = 50_000
    stable_steps: int | None = None
    decay_steps: int | None = None
    min_lr_ratio: float = 0.1
    seed: int = 0
    device: str = "auto"
    precision: str = "bf16-mixed"


@dataclass(frozen=True)
class TrainerConfig:
    name: str = "default"
    default_root_dir: str | Path | None = None
    accelerator: str | None = None
    devices: int | str = 1
    strategy: str = "auto"
    ckpt_path: str | Path | None = None
    log_every_n_steps: int | None = None
    checkpoint_every_n_steps: int = 10_000
    sample_log_every_n_steps: int | None = 0
    samples_per_task: int = 1
    sample_log_max_audio_samples: int | None = 320_000
    generation_log_every_n_steps: int | None = 5_000
    generation_sample_index: int = 0
    generation_flow_steps: int = 32
    generation_chunk_size: int | None = 64
    generation_guidance_scale: float = 1.0
    generation_acoustic_sampler: str = "diagonal"
    generation_preview_tokens: int = 1024
    generation_log_max_audio_samples: int | None = 320_000
    save_top_k: int = 2
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
