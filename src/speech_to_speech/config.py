from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

from anydataset import Preset, Spec

from .types import Task

DatasetInput: TypeAlias = str | Preset | Spec


@dataclass(frozen=True)
class DataConfig:
    datasets: tuple[DatasetInput, ...]
    cache_root: str | Path | None = None
    lba_log_dir: str | Path | None = None
    batch_size: int = 1
    num_workers: int = 8
    pin_memory: bool = False
    drop_last: bool = False


@dataclass(frozen=True)
class BPEConfig:
    cache_dir_env: str = "BPE_CACHE_DIR"
    codec_name: str = "longcat"
    vocab_size: int = 100_000
    max_piece_frames: int = 32

    @property
    def artifact_name(self) -> str:
        vocab = (
            f"{self.vocab_size // 1000}k" if self.vocab_size % 1000 == 0 else str(self.vocab_size)
        )
        return f"vocab_{vocab}_piece_{self.max_piece_frames}"

    def artifact_path(self, cache_dir: str | Path) -> Path:
        return Path(cache_dir) / self.codec_name / self.artifact_name


@dataclass(frozen=True)
class TaskConfig:
    enabled: tuple[str, ...] = field(
        default_factory=lambda: (
            Task.AUTOREGRESSION.value,
            Task.TRANSLATION.value,
        )
    )


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
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass(frozen=True)
class TrainConfig:
    max_steps: int = 100
    learning_rate: float = 1e-4
    optimizer_preset: str = "pretrain"
    optimizer: str = "muon"
    weight_decay: float | None = None
    schedule: str = "constant"
    warmup_steps: int = 0
    stable_steps: int | None = None
    decay_steps: int | None = None
    min_lr_ratio: float = 0.1
    seed: int = 0
    device: str = "cuda"
    precision: str = "bf16-mixed"


@dataclass(frozen=True)
class SpeechToSpeechConfig:
    data: DataConfig
    bpe: BPEConfig = field(default_factory=BPEConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
