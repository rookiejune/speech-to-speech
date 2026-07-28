from __future__ import annotations

from dataclasses import dataclass, field

from .dataset import DatasetConfig, DatasetName
from .types import DataShape


@dataclass
class DataLoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool = False
    persistent_workers: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise TypeError("dataloader batch_size must be an integer.")
        if self.batch_size <= 0:
            raise ValueError("dataloader batch_size must be positive.")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int):
            raise TypeError("dataloader num_workers must be an integer.")
        if self.num_workers < 0:
            raise ValueError("dataloader num_workers must be non-negative.")
        for name in ("pin_memory", "persistent_workers"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"dataloader {name} must be a boolean.")


@dataclass
class SpeechConfig:
    codec: str
    dataloader: DataLoaderConfig
    shape: DataShape = DataShape.PAIR
    encode_missing_codes: bool = False
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.dataloader, DataLoaderConfig):
            raise TypeError("dataloader must be a DataLoaderConfig.")
        if not isinstance(self.shape, DataShape):
            raise TypeError("data shape must be a DataShape.")
        if not isinstance(self.encode_missing_codes, bool):
            raise TypeError("encode_missing_codes must be a boolean.")
        if (
            self.dataset.name is DatasetName.QWEN_TTS_SPEAKER
            and self.shape is not DataShape.SINGLE
        ):
            raise ValueError("qwen_tts_speaker requires data shape single.")


__all__ = ["DataLoaderConfig", "SpeechConfig"]
