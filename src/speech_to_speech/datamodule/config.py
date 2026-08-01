from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .dataset.speech import DatasetConfig, DatasetName
from .types import DataShape


@dataclass
class DataLoaderCostsConfig:
    enabled: bool = False
    max_batch_frames: Optional[int] = None
    planning_window: int = 256

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("dataloader costs enabled must be a boolean.")
        if self.max_batch_frames is not None:
            if (
                isinstance(self.max_batch_frames, bool)
                or not isinstance(self.max_batch_frames, int)
            ):
                raise TypeError("dataloader costs max_batch_frames must be an integer or None.")
            if self.max_batch_frames <= 0:
                raise ValueError("dataloader costs max_batch_frames must be positive.")
        if isinstance(self.planning_window, bool) or not isinstance(
            self.planning_window,
            int,
        ):
            raise TypeError("dataloader costs planning_window must be an integer.")
        if self.planning_window <= 0:
            raise ValueError("dataloader costs planning_window must be positive.")
        if self.enabled and self.max_batch_frames is None:
            raise ValueError(
                "enabled dataloader costs require max_batch_frames.",
            )


@dataclass
class DataLoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool = False
    persistent_workers: bool = False
    costs: DataLoaderCostsConfig = field(default_factory=DataLoaderCostsConfig)

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
        if not isinstance(self.costs, DataLoaderCostsConfig):
            raise TypeError("dataloader costs must be a DataLoaderCostsConfig.")


@dataclass
class SpeechConfig:
    codec: str
    dataloader: DataLoaderConfig
    shape: DataShape = DataShape.PAIR
    encode_missing_codes: bool = False
    interleave_audio_frames: int = 25
    mask_text_ratio: float = 0.5
    mask_audio_ratio: float = 0.5
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.dataloader, DataLoaderConfig):
            raise TypeError("dataloader must be a DataLoaderConfig.")
        if not isinstance(self.shape, DataShape):
            raise TypeError("data shape must be a DataShape.")
        if not isinstance(self.encode_missing_codes, bool):
            raise TypeError("encode_missing_codes must be a boolean.")
        if (
            isinstance(self.interleave_audio_frames, bool)
            or not isinstance(self.interleave_audio_frames, int)
            or self.interleave_audio_frames < 1
        ):
            raise ValueError("interleave_audio_frames must be a positive integer.")
        for name in ("mask_text_ratio", "mask_audio_ratio"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise TypeError(f"{name} must be a float.")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if (
            self.dataset.name is DatasetName.QWEN_TTS_SPEAKER
            and self.shape is not DataShape.SINGLE
        ):
            raise ValueError("qwen_tts_speaker requires data shape single.")


__all__ = ["DataLoaderConfig", "DataLoaderCostsConfig", "SpeechConfig"]
