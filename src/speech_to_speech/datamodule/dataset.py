from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import torch
from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioView,
    Lang,
    Modality,
    Role,
    Sample,
    TextItem,
    TextMeta,
    TextView,
)
from torch.utils.data import Dataset

from .._compat import StrEnum, auto
from ..runtime.types import Codec, acoustic_codec
from .protocol import DatasetRuntime


class DatasetName(StrEnum):
    QWEN_TTS_BICODEC = auto()
    WMT19_TTS = auto()
    TOY = auto()


@dataclass
class DatasetConfig:
    name: DatasetName = DatasetName.WMT19_TTS
    root: Optional[str] = None
    split: str = "train"
    toy_samples: int = 8
    toy_frames: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.name, DatasetName):
            raise TypeError("dataset name must be a DatasetName.")
        if self.root is not None and not isinstance(self.root, str):
            raise TypeError("dataset root must be a string or None.")
        if not isinstance(self.split, str):
            raise TypeError("dataset split must be a string.")
        if not self.split:
            raise ValueError("dataset split must not be empty.")
        for name, value in (
            ("toy_samples", self.toy_samples),
            ("toy_frames", self.toy_frames),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")


class ToyDataset(Dataset[Sample]):
    """Deterministic in-memory codec samples for model contract tests."""

    def __init__(
        self,
        codec_name: str,
        codec: Codec,
        *,
        samples: int = 8,
        frames: int = 4,
    ) -> None:
        config = DatasetConfig(
            name=DatasetName.TOY,
            toy_samples=samples,
            toy_frames=frames,
        )
        try:
            self.view = AudioView(codec_name)
        except ValueError as error:
            raise ValueError(f"unsupported toy dataset codec: {codec_name}") from error
        self.samples = config.toy_samples
        self.frames = config.toy_frames
        self.frame_rate = codec.frame_rate
        self.codebook_sizes = _codebook_sizes(self.view, codec)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> Sample:
        if index < 0:
            index += self.samples
        if index < 0 or index >= self.samples:
            raise IndexError(index)
        return {
            (Role.SOURCE, Modality.AUDIO): self._audio(index),
            (Role.SOURCE, Modality.TEXT): TextItem(
                views={TextView.TEXT: f"toy source {index}"},
                meta={TextMeta.LANG: Lang.ZH},
            ),
            (Role.TARGET, Modality.AUDIO): self._audio(index + self.samples),
            (Role.TARGET, Modality.TEXT): TextItem(
                views={TextView.TEXT: f"toy target {index}"},
                meta={TextMeta.LANG: Lang.EN},
            ),
        }

    def _audio(self, offset: int) -> AudioItem:
        steps = torch.arange(self.frames, dtype=torch.long)
        columns = [
            (steps + offset + codebook) % size
            for codebook, size in enumerate(self.codebook_sizes)
        ]
        return AudioItem(
            views={self.view: torch.stack(columns, dim=-1)},
            meta={AudioMeta.DURATION: self.frames / self.frame_rate},
        )


def load_dataset(config: DatasetConfig, runtime: DatasetRuntime) -> Dataset[Sample]:
    if config.name is DatasetName.TOY:
        return ToyDataset(
            runtime.codec_name,
            runtime.codec,
            samples=config.toy_samples,
            frames=config.toy_frames,
        )
    if config.name is DatasetName.WMT19_TTS:
        from zhuyin.datasets.wmt19_tts import wmt19_tts_codec

        return cast(
            Dataset[Sample],
            cast(
                object,
                wmt19_tts_codec(
                    codec=runtime.codec_name,
                    root=(
                        None
                        if config.root is None
                        else Path(config.root).expanduser()
                    ),
                    split=config.split,
                ),
            ),
        )
    if config.name is DatasetName.QWEN_TTS_BICODEC:
        from zhuyin.datasets.wmt19_tts import wmt19_qwen_tts_bicodec

        return cast(
            Dataset[Sample],
            cast(
                object,
                wmt19_qwen_tts_bicodec(
                    root=_optional_root(config.root),
                    split=config.split,
                ),
            ),
        )
    raise AssertionError(f"unsupported dataset: {config.name}")


def _optional_root(value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value).expanduser()


def _codebook_sizes(view: AudioView, codec: Codec) -> tuple[int, ...]:
    sizes = tuple(codec.codebook_sizes)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("codec codebook sizes must be positive and non-empty.")
    if view is AudioView.LONGCAT:
        acoustic_sizes = tuple(acoustic_codec(codec).acoustic_codebook_sizes)
        if len(sizes) != len(acoustic_sizes) + 1 or sizes[1:] != acoustic_sizes:
            raise ValueError(
                "LongCat codec codebook sizes must contain one semantic codebook "
                "followed by its acoustic codebooks."
            )
        return sizes
    if view is AudioView.UNICODEC:
        if len(sizes) != 1:
            raise ValueError("UniCodec toy data requires exactly one codebook.")
        return sizes
    if view is AudioView.BICODEC:
        return sizes
    raise ValueError(f"unsupported toy dataset audio view: {view.value}")


__all__ = [
    "DatasetConfig",
    "DatasetName",
    "ToyDataset",
    "load_dataset",
]
