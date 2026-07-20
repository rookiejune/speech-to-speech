from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import torch
from anydataset.types import (
    AudioItem,
    AudioView,
    Modality,
    Role,
    Sample,
    TextItem,
    TextMeta,
    TextView,
)
from torch import Tensor
from torch.utils.data import Dataset

from .._compat import StrEnum, auto
from ..runtime.types import Codec
from .protocol import DatasetRuntime


class DatasetName(StrEnum):
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
                meta={TextMeta.LANG: "zh"},
            ),
            (Role.TARGET, Modality.AUDIO): self._audio(index + self.samples),
            (Role.TARGET, Modality.TEXT): TextItem(
                views={TextView.TEXT: f"toy target {index}"},
                meta={TextMeta.LANG: "en"},
            ),
        }

    def _audio(self, offset: int) -> AudioItem:
        steps = torch.arange(self.frames, dtype=torch.long)
        columns = [
            (steps + offset + codebook) % size
            for codebook, size in enumerate(self.codebook_sizes)
        ]
        return AudioItem(views={self.view: torch.stack(columns, dim=-1)})


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
    raise AssertionError(f"unsupported dataset: {config.name}")


def _codebook_sizes(view: AudioView, codec: Codec) -> tuple[int, ...]:
    semantic = codec.semantic_codebook
    if not isinstance(semantic, Tensor):
        raise TypeError("codec semantic_codebook must be a tensor.")
    if semantic.dim() == 2:
        semantic_sizes = (semantic.size(0),)
    elif semantic.dim() == 3:
        semantic_sizes = (semantic.size(1),) * semantic.size(0)
    else:
        raise ValueError(
            "codec semantic_codebook must have shape [vocab, dim] or "
            "[codebook, vocab, dim]."
        )
    if any(size <= 0 for size in semantic_sizes):
        raise ValueError("codec semantic codebooks must be non-empty.")

    acoustic_sizes = tuple(codec.acoustic_codebook_sizes)
    if any(size <= 0 for size in acoustic_sizes):
        raise ValueError("codec acoustic codebook sizes must be positive.")
    if view is AudioView.LONGCAT:
        if len(semantic_sizes) != 1 or not acoustic_sizes:
            raise ValueError(
                "LongCat toy data requires one semantic and at least one acoustic "
                "codebook."
            )
        return semantic_sizes + acoustic_sizes
    if view is AudioView.UNICODEC:
        if acoustic_sizes:
            raise ValueError("UniCodec toy data cannot contain acoustic codebooks.")
        return semantic_sizes
    raise ValueError(f"unsupported toy dataset audio view: {view.value}")


__all__ = [
    "DatasetConfig",
    "DatasetName",
    "ToyDataset",
    "load_dataset",
]
