from __future__ import annotations

import json
from collections.abc import Iterator, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

from anydataset.dataset import MapStyleABC
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
    WMT19_TTS = auto()
    TOY = auto()


@dataclass
class DatasetConfig:
    name: DatasetName = DatasetName.WMT19_TTS
    root: Optional[str] = None
    split: str = "train"
    split_manifest: Optional[str] = None
    split_label: str = "train"
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
        if self.split_manifest is not None and not isinstance(
            self.split_manifest,
            str,
        ):
            raise TypeError("split_manifest must be a string or None.")
        if not isinstance(self.split_label, str):
            raise TypeError("split_label must be a string.")
        if not self.split_label:
            raise ValueError("split_label must not be empty.")
        for name, value in (
            ("toy_samples", self.toy_samples),
            ("toy_frames", self.toy_frames),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")


class SplitManifestDataset(MapStyleABC):
    """Dataset view backed by an explicit split manifest index list."""

    def __init__(
        self,
        dataset: Dataset[Sample],
        indices: Sequence[int],
        *,
        manifest: Path,
        label: str,
    ) -> None:
        self.dataset = dataset
        self.manifest = manifest
        self.label = label
        self.indices = _validate_indices(
            indices,
            label=label,
            count=len(cast(Sized, dataset)),
        )
        self._positions = {
            global_index: position
            for position, global_index in enumerate(self.indices)
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Sample:
        return self.dataset[self.global_index(index)]

    def global_index(self, index: int) -> int:
        if index < 0:
            index += len(self.indices)
        if index < 0 or index >= len(self.indices):
            raise IndexError(index)
        return self.indices[index]

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        if not isinstance(self.dataset, MapStyleABC):
            yield from super()._shuffle(
                shuffle=shuffle,
                seed=seed,
                epoch=epoch,
                num_replicas=num_replicas,
                rank=rank,
            )
            return
        for group in self.dataset._shuffle(
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
            num_replicas=num_replicas,
            rank=rank,
        ):
            positions = tuple(
                self._positions[index]
                for index in group
                if index in self._positions
            )
            if positions:
                yield positions


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
        return _apply_split_manifest(
            ToyDataset(
                runtime.codec_name,
                runtime.codec,
                samples=config.toy_samples,
                frames=config.toy_frames,
            ),
            config,
        )
    if config.name is DatasetName.WMT19_TTS:
        from zhuyin.datasets.wmt19_tts import wmt19_tts_codec

        return _apply_split_manifest(
            cast(
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
            ),
            config,
        )
    raise AssertionError(f"unsupported dataset: {config.name}")


def _apply_split_manifest(
    dataset: Dataset[Sample],
    config: DatasetConfig,
) -> Dataset[Sample]:
    if config.split_manifest is None:
        return dataset
    manifest = Path(config.split_manifest).expanduser()
    return SplitManifestDataset(
        dataset,
        _read_split_indices(manifest, config.split_label),
        manifest=manifest,
        label=config.split_label,
    )


def _read_split_indices(path: Path, label: str) -> tuple[int, ...]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise FileNotFoundError(f"split manifest does not exist: {path}") from error
    if not isinstance(payload, dict):
        raise TypeError("split manifest root must be a JSON object.")
    splits = payload.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("split manifest must contain a 'splits' object.")
    try:
        raw = splits[label]
    except KeyError as error:
        raise ValueError(f"split manifest does not contain split {label!r}.") from error
    if not isinstance(raw, list):
        raise TypeError(f"split manifest split {label!r} must be a list.")
    if not raw:
        raise ValueError(f"split manifest split {label!r} must not be empty.")
    return _index_tuple(raw, label=label)


def _index_tuple(raw: Sequence[object], *, label: str) -> tuple[int, ...]:
    indices: list[int] = []
    seen: set[int] = set()
    for offset, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"split manifest split {label!r} index {offset} must be an integer."
            )
        index = value
        if index < 0:
            raise ValueError(
                f"split manifest split {label!r} index {offset} must be non-negative."
            )
        if index in seen:
            raise ValueError(
                f"split manifest split {label!r} repeats dataset index {index}."
            )
        indices.append(index)
        seen.add(index)
    return tuple(indices)


def _validate_indices(
    indices: Sequence[int],
    *,
    label: str,
    count: int,
) -> tuple[int, ...]:
    result = tuple(indices)
    for offset, index in enumerate(result):
        if index >= count:
            raise IndexError(
                f"split manifest split {label!r} index {offset} points outside "
                f"dataset length {count}: {index}."
            )
    return result


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
    raise ValueError(f"unsupported toy dataset audio view: {view.value}")


__all__ = [
    "DatasetConfig",
    "DatasetName",
    "SplitManifestDataset",
    "ToyDataset",
    "load_dataset",
]
