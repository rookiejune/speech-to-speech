from __future__ import annotations

import json
from collections.abc import Iterator, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

from anydataset.dataset import MapStyleABC
from anydataset.dataset.speaker import SpeakerAudioGrid
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

from ..._compat import StrEnum, auto
from ...audio_route import PromptSource
from ...runtime.types import (
    CodecBackend,
    acoustic_codec,
    frame_codebook_sizes,
    structured_codec,
)
from ..protocol import DatasetRuntime
from ..types import AudioContextSample


class DatasetName(StrEnum):
    QWEN_TTS_SPEAKER = auto()
    WMT19_TTS = auto()
    TOY = auto()


@dataclass
class DatasetConfig:
    name: DatasetName = DatasetName.WMT19_TTS
    root: Optional[str] = None
    split: str = "train"
    filter: Optional[str] = "speech_translation_v1"
    split_manifest: Optional[str] = None
    split_label: str = "train"
    speaker: Optional[str] = None
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
        if self.filter is not None and not isinstance(self.filter, str):
            raise TypeError("dataset filter must be a string or None.")
        if self.filter == "":
            raise ValueError("dataset filter must not be empty.")
        if self.split_manifest is not None and not isinstance(
            self.split_manifest,
            str,
        ):
            raise TypeError("split_manifest must be a string or None.")
        if not isinstance(self.split_label, str):
            raise TypeError("split_label must be a string.")
        if not self.split_label:
            raise ValueError("split_label must not be empty.")
        if self.speaker is not None:
            if not isinstance(self.speaker, str):
                raise TypeError("dataset speaker must be a string or None.")
            if not self.speaker:
                raise ValueError("dataset speaker must not be empty.")
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


class SpeakerGridCellsDataset(MapStyleABC):
    """Expose all cells or one speaker column from a speaker audio grid."""

    def __init__(
        self,
        grid: SpeakerAudioGrid,
        *,
        speaker: str | None = None,
        with_audio_context: bool = False,
    ) -> None:
        default_text = (Role.DEFAULT, Modality.TEXT)
        default_audio = (Role.DEFAULT, Modality.AUDIO)
        if grid.text_ref != default_text or grid.audio_ref != default_audio:
            raise ValueError(
                "Qwen TTS speaker grids must use Role.DEFAULT text and audio cells."
            )
        self.grid = grid
        self.cells = cast(Dataset[Sample], cast(object, grid.cells))
        self.speaker_ids = tuple(grid.speaker_ids)
        self.with_audio_context = with_audio_context
        if with_audio_context and len(grid) < 2:
            raise ValueError(
                "Qwen TTS audio context requires at least two text rows."
            )
        if speaker is None:
            self.speaker_index = None
        else:
            try:
                self.speaker_index = self.speaker_ids.index(speaker)
            except ValueError as error:
                raise ValueError(
                    f"speaker id {speaker!r} is not present in the Qwen TTS grid."
                ) from error

    def __len__(self) -> int:
        if self.speaker_index is None:
            return len(cast(Sized, self.cells))
        return len(self.grid)

    def __getitem__(self, index: int) -> Sample:
        global_index = self.global_index(index)
        sample = _single_cell(self.cells[global_index], global_index=global_index)
        if not self.with_audio_context:
            return sample
        speaker_count = len(self.speaker_ids)
        row = global_index // speaker_count
        speaker_index = global_index % speaker_count
        context_row = (row + 1) % len(self.grid)
        context_index = context_row * speaker_count + speaker_index
        if context_index == global_index:
            raise RuntimeError("Qwen TTS audio context must differ from its target cell.")
        context = _single_cell(
            self.cells[context_index],
            global_index=context_index,
        )
        return cast(
            Sample,
            cast(
                object,
                AudioContextSample(sample=sample, audio_context=context),
            ),
        )

    def global_index(self, index: int) -> int:
        count = len(self)
        if index < 0:
            index += count
        if index < 0 or index >= count:
            raise IndexError(index)
        if self.speaker_index is None:
            return index
        return index * len(self.speaker_ids) + self.speaker_index

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        if not isinstance(self.cells, MapStyleABC):
            yield from super()._shuffle(
                shuffle=shuffle,
                seed=seed,
                epoch=epoch,
                num_replicas=num_replicas,
                rank=rank,
            )
            return
        if self.speaker_index is None:
            yield from self.cells._shuffle(
                shuffle=shuffle,
                seed=seed,
                epoch=epoch,
                num_replicas=num_replicas,
                rank=rank,
            )
            return

        selected_offset = 0
        speaker_count = len(self.speaker_ids)
        for group in self.cells._shuffle(
            shuffle=shuffle,
            seed=seed,
            epoch=epoch,
            num_replicas=1,
            rank=0,
        ):
            positions: list[int] = []
            for global_index in group:
                if global_index % speaker_count != self.speaker_index:
                    continue
                if selected_offset % num_replicas == rank:
                    positions.append(global_index // speaker_count)
                selected_offset += 1
            if positions:
                yield tuple(positions)


class ToyDataset(MapStyleABC):
    """Deterministic in-memory codec samples for model contract tests."""

    def __init__(
        self,
        codec_name: str,
        codec: CodecBackend,
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
        if self.view is AudioView.BICODEC:
            structured = structured_codec(codec)
            self.codebook_sizes = tuple(structured.semantic_codebook_sizes)
            self.acoustic_codebook_sizes = tuple(structured.acoustic_codebook_sizes)
            unit_length = structured.acoustic_unit_length
            if unit_length is None:
                raise ValueError("BiCodec toy data requires a fixed acoustic unit length.")
            self.acoustic_unit_length = unit_length
        else:
            self.codebook_sizes = _codebook_sizes(self.view, codec)
            self.acoustic_codebook_sizes = ()
            self.acoustic_unit_length = None

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
        if self.view is AudioView.BICODEC:
            semantic = ((steps + offset) % self.codebook_sizes[0]).unsqueeze(-1)
            unit_length = self.acoustic_unit_length
            if unit_length is None:
                raise RuntimeError("BiCodec toy data is missing its acoustic unit length.")
            acoustic_steps = torch.arange(unit_length, dtype=torch.long)
            acoustic = torch.stack(
                [
                    (acoustic_steps + offset + codebook) % size
                    for codebook, size in enumerate(self.acoustic_codebook_sizes)
                ],
                dim=-1,
            )
            return AudioItem(
                views={AudioView.BICODEC: {"semantic": semantic, "acoustic": acoustic}},
                meta={AudioMeta.DURATION: self.frames / self.frame_rate},
            )
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
        _reject_speaker(config)
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
        _reject_speaker(config)
        from zhuyin.datasets.wmt19_tts import wmt19_tts_codec

        codec_name = (
            "stable" if runtime.codec_name == "stable_codec" else runtime.codec_name
        )

        return _apply_split_manifest(
            cast(
                Dataset[Sample],
                cast(
                    object,
                    wmt19_tts_codec(
                        codec=codec_name,
                        root=(
                            None
                            if config.root is None
                            else Path(config.root).expanduser()
                        ),
                        split=config.split,
                        filter=config.filter,
                    ),
                ),
            ),
            config,
        )
    if config.name is DatasetName.QWEN_TTS_SPEAKER:
        if runtime.codec_name not in {"bicodec", "longcat"}:
            raise ValueError(
                "qwen_tts_speaker supports only the bicodec and longcat runtimes."
            )
        from zhuyin.datasets.wmt19.qwen_tts import (
            speaker_grid as qwen_tts_speaker_codec_grid,
        )

        grid = qwen_tts_speaker_codec_grid(
            codec=runtime.codec_name,
            root=None if config.root is None else Path(config.root).expanduser(),
            split=config.split,
        )
        return _apply_split_manifest(
            SpeakerGridCellsDataset(
                grid,
                speaker=config.speaker,
                with_audio_context=(
                    runtime.audio_route is not None
                    and bool(runtime.audio_route.prompt.streams)
                    and runtime.audio_route.prompt.source is PromptSource.REFERENCE
                ),
            ),
            config,
        )
    raise AssertionError(f"unsupported dataset: {config.name}")


def _reject_speaker(config: DatasetConfig) -> None:
    if config.speaker is not None:
        raise ValueError(
            "dataset speaker selection is supported only by qwen_tts_speaker."
        )


def _single_cell(sample: Sample, *, global_index: int) -> Sample:
    for ref, expected in (
        ((Role.DEFAULT, Modality.TEXT), TextItem),
        ((Role.DEFAULT, Modality.AUDIO), AudioItem),
    ):
        try:
            item = sample[ref]
        except KeyError as error:
            raise ValueError(
                f"Qwen TTS speaker cell {global_index} must use Role.DEFAULT "
                "text and audio items."
            ) from error
        if not isinstance(item, expected):
            raise TypeError(
                f"Qwen TTS speaker cell {global_index} {ref!r} must contain "
                f"{expected.__name__}."
            )
    return sample


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


def _codebook_sizes(view: AudioView, codec: object) -> tuple[int, ...]:
    sizes = frame_codebook_sizes(codec)
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
    if view is AudioView.STABLE:
        return sizes
    raise ValueError(f"unsupported toy dataset audio view: {view.value}")


__all__ = [
    "DatasetConfig",
    "DatasetName",
    "SpeakerGridCellsDataset",
    "SplitManifestDataset",
    "ToyDataset",
    "load_dataset",
]
