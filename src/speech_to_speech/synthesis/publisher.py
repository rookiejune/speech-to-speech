"""Atomic publisher for immutable streaming-synthesis snapshot stores."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence, Sized
from pathlib import Path
from typing import TextIO

from anydataset.store import DatasetWriter
from anydataset.store.reader import read_store_manifest
from anydataset.types import Sample
from torch.utils.data import Dataset

from speech_to_speech.datamodule.streaming import (
    SnapshotFeed,
    SnapshotLoader,
    directional_codec_sample,
)


_SNAPSHOT_SCHEMA = "speech-to-speech-stream-snapshot-v1"
_SEAL_SCHEMA = "speech-to-speech-stream-seal-v1"


class SnapshotPublisher:
    """Publish one complete base/input/output store set as one immutable snapshot."""

    def __init__(
        self,
        root: Path,
        *,
        stream_id: str,
        expected_samples: int,
        codec: str,
        split: str,
        loader: SnapshotLoader,
        input_codec: str | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.stream_id = _string(stream_id, "stream_id")
        self.expected_samples = _positive_int(expected_samples, "expected_samples")
        self.codec = _segment(codec, "codec")
        self.input_codec = _segment(
            self.codec if input_codec is None else input_codec,
            "input_codec",
        )
        self.split = _string(split, "split")
        self.feed = SnapshotFeed(
            self.root,
            stream_id=self.stream_id,
            expected_samples=self.expected_samples,
            loader=loader,
            codec=self.codec,
            input_codec=self.input_codec,
        )

    def publish(
        self,
        *,
        snapshot_id: str,
        sample_indices: Sequence[int],
        base_samples: Sequence[Sample],
        codec_samples: Sequence[Sample],
        input_codec_samples: Sequence[Sample] | None = None,
    ) -> Path:
        """Atomically publish one non-overlapping snapshot and seal if complete."""

        snapshot_id = _segment(snapshot_id, "snapshot_id")
        indices = _indices(sample_indices, expected=self.expected_samples)
        if len(base_samples) != len(indices) or len(codec_samples) != len(indices):
            raise ValueError("streaming snapshot stores must each match sample_indices.")
        decoupled = self.input_codec != self.codec
        if decoupled:
            if input_codec_samples is None or len(input_codec_samples) != len(indices):
                raise ValueError(
                    "decoupled streaming snapshots require an input codec store "
                    "matching sample_indices."
                )
            for index in range(len(indices)):
                directional_codec_sample(
                    input_codec_samples[index],
                    codec_samples[index],
                )
        elif input_codec_samples is not None:
            raise ValueError(
                "input_codec_samples are only accepted when input/output codecs differ."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        with _PublicationLock(self.root / ".publisher.lock"):
            status = self.feed.status()
            existing = next(
                (item for item in status.catalog.snapshots if item.snapshot_id == snapshot_id),
                None,
            )
            if existing is not None:
                if existing.sample_indices != indices:
                    raise ValueError("streaming snapshot id was already published with other indices.")
                _validate_store(existing.root / "base", count=len(indices))
                if decoupled:
                    _validate_store(existing.root / self.input_codec, count=len(indices))
                _validate_store(existing.root / self.codec, count=len(indices))
                dataset = self.feed.load(existing)
                if decoupled:
                    _validate_dataset(dataset, count=len(indices))
                self._seal_if_complete()
                return existing.root
            if status.seal is not None:
                raise RuntimeError("cannot publish a new streaming snapshot after sealing.")
            overlap = set(indices) & set(status.catalog.locations)
            if overlap:
                raise ValueError(
                    f"streaming snapshot overlaps already published index {min(overlap)}."
                )
            sequence = len(status.catalog.snapshots)
            target = self.root / "snapshots" / f"{sequence:06d}-{snapshot_id}"
            if target.exists():
                raise RuntimeError(f"streaming snapshot target already exists: {target}.")
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            temporary.mkdir(parents=True)
            try:
                DatasetWriter(
                    temporary / "base",
                    dataset_id=f"{self.stream_id}-base",
                    split=self.split,
                ).write(base_samples)
                if decoupled:
                    if input_codec_samples is None:
                        raise AssertionError("decoupled input codec samples were not resolved.")
                    DatasetWriter(
                        temporary / self.input_codec,
                        dataset_id=f"{self.stream_id}-{self.input_codec}",
                        split=self.split,
                    ).write(input_codec_samples)
                DatasetWriter(
                    temporary / self.codec,
                    dataset_id=f"{self.stream_id}-{self.codec}",
                    split=self.split,
                ).write(codec_samples)
                _validate_store(temporary / "base", count=len(indices))
                if decoupled:
                    _validate_store(temporary / self.input_codec, count=len(indices))
                _validate_store(temporary / self.codec, count=len(indices))
                _write_json(
                    temporary / "snapshot.json",
                    {
                        "schema": _SNAPSHOT_SCHEMA,
                        "stream_id": self.stream_id,
                        "expected_samples": self.expected_samples,
                        "input_codec": self.input_codec,
                        "codec": self.codec,
                        "sequence": sequence,
                        "snapshot_id": snapshot_id,
                        "sample_indices": list(indices),
                        "sample_count": len(indices),
                    },
                )
                if decoupled:
                    _validate_dataset(
                        self.feed.loader(temporary),
                        count=len(indices),
                    )
                os.replace(temporary, target)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
            self._seal_if_complete()
            return target

    def _seal_if_complete(self) -> None:
        status = self.feed.status()
        if status.seal is not None:
            return
        catalog = status.catalog
        if catalog.sample_count != self.expected_samples:
            return
        if set(catalog.locations) != set(range(self.expected_samples)):
            return
        _write_json(
            self.root / "sealed.json",
            {
                "schema": _SEAL_SCHEMA,
                "stream_id": self.stream_id,
                "expected_samples": self.expected_samples,
                "input_codec": self.input_codec,
                "codec": self.codec,
                "snapshot_count": len(catalog.snapshots),
                "sample_count": catalog.sample_count,
                "catalog_sha256": catalog.sha256,
            },
        )


class _PublicationLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: TextIO | None = None

    def __enter__(self) -> _PublicationLock:
        import fcntl

        self._file = self.path.open("a", encoding="utf-8")
        fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        import fcntl

        if self._file is None:
            return
        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()
        self._file = None


def _indices(values: Sequence[int], *, expected: int) -> tuple[int, ...]:
    if not values:
        raise ValueError("streaming snapshot sample_indices must be non-empty.")
    result = tuple(values)
    if any(type(value) is not int for value in result):
        raise TypeError("streaming snapshot sample_indices must be integers.")
    if any(value < 0 or value >= expected for value in result):
        raise ValueError("streaming snapshot sample_indices are outside the logical epoch.")
    if len(set(result)) != len(result):
        raise ValueError("streaming snapshot sample_indices must be unique.")
    return result


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _segment(value: object, name: str) -> str:
    result = _string(value, name)
    if result in {".", ".."} or Path(result).name != result or "\\" in result:
        raise ValueError(f"{name} must be one safe path segment.")
    return result


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_store(root: Path, *, count: int) -> None:
    manifest = read_store_manifest(root)
    if manifest.sample_count != count:
        raise RuntimeError(
            f"streaming snapshot store sample_count mismatch at {root}: "
            f"{manifest.sample_count} != {count}."
        )


def _validate_dataset(dataset: Dataset[Sample], *, count: int) -> None:
    if not isinstance(dataset, Sized):
        raise TypeError("streaming snapshot datasets must expose __len__().")
    actual = len(dataset)
    if actual != count:
        raise RuntimeError(
            "streaming snapshot dataset sample_count mismatch: "
            f"{actual} != {count}."
        )
    for index in range(actual):
        dataset[index]


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)
