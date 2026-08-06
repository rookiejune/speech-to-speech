"""Durable per-stage artifacts for resumable streaming synthesis batches."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Sequence, Sized
from pathlib import Path
from typing import cast

from anydataset import AnyDataset
from anydataset.store import DatasetWriter
from anydataset.types import Sample


_SCHEMA = "speech-to-speech-synthesis-stage-cache-v1"
_STAGES = frozenset(
    {"source_tts", "translation", "target_tts", "codec", "input_codec"}
)


class SynthesisStageCache:
    """Atomically retain completed model stages until a snapshot publishes."""

    def __init__(
        self,
        root: Path,
        *,
        stream_id: str,
        split: str,
        identity_sha256: str,
    ) -> None:
        self.root = root.expanduser().resolve() / ".stage-cache"
        self.stream_id = _string(stream_id, "stream_id")
        self.split = _string(split, "split")
        self.identity_sha256 = _digest(identity_sha256, "identity_sha256")
        self.root.mkdir(parents=True, exist_ok=True)
        self._discard_temporary()

    def load(
        self,
        snapshot_id: str,
        stage: str,
        indices: Sequence[int],
    ) -> list[Sample] | None:
        """Load one completed stage or return ``None`` before it is durable."""

        expected = _indices(indices)
        stage_root = self._stage(snapshot_id, stage)
        if not stage_root.exists():
            return None
        metadata = _metadata(stage_root / "stage.json")
        _validate_metadata(
            metadata,
            path=stage_root / "stage.json",
            stream_id=self.stream_id,
            identity_sha256=self.identity_sha256,
            snapshot_id=snapshot_id,
            stage=stage,
            indices=expected,
        )
        dataset = AnyDataset.from_store(stage_root / "store", split=self.split)
        if not isinstance(dataset, Sized):
            raise TypeError("synthesis stage cache dataset must expose __len__().")
        if len(dataset) != len(expected):
            raise ValueError(
                "synthesis stage cache sample count does not match its metadata."
            )
        return [cast(Sample, dataset[index]) for index in range(len(dataset))]

    def save(
        self,
        snapshot_id: str,
        stage: str,
        indices: Sequence[int],
        samples: Sequence[Sample],
    ) -> None:
        """Publish one stage directory only after every payload is ready."""

        expected = _indices(indices)
        if len(samples) != len(expected):
            raise ValueError("synthesis stage cache samples must match indices.")
        target = self._stage(snapshot_id, stage)
        if target.exists():
            raise RuntimeError(f"synthesis stage cache already exists: {target}.")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{stage}.{uuid.uuid4().hex}.tmp")
        temporary.mkdir()
        try:
            DatasetWriter(
                temporary / "store",
                dataset_id=f"{self.stream_id}-stage-{stage}",
                split=self.split,
            ).write(samples)
            _write_json(
                temporary / "stage.json",
                {
                    "schema": _SCHEMA,
                    "stream_id": self.stream_id,
                    "identity_sha256": self.identity_sha256,
                    "snapshot_id": snapshot_id,
                    "stage": stage,
                    "sample_indices": list(expected),
                    "sample_count": len(expected),
                },
            )
            os.replace(temporary, target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def discard(self, snapshot_id: str) -> None:
        """Remove cache data after the corresponding snapshot is immutable."""

        batch = self.root / _segment(snapshot_id, "snapshot_id")
        if batch.exists():
            shutil.rmtree(batch)

    def discard_through(self, position: int) -> None:
        """Clean cache batches already covered by a resumed snapshot prefix."""

        if type(position) is not int or position < 0:
            raise ValueError("synthesis stage cache position must be non-negative.")
        for batch in self.root.iterdir():
            if not batch.is_dir() or batch.name.startswith("."):
                continue
            metadata_paths = sorted(batch.glob("*/stage.json"))
            if not metadata_paths:
                continue
            metadata = _metadata(metadata_paths[0])
            values = metadata.get("sample_indices")
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"synthesis stage cache has invalid sample_indices: {metadata_paths[0]}."
                )
            if all(type(index) is int and 0 <= index < position for index in values):
                shutil.rmtree(batch)

    def _stage(self, snapshot_id: str, stage: str) -> Path:
        return self.root / _segment(snapshot_id, "snapshot_id") / _stage(stage)

    def _discard_temporary(self) -> None:
        for batch in self.root.iterdir():
            if not batch.is_dir() or batch.name.startswith("."):
                continue
            for path in batch.iterdir():
                if path.is_dir() and path.name.startswith(".") and path.name.endswith(".tmp"):
                    shutil.rmtree(path)


def _validate_metadata(
    value: dict[str, object],
    *,
    path: Path,
    stream_id: str,
    identity_sha256: str,
    snapshot_id: str,
    stage: str,
    indices: tuple[int, ...],
) -> None:
    expected: dict[str, object] = {
        "schema": _SCHEMA,
        "stream_id": stream_id,
        "identity_sha256": identity_sha256,
        "snapshot_id": snapshot_id,
        "stage": stage,
        "sample_indices": list(indices),
        "sample_count": len(indices),
    }
    if value != expected:
        raise ValueError(f"synthesis stage cache metadata is incompatible: {path}.")


def _metadata(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"synthesis stage cache metadata is invalid JSON: {path}.") from error
    if not isinstance(value, dict):
        raise TypeError(f"synthesis stage cache metadata must be an object: {path}.")
    return cast(dict[str, object], value)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def _indices(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise ValueError("synthesis stage cache indices must be non-empty.")
    if any(type(value) is not int or value < 0 for value in result):
        raise ValueError("synthesis stage cache indices must be non-negative integers.")
    if len(set(result)) != len(result):
        raise ValueError("synthesis stage cache indices must be unique.")
    return result


def _stage(value: str) -> str:
    result = _string(value, "stage")
    if result not in _STAGES:
        raise ValueError(f"unsupported synthesis cache stage: {result!r}.")
    return result


def _segment(value: str, name: str) -> str:
    result = _string(value, name)
    if result in {".", ".."} or Path(result).name != result or "\\" in result:
        raise ValueError(f"{name} must be one safe path segment.")
    return result


def _digest(value: str, name: str) -> str:
    result = _string(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a lowercase SHA256 digest.")
    return result


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


__all__ = ["SynthesisStageCache"]
