from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import time
from bisect import bisect_right
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch.distributed as dist
from anydataset.types import Modality, Role, Sample
from lightning.pytorch.utilities.exceptions import SIGTERMException
from torch.utils.data import DataLoader, Dataset, IterableDataset


_SNAPSHOT_SCHEMA_V1 = "speech-to-speech-stream-snapshot-v1"
_SNAPSHOT_SCHEMA = "speech-to-speech-stream-snapshot-v2"
_SEAL_SCHEMA = "speech-to-speech-stream-seal-v1"
_FAILURE_SCHEMA = "speech-to-speech-stream-failure-v1"
_CURSOR_SCHEMA = "speech-to-speech-stream-cursor-v1"
_LOADER_SCHEMA = "speech-to-speech-stream-loader-v1"
_TRANSLATION_REFERENCE_SCHEMA = "speech-to-speech-translation-references-v1"
_TRANSLATION_REFERENCE_FILE = "translation_references.jsonl"
_LOGGER = logging.getLogger(__name__)


class SnapshotLoader(Protocol):
    def __call__(self, root: Path) -> Dataset[Sample]: ...


class SynthesisController(Protocol):
    """Long-lived producer resumed by each invocation of the training entry."""

    def start(self) -> None: ...

    def check(self) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SynthesisRequest:
    root: Path
    stream_id: str
    expected_samples: int
    codec: str
    split: str
    options: Mapping[str, object]
    input_codec: str | None = None

    @property
    def resolved_input_codec(self) -> str:
        return self.codec if self.input_codec is None else self.input_codec


class SynthesisFactory(Protocol):
    def __call__(self, request: SynthesisRequest) -> SynthesisController: ...


@dataclass(frozen=True)
class Snapshot:
    root: Path
    snapshot_id: str
    sequence: int
    sample_indices: tuple[int, ...]
    manifest_sha256: str
    translation_references_sha256: str | None

    @property
    def sample_count(self) -> int:
        return len(self.sample_indices)


@dataclass(frozen=True)
class Catalog:
    snapshots: tuple[Snapshot, ...]
    cumulative_counts: tuple[int, ...]
    locations: Mapping[int, tuple[int, int]]
    prefix_sha256: tuple[str, ...]
    sha256: str

    @property
    def sample_count(self) -> int:
        return self.cumulative_counts[-1] if self.cumulative_counts else 0


@dataclass(frozen=True)
class Seal:
    snapshot_count: int
    sample_count: int
    catalog_sha256: str


@dataclass(frozen=True)
class StreamStatus:
    catalog: Catalog
    seal: Seal | None


@dataclass(frozen=True)
class PublishedSample:
    index: int
    snapshot_id: str
    sample: Sample
    reference_translation: str


@dataclass(frozen=True)
class BatchSpan:
    start: int
    stop: int


@dataclass(frozen=True)
class StreamingTelemetry:
    """Timing and cursor counters for one streaming loader."""

    batch_fetch_seconds: float
    batch_wait_seconds: float
    batch_load_seconds: float
    total_fetch_seconds: float
    total_wait_seconds: float
    total_load_seconds: float
    wait_events: int
    poll_count: int
    read_position: int
    committed_position: int
    committed_batches: int
    published_samples: int
    expected_samples: int


@dataclass(frozen=True)
class WorkspaceSnapshotLoader:
    codec: str
    split: str
    input_codec: str | None = None

    def __call__(self, root: Path) -> Dataset[Sample]:
        from zhuyin.datasets.wmt19 import streaming_s2st

        output = _workspace_codec_dataset(
            streaming_s2st,
            self.codec,
            root=root,
            split=self.split,
        )
        input_codec = self.codec if self.input_codec is None else self.input_codec
        if input_codec == self.codec:
            return output
        input_dataset = _workspace_codec_dataset(
            streaming_s2st,
            input_codec,
            root=root,
            split=self.split,
        )
        return _DirectionalCodecDataset(input_dataset, output)


class _DirectionalCodecDataset(Dataset[Sample]):
    """Join source/input and target/output codec stores by snapshot-local index."""

    def __init__(
        self,
        input_dataset: Dataset[Sample],
        output_dataset: Dataset[Sample],
    ) -> None:
        if not isinstance(input_dataset, Sized) or not isinstance(output_dataset, Sized):
            raise TypeError("streaming codec stores must expose __len__().")
        input_count = len(input_dataset)
        output_count = len(output_dataset)
        if input_count != output_count:
            raise ValueError(
                "streaming input/output codec stores must have the same length: "
                f"{input_count} != {output_count}."
            )
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset
        self._count = input_count

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> Sample:
        input_sample = self.input_dataset[index]
        output_sample = self.output_dataset[index]
        source_audio = (Role.SOURCE, Modality.AUDIO)
        target_audio = (Role.TARGET, Modality.AUDIO)
        if source_audio not in input_sample:
            raise KeyError("streaming input codec sample is missing source audio.")
        if target_audio not in output_sample:
            raise KeyError("streaming output codec sample is missing target audio.")
        for role in (Role.SOURCE, Role.TARGET):
            reference = (role, Modality.TEXT)
            if reference not in input_sample or reference not in output_sample:
                raise KeyError(
                    "streaming input/output codec samples must both contain aligned text."
                )
            if input_sample.get(reference) != output_sample.get(reference):
                raise ValueError(
                    "streaming input/output codec stores disagree on aligned text."
                )
        merged = dict(output_sample)
        merged[source_audio] = input_sample[source_audio]
        return cast(Sample, merged)


def _workspace_codec_dataset(
    streaming_s2st: Any,
    codec: str,
    *,
    root: Path,
    split: str,
) -> Dataset[Sample]:
    if codec == "glm4":
        from anydataset import AnyDataset

        dataset = AnyDataset.from_store(root / codec, split=split)
    else:
        workspace_codec = "stable" if codec == "stable_codec" else codec
        dataset = streaming_s2st.codec(
            workspace_codec,
            root=root,
            split=split,
        ).load()
    if not isinstance(dataset, Dataset):
        raise TypeError("streaming snapshot loader must return a torch Dataset.")
    return cast(Dataset[Sample], cast(object, dataset))


def workspace_stream_root(root: str | None) -> Path:
    from zhuyin.datasets.wmt19 import streaming_s2st

    return streaming_s2st.dataset_root(root).expanduser().resolve()


class SnapshotFeed:
    """Discover append-only, non-overlapping immutable synthesis chunks."""

    def __init__(
        self,
        root: Path,
        *,
        stream_id: str,
        expected_samples: int,
        loader: SnapshotLoader,
        codec: str | None = None,
        input_codec: str | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.stream_id = _nonempty_string(stream_id, "stream_id")
        self.expected_samples = _positive_int(expected_samples, "expected_samples")
        if codec is None:
            if input_codec is not None:
                raise ValueError("streaming input_codec requires codec.")
            self.codec = None
            self.input_codec = None
        else:
            self.codec = _nonempty_string(codec, "codec")
            self.input_codec = _nonempty_string(
                self.codec if input_codec is None else input_codec,
                "input_codec",
            )
        if not callable(loader):
            raise TypeError("snapshot loader must be callable.")
        self.loader = loader
        self._datasets: dict[str, Dataset[Sample]] = {}
        self._reference_cache: dict[
            str,
            tuple[int, int, tuple[str, ...]],
        ] = {}
        self._known: dict[Path, tuple[int, int, Snapshot]] = {}
        self._catalog_signature: tuple[
            tuple[Path, str, int, int, str], ...
        ] | None = None
        self._catalog: Catalog | None = None
        self._known_seal: tuple[int, int, Seal] | None = None

    def status(self) -> StreamStatus:
        catalog = self._catalog_value()
        seal = self._seal()
        if seal is not None:
            # A final snapshot may become visible between the first catalog scan
            # and the seal read.  Scan once more after observing the seal so one
            # status call cannot combine an old catalog with the new seal.
            catalog = self._catalog_value()
        if seal is None:
            self._raise_failure()
        if seal is not None:
            if seal.snapshot_count != len(catalog.snapshots):
                raise RuntimeError(
                    "sealed streaming synthesis snapshot count differs from its catalog."
                )
            if seal.sample_count != catalog.sample_count:
                raise RuntimeError(
                    "sealed streaming synthesis sample count differs from its catalog."
                )
            if seal.catalog_sha256 != catalog.sha256:
                raise RuntimeError(
                    "sealed streaming synthesis catalog digest does not match."
                )
            if catalog.sample_count != self.expected_samples:
                raise RuntimeError(
                    "sealed streaming synthesis does not cover the logical epoch."
                )
            if set(catalog.locations) != set(range(self.expected_samples)):
                raise RuntimeError(
                    "sealed streaming synthesis membership is not exactly 0..2N-1."
                )
        return StreamStatus(catalog=catalog, seal=seal)

    def _raise_failure(self) -> None:
        path = self.root / "failed.json"
        if not path.is_file():
            return
        payload, _digest = _manifest(path, schema=_FAILURE_SCHEMA)
        _identity(
            payload,
            path=path,
            stream_id=self.stream_id,
            expected=self.expected_samples,
            codec=self.codec,
            input_codec=self.input_codec,
        )
        error = _manifest_string(payload, "error", path=path)
        exit_code = payload.get("exit_code")
        if exit_code is not None and type(exit_code) is not int:
            raise TypeError(
                f"streaming failure manifest exit_code must be an integer: {path}."
            )
        suffix = "" if exit_code is None else f" (exit code {exit_code})"
        raise RuntimeError(f"streaming synthesis producer failed{suffix}: {error}")

    def load(self, snapshot: Snapshot) -> Dataset[Sample]:
        cached = self._datasets.get(snapshot.snapshot_id)
        if cached is not None:
            return cached
        dataset = self.loader(snapshot.root)
        if not isinstance(dataset, Sized):
            raise TypeError("streaming snapshot datasets must expose __len__().")
        actual = len(dataset)
        if actual != snapshot.sample_count:
            raise ValueError(
                "streaming snapshot sample_count does not match its dataset: "
                f"{snapshot.sample_count} != {actual}."
            )
        self._datasets[snapshot.snapshot_id] = dataset
        return dataset

    def sample_at(self, position: int, catalog: Catalog) -> tuple[Snapshot, int, Sample]:
        if position < 0 or position >= catalog.sample_count:
            raise IndexError(position)
        sequence = bisect_right(catalog.cumulative_counts, position)
        previous = 0 if sequence == 0 else catalog.cumulative_counts[sequence - 1]
        offset = position - previous
        snapshot = catalog.snapshots[sequence]
        return snapshot, offset, self.load(snapshot)[offset]

    def published(self, indices: Sequence[int]) -> list[PublishedSample]:
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("streaming sample indices must be non-negative integers.")
        catalog = self.status().catalog
        result: list[PublishedSample] = []
        for index in indices:
            location = catalog.locations.get(index)
            if location is None:
                continue
            sequence, offset = location
            snapshot = catalog.snapshots[sequence]
            references = self._translation_references(snapshot)
            result.append(
                PublishedSample(
                    index,
                    snapshot.snapshot_id,
                    self.load(snapshot)[offset],
                    references[offset],
                )
            )
        return result

    def _translation_references(self, snapshot: Snapshot) -> tuple[str, ...]:
        expected_sha256 = snapshot.translation_references_sha256
        if expected_sha256 is None:
            raise RuntimeError(
                "published streaming snapshot has no dataset translation references: "
                f"{snapshot.root}."
            )
        path = snapshot.root / _TRANSLATION_REFERENCE_FILE
        try:
            stat = path.stat()
        except FileNotFoundError as error:
            raise RuntimeError(
                f"streaming translation reference sidecar is missing: {path}."
            ) from error
        cached = self._reference_cache.get(snapshot.snapshot_id)
        identity = (stat.st_mtime_ns, stat.st_size)
        if cached is not None:
            if cached[:2] != identity:
                raise RuntimeError(
                    f"published translation reference sidecar was modified: {path}."
                )
            return cached[2]
        data = path.read_bytes()
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"streaming translation reference sidecar digest mismatch: {path}."
            )
        references = _translation_reference_texts(
            data,
            indices=snapshot.sample_indices,
            path=path,
        )
        self._reference_cache[snapshot.snapshot_id] = (*identity, references)
        return references

    def cursor(self, position: int, catalog: Catalog) -> tuple[int, int]:
        if position < 0 or position > catalog.sample_count:
            raise ValueError("streaming cursor is outside the published catalog.")
        if position == catalog.sample_count:
            return len(catalog.snapshots), 0
        sequence = bisect_right(catalog.cumulative_counts, position)
        previous = 0 if sequence == 0 else catalog.cumulative_counts[sequence - 1]
        return sequence, position - previous

    def _catalog_value(self) -> Catalog:
        directory = self.root / "snapshots"
        paths = (
            sorted(
                path
                for path in directory.glob("*/snapshot.json")
                if not path.parent.name.startswith(".")
            )
            if directory.exists()
            else []
        )
        snapshots = tuple(self._read_snapshot(path) for path in paths)
        snapshots = tuple(sorted(snapshots, key=lambda value: value.sequence))
        signature = tuple(
            (
                snapshot.root,
                snapshot.snapshot_id,
                snapshot.sequence,
                snapshot.sample_count,
                snapshot.manifest_sha256,
            )
            for snapshot in snapshots
        )
        if signature == self._catalog_signature and self._catalog is not None:
            return self._catalog
        catalog = _catalog(snapshots, expected=self.expected_samples)
        previous = self._catalog
        if previous is not None:
            if len(catalog.snapshots) < len(previous.snapshots):
                raise RuntimeError("published streaming snapshot catalog shrank.")
            if catalog.snapshots[: len(previous.snapshots)] != previous.snapshots:
                raise RuntimeError("published streaming snapshot catalog forked.")
        self._catalog_signature = signature
        self._catalog = catalog
        return catalog

    def _read_snapshot(self, path: Path) -> Snapshot:
        stat = path.stat()
        cached = self._known.get(path)
        identity = (stat.st_mtime_ns, stat.st_size)
        if cached is not None:
            if cached[:2] != identity:
                raise RuntimeError(f"published streaming snapshot was modified: {path}.")
            return cached[2]
        snapshot = _snapshot(
            path,
            stream_id=self.stream_id,
            expected=self.expected_samples,
            codec=self.codec,
            input_codec=self.input_codec,
        )
        self._known[path] = (*identity, snapshot)
        return snapshot

    def _seal(self) -> Seal | None:
        path = self.root / "sealed.json"
        if not path.is_file():
            if self._known_seal is not None:
                raise RuntimeError("published streaming synthesis seal disappeared.")
            return None
        stat = path.stat()
        identity = (stat.st_mtime_ns, stat.st_size)
        if self._known_seal is not None:
            if self._known_seal[:2] != identity:
                raise RuntimeError("published streaming synthesis seal was modified.")
            return self._known_seal[2]
        payload, _digest = _manifest(path, schema=_SEAL_SCHEMA)
        _identity(
            payload,
            path=path,
            stream_id=self.stream_id,
            expected=self.expected_samples,
            codec=self.codec,
            input_codec=self.input_codec,
        )
        sample_count = _manifest_int(payload, "sample_count", path=path, positive=True)
        if sample_count != self.expected_samples:
            raise ValueError(
                "sealed streaming synthesis must contain every expected sample."
            )
        seal = Seal(
            snapshot_count=_manifest_int(
                payload,
                "snapshot_count",
                path=path,
                positive=True,
            ),
            sample_count=sample_count,
            catalog_sha256=_manifest_digest(payload, "catalog_sha256", path=path),
        )
        self._known_seal = (*identity, seal)
        return seal


class StreamingSnapshotDataset(IterableDataset[Sample]):
    """Tail published chunks while separating read and committed positions."""

    def __init__(
        self,
        feed: SnapshotFeed,
        *,
        batch_size: int,
        poll_seconds: float,
        status_seconds: float,
        rank_world: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.feed = feed
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.poll_seconds = _positive_float(poll_seconds, "poll_seconds")
        self.status_seconds = _positive_float(status_seconds, "status_seconds")
        self._rank_world = rank_world or _distributed_rank_world
        self._read_position = 0
        self._committed_position = 0
        self._committed_batches = 0
        self._world_size: int | None = None
        self._checkpoint_catalog: Mapping[str, object] | None = None
        self._waiting_since: float | None = None
        self._wait_seconds = 0.0
        self._wait_events = 0
        self._poll_count = 0
        self._stop_requested: Callable[[], bool] | None = None

    @property
    def read_position(self) -> int:
        return self._read_position

    @property
    def committed_position(self) -> int:
        return self._committed_position

    @property
    def committed_batches(self) -> int:
        return self._committed_batches

    @property
    def wait_seconds(self) -> float:
        return self._wait_seconds

    @property
    def wait_events(self) -> int:
        return self._wait_events

    @property
    def poll_count(self) -> int:
        return self._poll_count

    def set_stop_requested(self, requested: Callable[[], bool]) -> None:
        if not callable(requested):
            raise TypeError("streaming stop request must be callable.")
        self._stop_requested = requested

    def logical_batch_count(self) -> int:
        rank, world_size = self._rank_world()
        _validate_rank_world(rank, world_size)
        if self._world_size is not None and self._world_size != world_size:
            raise ValueError(
                "streaming resume requires the same DDP world size: "
                f"{self._world_size} != {world_size}."
            )
        if self.feed.expected_samples % world_size != 0:
            raise ValueError(
                "streaming expected_samples must be divisible by the DDP "
                f"world size: {self.feed.expected_samples} % {world_size} != 0."
            )
        per_rank = self.feed.expected_samples // world_size
        return (per_rank + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Sample]:
        rank, world_size = self._rank_world()
        _validate_rank_world(rank, world_size)
        if self._world_size is not None and self._world_size != world_size:
            raise ValueError(
                "streaming resume requires the same DDP world size: "
                f"{self._world_size} != {world_size}."
            )
        self._world_size = world_size
        if self.feed.expected_samples % world_size != 0:
            raise ValueError(
                "streaming expected_samples must be divisible by the DDP "
                f"world size: {self.feed.expected_samples} % {world_size} != 0."
            )
        if self._read_position % world_size != 0:
            raise ValueError("streaming cursor is not aligned to the DDP world size.")
        status = self.feed.status()
        self._validate_checkpoint_catalog(status.catalog)
        next_status = 0.0
        try:
            while True:
                self._check_stop_requested()
                catalog = status.catalog
                next_position = self._read_position + world_size
                final_group = next_position == self.feed.expected_samples
                if (
                    next_position <= catalog.sample_count
                    and (not final_group or status.seal is not None)
                ):
                    self._finish_wait()
                    position = self._read_position + rank
                    _snapshot, _offset, sample = self.feed.sample_at(position, catalog)
                    self._read_position = next_position
                    yield sample
                    continue
                if status.seal is not None:
                    self._finish_wait()
                    if self._read_position == self.feed.expected_samples:
                        return
                    raise RuntimeError(
                        "streaming cursor cannot assign the sealed tail without repeats."
                    )
                self._start_wait()
                now = time.monotonic()
                if now >= next_status:
                    waiting_since = self._waiting_since
                    if waiting_since is None:
                        raise RuntimeError("streaming wait timer was not started.")
                    _LOGGER.info(
                        "waiting for streaming synthesis: stream=%s read=%d committed=%d "
                        "published=%d expected=%d wait_seconds=%.3f wait_events=%d polls=%d",
                        self.feed.stream_id,
                        self._read_position,
                        self._committed_position,
                        catalog.sample_count,
                        self.feed.expected_samples,
                        self._wait_seconds + (time.perf_counter() - waiting_since),
                        self._wait_events,
                        self._poll_count,
                    )
                    next_status = now + self.status_seconds
                self._interruptible_sleep()
                self._check_stop_requested()
                self._poll_count += 1
                status = self.feed.status()
        except BaseException:
            self._finish_wait()
            raise

    def _start_wait(self) -> None:
        if self._waiting_since is None:
            self._waiting_since = time.perf_counter()

    def _interruptible_sleep(self) -> None:
        if self._stop_requested is None:
            time.sleep(self.poll_seconds)
            return
        deadline = time.monotonic() + self.poll_seconds
        while True:
            self._check_stop_requested()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.5))

    def _check_stop_requested(self) -> None:
        requested = self._stop_requested
        if requested is not None and requested():
            raise SIGTERMException()

    def _finish_wait(self) -> None:
        started = self._waiting_since
        if started is None:
            return
        elapsed = time.perf_counter() - started
        if elapsed < 0 or not math.isfinite(elapsed):
            raise RuntimeError("streaming wait timer returned an invalid duration.")
        self._wait_seconds += elapsed
        self._wait_events += 1
        self._waiting_since = None

    def commit(self, span: BatchSpan, *, batches: int) -> None:
        if span.start != self._committed_position:
            raise RuntimeError(
                "streaming committed batches are not contiguous: "
                f"{span.start} != {self._committed_position}."
            )
        if span.stop > self._read_position:
            raise RuntimeError("streaming commit is beyond delivered samples.")
        if span.stop <= span.start:
            raise ValueError("streaming commit span must be non-empty.")
        self._committed_position = span.stop
        self._committed_batches += _positive_int(batches, "committed batches")

    def state_dict(self) -> dict[str, object]:
        catalog = self.feed.status().catalog
        sequence, offset = self.feed.cursor(self._committed_position, catalog)
        next_snapshot = (
            catalog.snapshots[sequence] if sequence < len(catalog.snapshots) else None
        )
        return {
            "schema": _CURSOR_SCHEMA,
            "stream_id": self.feed.stream_id,
            "expected_samples": self.feed.expected_samples,
            "input_codec": self.feed.input_codec,
            "codec": self.feed.codec,
            "batch_size": self.batch_size,
            "world_size": self._world_size,
            "committed_position": self._committed_position,
            "committed_batches": self._committed_batches,
            "next_snapshot_sequence": sequence,
            "next_sample_offset": offset,
            "next_snapshot_id": (
                None if next_snapshot is None else next_snapshot.snapshot_id
            ),
            "next_snapshot_manifest_sha256": (
                None if next_snapshot is None else next_snapshot.manifest_sha256
            ),
            "consumed_catalog_sha256": catalog.prefix_sha256[sequence],
            "wait_seconds": self._wait_seconds,
            "wait_events": self._wait_events,
            "poll_count": self._poll_count,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("streaming cursor state must be a mapping.")
        if state.get("schema") != _CURSOR_SCHEMA:
            raise ValueError("streaming cursor checkpoint schema is incompatible.")
        if state.get("stream_id") != self.feed.stream_id:
            raise ValueError("streaming cursor checkpoint belongs to another stream.")
        if state.get("expected_samples") != self.feed.expected_samples:
            raise ValueError(
                "streaming cursor checkpoint expected_samples does not match the run."
            )
        checkpoint_input_codec = state.get("input_codec")
        if (
            "input_codec" not in state
            and self.feed.input_codec == self.feed.codec
        ):
            checkpoint_input_codec = state.get("codec")
        if checkpoint_input_codec != self.feed.input_codec:
            raise ValueError(
                "streaming cursor checkpoint input_codec does not match the run."
            )
        if state.get("codec") != self.feed.codec:
            raise ValueError(
                "streaming cursor checkpoint output codec does not match the run."
            )
        if state.get("batch_size") != self.batch_size:
            raise ValueError("streaming cursor checkpoint batch_size does not match.")
        world_size = state.get("world_size")
        if world_size is not None and (type(world_size) is not int or world_size < 1):
            raise TypeError("streaming cursor world_size must be a positive integer.")
        committed = _state_int(state, "committed_position")
        if committed > self.feed.expected_samples:
            raise ValueError("streaming cursor is beyond the logical epoch.")
        self._committed_position = committed
        self._read_position = committed
        self._committed_batches = _state_int(state, "committed_batches")
        self._world_size = world_size
        self._waiting_since = None
        self._wait_seconds = _state_float(state, "wait_seconds", default=0.0)
        self._wait_events = _state_int(state, "wait_events", default=0)
        self._poll_count = _state_int(state, "poll_count", default=0)
        self._checkpoint_catalog = {
            name: state.get(name)
            for name in (
                "next_snapshot_sequence",
                "next_sample_offset",
                "next_snapshot_id",
                "next_snapshot_manifest_sha256",
                "consumed_catalog_sha256",
            )
        }

    def _validate_checkpoint_catalog(self, catalog: Catalog) -> None:
        state = self._checkpoint_catalog
        if state is None:
            return
        sequence = _state_int(state, "next_snapshot_sequence")
        offset = _state_int(state, "next_sample_offset")
        if sequence > len(catalog.snapshots):
            raise RuntimeError("streaming checkpoint references missing snapshots.")
        if catalog.prefix_sha256[sequence] != state.get("consumed_catalog_sha256"):
            raise RuntimeError("streaming snapshot catalog forked before the checkpoint.")
        expected_sequence, expected_offset = self.feed.cursor(
            self._committed_position,
            catalog,
        )
        if (sequence, offset) != (expected_sequence, expected_offset):
            raise RuntimeError("streaming checkpoint cursor does not match the catalog.")
        next_snapshot_id = state.get("next_snapshot_id")
        next_snapshot_manifest = state.get("next_snapshot_manifest_sha256")
        if next_snapshot_id is not None:
            if sequence >= len(catalog.snapshots):
                raise RuntimeError("streaming checkpoint references missing snapshots.")
            snapshot = catalog.snapshots[sequence]
            if snapshot.snapshot_id != next_snapshot_id:
                raise RuntimeError("streaming checkpoint next snapshot id changed.")
            if snapshot.manifest_sha256 != next_snapshot_manifest:
                raise RuntimeError("streaming checkpoint next snapshot manifest changed.")
        elif next_snapshot_manifest is not None:
            raise RuntimeError(
                "streaming checkpoint next snapshot identity is incomplete."
            )
        self._checkpoint_catalog = None


class StreamingDataLoader:
    """Expose delivered/pending/committed batch state to Lightning callbacks."""

    def __init__(
        self,
        dataset: StreamingSnapshotDataset,
        *,
        collate_fn: Callable[[list[Sample]], Any],
        pin_memory: bool,
    ) -> None:
        self.dataset = dataset
        self.loader = DataLoader(
            dataset,
            batch_size=dataset.batch_size,
            num_workers=0,
            pin_memory=pin_memory,
            persistent_workers=False,
            collate_fn=collate_fn,
        )
        self._delivered: deque[BatchSpan] = deque()
        self._pending: list[BatchSpan] = []
        self._last_global_step = 0
        self._batch_fetch_seconds = 0.0
        self._batch_wait_seconds = 0.0
        self._batch_load_seconds = 0.0
        self._total_fetch_seconds = 0.0
        self._total_load_seconds = 0.0

    def __len__(self) -> int:
        return self.dataset.logical_batch_count()

    def __iter__(self) -> Iterator[Any]:
        iterator = iter(self.loader)
        while True:
            start = self.dataset.read_position
            wait_before = self.dataset.wait_seconds
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            fetch_seconds = time.perf_counter() - started
            wait_seconds = self.dataset.wait_seconds - wait_before
            if wait_seconds < 0 or not math.isfinite(wait_seconds):
                raise RuntimeError("streaming batch wait timer moved backwards.")
            load_seconds = max(0.0, fetch_seconds - wait_seconds)
            self._batch_fetch_seconds = fetch_seconds
            self._batch_wait_seconds = wait_seconds
            self._batch_load_seconds = load_seconds
            self._total_fetch_seconds += fetch_seconds
            self._total_load_seconds += load_seconds
            stop = self.dataset.read_position
            if stop <= start:
                raise RuntimeError("streaming DataLoader delivered an empty cursor span.")
            self._delivered.append(BatchSpan(start, stop))
            yield batch

    def telemetry(self) -> StreamingTelemetry:
        status = self.dataset.feed.status()
        return StreamingTelemetry(
            batch_fetch_seconds=self._batch_fetch_seconds,
            batch_wait_seconds=self._batch_wait_seconds,
            batch_load_seconds=self._batch_load_seconds,
            total_fetch_seconds=self._total_fetch_seconds,
            total_wait_seconds=self.dataset.wait_seconds,
            total_load_seconds=self._total_load_seconds,
            wait_events=self.dataset.wait_events,
            poll_count=self.dataset.poll_count,
            read_position=self.dataset.read_position,
            committed_position=self.dataset.committed_position,
            committed_batches=self.dataset.committed_batches,
            published_samples=status.catalog.sample_count,
            expected_samples=self.dataset.feed.expected_samples,
        )

    def set_global_step(self, step: int) -> None:
        if type(step) is not int or step < 0:
            raise ValueError("streaming global step must be a non-negative integer.")
        self._last_global_step = step

    def acknowledge(self, global_step: int) -> None:
        if type(global_step) is not int or global_step < 0:
            raise ValueError("streaming global step must be a non-negative integer.")
        if not self._delivered:
            raise RuntimeError("streaming batch acknowledgement has no delivered batch.")
        self._pending.append(self._delivered.popleft())
        if global_step <= self._last_global_step:
            return
        span = BatchSpan(self._pending[0].start, self._pending[-1].stop)
        self.dataset.commit(span, batches=len(self._pending))
        self._pending.clear()
        self._last_global_step = global_step

    def state_dict(self) -> dict[str, object]:
        return {
            "schema": _LOADER_SCHEMA,
            "last_global_step": self._last_global_step,
            "total_fetch_seconds": self._total_fetch_seconds,
            "total_load_seconds": self._total_load_seconds,
            "dataset": self.dataset.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("streaming loader state must be a mapping.")
        if state.get("schema") != _LOADER_SCHEMA:
            raise ValueError("streaming loader checkpoint schema is incompatible.")
        dataset = state.get("dataset")
        if not isinstance(dataset, Mapping):
            raise TypeError("streaming loader dataset state must be a mapping.")
        self.dataset.load_state_dict(cast(Mapping[str, object], dataset))
        self._last_global_step = _state_int(state, "last_global_step")
        self._batch_fetch_seconds = 0.0
        self._batch_wait_seconds = 0.0
        self._batch_load_seconds = 0.0
        self._total_fetch_seconds = _state_float(state, "total_fetch_seconds", default=0.0)
        self._total_load_seconds = _state_float(state, "total_load_seconds", default=0.0)
        self._delivered.clear()
        self._pending.clear()


def synthesis_controller(
    factory_path: str,
    request: SynthesisRequest,
) -> SynthesisController:
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("synthesis factory must use 'module:attribute' syntax.")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"synthesis factory {factory_path!r} must be callable.")
    controller = cast(SynthesisFactory, factory)(request)
    for method in ("start", "check", "close"):
        if not callable(getattr(controller, method, None)):
            raise TypeError(
                f"synthesis controller from {factory_path!r} is missing {method}()."
            )
    return controller


def _snapshot(
    path: Path,
    *,
    stream_id: str,
    expected: int,
    codec: str | None,
    input_codec: str | None,
) -> Snapshot:
    payload, digest = _manifest(
        path,
        schema=(_SNAPSHOT_SCHEMA_V1, _SNAPSHOT_SCHEMA),
    )
    _identity(
        payload,
        path=path,
        stream_id=stream_id,
        expected=expected,
        codec=codec,
        input_codec=input_codec,
    )
    sequence = _manifest_int(payload, "sequence", path=path, positive=False)
    snapshot_id = _manifest_string(payload, "snapshot_id", path=path)
    raw_indices = payload.get("sample_indices")
    if not isinstance(raw_indices, list) or not raw_indices:
        raise TypeError(
            f"streaming snapshot sample_indices must be a non-empty list: {path}."
        )
    indices: list[int] = []
    seen: set[int] = set()
    for position, value in enumerate(raw_indices):
        if type(value) is not int:
            raise TypeError(
                f"streaming snapshot sample_indices[{position}] must be an integer."
            )
        if value < 0 or value >= expected:
            raise ValueError(
                f"streaming snapshot sample index {value} is outside 0..{expected - 1}."
            )
        if value in seen:
            raise ValueError(f"streaming snapshot repeats sample index {value}.")
        seen.add(value)
        indices.append(value)
    declared = _manifest_int(payload, "sample_count", path=path, positive=True)
    if declared != len(indices):
        raise ValueError(
            f"streaming snapshot sample_count mismatch at {path}: "
            f"{declared} != {len(indices)}."
        )
    references_sha256 = None
    if payload.get("schema") == _SNAPSHOT_SCHEMA:
        references_sha256 = _translation_reference_digest(payload, path=path)
    return Snapshot(
        root=path.parent,
        snapshot_id=snapshot_id,
        sequence=sequence,
        sample_indices=tuple(indices),
        manifest_sha256=digest,
        translation_references_sha256=references_sha256,
    )


def _catalog(snapshots: tuple[Snapshot, ...], *, expected: int) -> Catalog:
    locations: dict[int, tuple[int, int]] = {}
    cumulative: list[int] = []
    prefix: list[str] = [_digest_json([])]
    identity: list[dict[str, object]] = []
    total = 0
    seen_ids: set[str] = set()
    for sequence, snapshot in enumerate(snapshots):
        if snapshot.sequence != sequence:
            raise ValueError(
                "streaming snapshot sequences must be contiguous from zero."
            )
        if snapshot.snapshot_id in seen_ids:
            raise ValueError("streaming snapshot ids must be unique.")
        seen_ids.add(snapshot.snapshot_id)
        for offset, index in enumerate(snapshot.sample_indices):
            previous = locations.get(index)
            if previous is not None:
                raise ValueError(
                    "streaming snapshots must not overlap sample membership: "
                    f"index {index} is in snapshots {previous[0]} and {sequence}."
                )
            locations[index] = (sequence, offset)
        total += snapshot.sample_count
        if total > expected:
            raise ValueError("streaming snapshots exceed expected_samples.")
        cumulative.append(total)
        identity.append(
            {
                "sequence": sequence,
                "snapshot_id": snapshot.snapshot_id,
                "manifest_sha256": snapshot.manifest_sha256,
                "sample_count": snapshot.sample_count,
            }
        )
        prefix.append(_digest_json(identity))
    return Catalog(
        snapshots=snapshots,
        cumulative_counts=tuple(cumulative),
        locations=locations,
        prefix_sha256=tuple(prefix),
        sha256=_digest_json(identity),
    )


def _manifest(
    path: Path,
    *,
    schema: str | tuple[str, ...],
) -> tuple[Mapping[str, object], str]:
    data = path.read_bytes()
    try:
        value = json.loads(data)
    except json.JSONDecodeError as error:
        raise ValueError(f"streaming manifest is invalid JSON: {path}.") from error
    if not isinstance(value, Mapping):
        raise TypeError(f"streaming manifest must contain one object: {path}.")
    schemas = (schema,) if isinstance(schema, str) else schema
    if value.get("schema") not in schemas:
        raise ValueError(f"streaming manifest schema is incompatible: {path}.")
    return cast(Mapping[str, object], value), hashlib.sha256(data).hexdigest()


def _translation_reference_digest(
    payload: Mapping[str, object],
    *,
    path: Path,
) -> str:
    raw = payload.get("translation_references")
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"streaming snapshot translation_references must be an object: {path}."
        )
    values = cast(Mapping[str, object], raw)
    fields = set(values)
    expected_fields = {"schema", "file", "sha256"}
    if fields != expected_fields:
        raise ValueError(
            "streaming snapshot translation_references must contain exactly "
            f"{sorted(expected_fields)}: {path}."
        )
    if values.get("schema") != _TRANSLATION_REFERENCE_SCHEMA:
        raise ValueError(
            f"streaming translation reference schema is incompatible: {path}."
        )
    if values.get("file") != _TRANSLATION_REFERENCE_FILE:
        raise ValueError(
            f"streaming translation reference file is incompatible: {path}."
        )
    return _manifest_digest(values, "sha256", path=path)


def _translation_reference_texts(
    data: bytes,
    *,
    indices: tuple[int, ...],
    path: Path,
) -> tuple[str, ...]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"streaming translation reference sidecar is not UTF-8: {path}."
        ) from error
    lines = text.splitlines()
    if len(lines) != len(indices):
        raise ValueError(
            "streaming translation reference sidecar sample count mismatch at "
            f"{path}: {len(lines)} != {len(indices)}."
        )
    references: list[str] = []
    for position, (line, expected_index) in enumerate(zip(lines, indices)):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                "streaming translation reference sidecar contains invalid JSON at "
                f"line {position + 1}: {path}."
            ) from error
        if not isinstance(raw, Mapping) or set(raw) != {"sample_index", "text"}:
            raise ValueError(
                "streaming translation reference records must contain exactly "
                f"sample_index and text at line {position + 1}: {path}."
            )
        sample_index = raw.get("sample_index")
        if sample_index != expected_index or type(sample_index) is not int:
            raise ValueError(
                "streaming translation reference index is not aligned with the "
                f"snapshot at line {position + 1}: {path}."
            )
        reference = raw.get("text")
        if not isinstance(reference, str) or not reference:
            raise ValueError(
                "streaming translation reference text must be non-empty at "
                f"line {position + 1}: {path}."
            )
        references.append(reference)
    return tuple(references)


def _identity(
    payload: Mapping[str, object],
    *,
    path: Path,
    stream_id: str,
    expected: int,
    codec: str | None,
    input_codec: str | None,
) -> None:
    if _manifest_string(payload, "stream_id", path=path) != stream_id:
        raise ValueError(f"streaming manifest belongs to another stream: {path}.")
    actual = _manifest_int(payload, "expected_samples", path=path, positive=True)
    if actual != expected:
        raise ValueError(
            f"streaming manifest expected_samples mismatch at {path}: {actual} != {expected}."
        )
    if codec is None:
        return
    if "input_codec" not in payload and input_codec == codec:
        actual_input = codec
    else:
        actual_input = _manifest_string(payload, "input_codec", path=path)
    if actual_input != input_codec:
        raise ValueError(
            "streaming manifest input_codec mismatch at "
            f"{path}: {actual_input!r} != {input_codec!r}."
        )
    actual_output = _manifest_string(payload, "codec", path=path)
    if actual_output != codec:
        raise ValueError(
            "streaming manifest output codec mismatch at "
            f"{path}: {actual_output!r} != {codec!r}."
        )


def _manifest_string(payload: Mapping[str, object], name: str, *, path: Path) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise TypeError(f"streaming manifest {name} must be a non-empty string: {path}.")
    return value


def _manifest_digest(payload: Mapping[str, object], name: str, *, path: Path) -> str:
    value = _manifest_string(payload, name, path=path)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"streaming manifest {name} must be a lowercase SHA256: {path}.")
    return value


def _manifest_int(
    payload: Mapping[str, object],
    name: str,
    *,
    path: Path,
    positive: bool,
) -> int:
    value = payload.get(name)
    if type(value) is not int:
        raise TypeError(f"streaming manifest {name} must be an integer: {path}.")
    minimum = 1 if positive else 0
    if value < minimum:
        relation = "positive" if positive else "non-negative"
        raise ValueError(f"streaming manifest {name} must be {relation}: {path}.")
    return value


def _state_int(
    state: Mapping[str, object],
    name: str,
    *,
    default: int | None = None,
) -> int:
    value = state.get(name, default)
    if value is None and default is not None:
        return default
    if type(value) is not int:
        raise TypeError(f"streaming cursor {name} must be an integer.")
    if value < 0:
        raise ValueError(f"streaming cursor {name} must be non-negative.")
    return value


def _state_float(
    state: Mapping[str, object],
    name: str,
    *,
    default: float | None = None,
) -> float:
    value = state.get(name, default)
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"streaming cursor {name} must be numeric.")
    result = float(value)
    if result < 0 or not math.isfinite(result):
        raise ValueError(f"streaming cursor {name} must be finite and non-negative.")
    return result


def _digest_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _distributed_rank_world() -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def _validate_rank_world(rank: int, world_size: int) -> None:
    if type(rank) is not int or type(world_size) is not int:
        raise TypeError("streaming rank and world size must be integers.")
    if world_size < 1:
        raise ValueError("streaming world size must be positive.")
    if rank < 0 or rank >= world_size:
        raise ValueError("streaming rank must be inside the world size.")


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be numeric.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return float(value)


__all__ = [
    "PublishedSample",
    "SnapshotFeed",
    "StreamingDataLoader",
    "StreamingSnapshotDataset",
    "StreamingTelemetry",
    "SynthesisController",
    "SynthesisRequest",
    "WorkspaceSnapshotLoader",
    "synthesis_controller",
    "workspace_stream_root",
]
