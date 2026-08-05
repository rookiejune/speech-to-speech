from __future__ import annotations

import json
import unittest
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch

from anydataset.types import Sample
from lightning.pytorch.utilities.exceptions import SIGTERMException
from torch.utils.data import Dataset

from speech_to_speech.datamodule.streaming import (
    BatchSpan,
    SnapshotFeed,
    StreamingDataLoader,
    StreamingSnapshotDataset,
)


_SNAPSHOT_SCHEMA = "speech-to-speech-stream-snapshot-v1"
_SEAL_SCHEMA = "speech-to-speech-stream-seal-v1"
_FAILURE_SCHEMA = "speech-to-speech-stream-failure-v1"
_STREAM_ID = "wmt19-bidirectional-v1"


class _Samples(Dataset[Sample]):
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Sample:
        return cast(Sample, cast(object, {"index": self.indices[index]}))


def _load_snapshot(root: Path) -> Dataset[Sample]:
    payload = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("test snapshot payload must be an object")
    indices = payload.get("sample_indices")
    if not isinstance(indices, list) or any(type(index) is not int for index in indices):
        raise TypeError("test snapshot indices must be integers")
    return _Samples(cast(list[int], indices))


def _feed(
    root: Path,
    *,
    expected: int,
    stream_id: str = _STREAM_ID,
    codec: str | None = None,
    input_codec: str | None = None,
) -> SnapshotFeed:
    return SnapshotFeed(
        root,
        stream_id=stream_id,
        expected_samples=expected,
        loader=_load_snapshot,
        codec=codec,
        input_codec=input_codec,
    )


def _dataset(
    feed: SnapshotFeed,
    *,
    batch_size: int = 2,
    rank: int = 0,
    world_size: int = 1,
) -> StreamingSnapshotDataset:
    return StreamingSnapshotDataset(
        feed,
        batch_size=batch_size,
        poll_seconds=0.001,
        status_seconds=60.0,
        rank_world=lambda: (rank, world_size),
    )


def _loader(dataset: StreamingSnapshotDataset) -> StreamingDataLoader:
    return StreamingDataLoader(
        dataset,
        collate_fn=_identity_batch,
        pin_memory=False,
    )


def _identity_batch(samples: list[Sample]) -> list[Sample]:
    return samples


def _sample_index(sample: Sample) -> int:
    index = cast(Mapping[str, object], cast(object, sample)).get("index")
    if type(index) is not int:
        raise TypeError("test sample has no integer index")
    return index


def _batch_indices(batch: list[Sample]) -> list[int]:
    return [_sample_index(sample) for sample in batch]


def _write_snapshot(
    root: Path,
    sequence: int,
    snapshot_id: str,
    indices: list[int],
    *,
    expected: int,
    stream_id: str = _STREAM_ID,
    revision: str | None = None,
    codec: str | None = None,
    input_codec: str | None = None,
) -> Path:
    directory = root / "snapshots" / f"{sequence:06d}-{snapshot_id}"
    directory.mkdir(parents=True)
    payload: dict[str, object] = {
        "schema": _SNAPSHOT_SCHEMA,
        "stream_id": stream_id,
        "expected_samples": expected,
        "sequence": sequence,
        "snapshot_id": snapshot_id,
        "sample_indices": indices,
        "sample_count": len(indices),
    }
    if codec is not None:
        payload["codec"] = codec
        payload["input_codec"] = codec if input_codec is None else input_codec
    if revision is not None:
        payload["revision"] = revision
    path = directory / "snapshot.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    return path


def _write_seal(
    root: Path,
    feed: SnapshotFeed,
    *,
    sample_count: int | None = None,
    snapshot_count: int | None = None,
    catalog_sha256: str | None = None,
) -> None:
    catalog = feed.status().catalog
    payload = {
        "schema": _SEAL_SCHEMA,
        "stream_id": feed.stream_id,
        "expected_samples": feed.expected_samples,
        "snapshot_count": (
            len(catalog.snapshots) if snapshot_count is None else snapshot_count
        ),
        "sample_count": feed.expected_samples if sample_count is None else sample_count,
        "catalog_sha256": catalog.sha256 if catalog_sha256 is None else catalog_sha256,
    }
    if feed.codec is not None:
        payload["codec"] = feed.codec
        payload["input_codec"] = feed.input_codec
    (root / "sealed.json").write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def _write_failure(root: Path, *, expected: int, error: str = "producer crashed") -> None:
    payload = {
        "schema": _FAILURE_SCHEMA,
        "stream_id": _STREAM_ID,
        "expected_samples": expected,
        "error": error,
        "exit_code": 17,
    }
    (root / "failed.json").write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


class SnapshotFeedTest(unittest.TestCase):
    def test_hidden_staging_snapshot_is_not_discoverable_before_rename(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            staged = _write_snapshot(root, 0, "chunk-a", [0, 1], expected=2)
            hidden = staged.parent.with_name(f".{staged.parent.name}.staging")
            staged.parent.rename(hidden)
            feed = _feed(root, expected=2)

            self.assertEqual(feed.status().catalog.sample_count, 0)
            visible = hidden.with_name("000000-chunk-a")
            hidden.rename(visible)
            self.assertEqual(feed.status().catalog.sample_count, 2)

    def test_status_rescans_catalog_after_observing_a_new_seal(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 0, "chunk-a", [0], expected=2)
            feed = _feed(root, expected=2)
            original_seal = feed._seal
            published = False

            def publish_final_snapshot():
                nonlocal published
                if not published:
                    _write_snapshot(root, 1, "chunk-b", [1], expected=2)
                    complete = _feed(root, expected=2)
                    _write_seal(root, complete)
                    published = True
                return original_seal()

            with patch.object(
                feed,
                "_seal",
                side_effect=publish_final_snapshot,
            ):
                status = feed.status()

        self.assertIsNotNone(status.seal)
        self.assertEqual(status.catalog.sample_count, 2)
        self.assertEqual(len(status.catalog.snapshots), 2)

    def test_catalog_tracks_non_overlapping_chunks_in_publication_order(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 1, "chunk-b", [2, 1], expected=4)
            _write_snapshot(root, 0, "chunk-a", [3, 0], expected=4)
            feed = _feed(root, expected=4)

            catalog = feed.status().catalog
            first, first_offset, first_sample = feed.sample_at(0, catalog)
            second, second_offset, second_sample = feed.sample_at(2, catalog)
            published = feed.published([1, 3])

        self.assertEqual(
            [snapshot.snapshot_id for snapshot in catalog.snapshots],
            ["chunk-a", "chunk-b"],
        )
        self.assertEqual(catalog.cumulative_counts, (2, 4))
        self.assertEqual(
            catalog.locations,
            {3: (0, 0), 0: (0, 1), 2: (1, 0), 1: (1, 1)},
        )
        self.assertEqual((first.snapshot_id, first_offset, _sample_index(first_sample)), ("chunk-a", 0, 3))
        self.assertEqual((second.snapshot_id, second_offset, _sample_index(second_sample)), ("chunk-b", 0, 2))
        self.assertEqual(
            [(item.index, item.snapshot_id, _sample_index(item.sample)) for item in published],
            [(1, "chunk-b", 1), (3, "chunk-a", 3)],
        )

    def test_catalog_rejects_overlapping_chunk_membership(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 0, "chunk-a", [0, 1], expected=4)
            _write_snapshot(root, 1, "chunk-b", [1, 2], expected=4)

            with self.assertRaisesRegex(ValueError, "must not overlap.*index 1"):
                _feed(root, expected=4).status()

    def test_seal_accepts_exact_coverage_and_rejects_incomplete_catalog(self) -> None:
        with TemporaryDirectory() as complete_directory:
            complete = Path(complete_directory)
            _write_snapshot(complete, 0, "chunk-a", [3, 0], expected=4)
            _write_snapshot(complete, 1, "chunk-b", [2, 1], expected=4)
            complete_feed = _feed(complete, expected=4)
            _write_seal(complete, complete_feed)

            status = complete_feed.status()

            self.assertIsNotNone(status.seal)
            self.assertEqual(set(status.catalog.locations), set(range(4)))

        with TemporaryDirectory() as incomplete_directory:
            incomplete = Path(incomplete_directory)
            _write_snapshot(incomplete, 0, "chunk-a", [0, 2], expected=4)
            incomplete_feed = _feed(incomplete, expected=4)
            _write_seal(incomplete, incomplete_feed)

            with self.assertRaisesRegex(RuntimeError, "sample count differs"):
                incomplete_feed.status()

    def test_manifest_identity_mismatch_fails_before_loading_samples(self) -> None:
        cases = (
            ("other-stream", 4, "another stream"),
            (_STREAM_ID, 6, "expected_samples mismatch"),
        )
        for stream_id, manifest_expected, message in cases:
            with self.subTest(message=message), TemporaryDirectory() as directory:
                root = Path(directory)
                _write_snapshot(
                    root,
                    0,
                    "chunk-a",
                    [0, 1],
                    expected=manifest_expected,
                    stream_id=stream_id,
                )

                with self.assertRaisesRegex(ValueError, message):
                    _feed(root, expected=4).status()

    def test_manifest_and_cursor_bind_input_and_output_codecs(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(
                root,
                0,
                "chunk-a",
                [0, 1],
                expected=2,
                codec="bicodec",
                input_codec="glm4",
            )
            feed = _feed(
                root,
                expected=2,
                codec="bicodec",
                input_codec="glm4",
            )
            state = _dataset(feed).state_dict()

            self.assertEqual(state["input_codec"], "glm4")
            self.assertEqual(state["codec"], "bicodec")
            missing_input = dict(state)
            missing_input.pop("input_codec")
            with self.assertRaisesRegex(ValueError, "input_codec"):
                _dataset(feed).load_state_dict(missing_input)
            with self.assertRaisesRegex(ValueError, "input_codec"):
                _dataset(
                    _feed(
                        root,
                        expected=2,
                        codec="bicodec",
                        input_codec="longcat",
                    )
                ).load_state_dict(state)
            with self.assertRaisesRegex(ValueError, "output codec"):
                _dataset(
                    _feed(
                        root,
                        expected=2,
                        codec="longcat",
                        input_codec="glm4",
                    )
                ).load_state_dict(state)

    def test_coupled_v1_manifests_and_cursor_default_input_to_output_codec(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot_path = _write_snapshot(
                root,
                0,
                "chunk-a",
                [0, 1],
                expected=2,
                codec="longcat",
            )
            snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            snapshot_payload.pop("input_codec")
            snapshot_path.write_text(
                json.dumps(
                    snapshot_payload,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            legacy_feed = _feed(root, expected=2, codec="longcat")
            _write_seal(root, legacy_feed)
            seal_path = root / "sealed.json"
            seal_payload = json.loads(seal_path.read_text(encoding="utf-8"))
            seal_payload.pop("input_codec")
            seal_path.write_text(
                json.dumps(
                    seal_payload,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            restored_feed = _feed(root, expected=2, codec="longcat")
            status = restored_feed.status()
            state = _dataset(restored_feed).state_dict()
            state.pop("input_codec")
            restored = _dataset(_feed(root, expected=2, codec="longcat"))
            restored.load_state_dict(state)

        self.assertIsNotNone(status.seal)
        self.assertEqual(status.catalog.sample_count, 2)

    def test_unsealed_failure_marker_aborts_polling_but_sealed_data_wins(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 0, "chunk-a", [0, 1], expected=2)
            feed = _feed(root, expected=2)
            _write_failure(root, expected=2)

            with self.assertRaisesRegex(
                RuntimeError,
                "producer failed.*exit code 17.*producer crashed",
            ):
                feed.status()

            (root / "failed.json").unlink()
            _write_seal(root, feed)
            _write_failure(root, expected=2, error="late monitor race")

            self.assertIsNotNone(feed.status().seal)

    def test_live_catalog_and_seal_are_append_only(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 0, "chunk-a", [0, 1], expected=4)
            second = _write_snapshot(root, 1, "chunk-b", [2, 3], expected=4)
            feed = _feed(root, expected=4)
            feed.status()

            second.unlink()
            with self.assertRaisesRegex(RuntimeError, "catalog shrank"):
                feed.status()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 0, "chunk-a", [0, 1], expected=2)
            feed = _feed(root, expected=2)
            _write_seal(root, feed)
            feed.status()

            (root / "sealed.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "seal disappeared"):
                feed.status()


class StreamingSnapshotDatasetTest(unittest.TestCase):
    def test_stop_request_interrupts_snapshot_polling(self) -> None:
        with TemporaryDirectory() as directory:
            feed = _feed(Path(directory), expected=4)
            dataset = _dataset(feed)
            requested = False

            def stop() -> bool:
                return requested

            def request_stop(_seconds: float) -> None:
                nonlocal requested
                requested = True

            dataset.set_stop_requested(stop)
            with patch(
                "speech_to_speech.datamodule.streaming.time.sleep",
                side_effect=request_stop,
            ), self.assertRaises(SIGTERMException):
                next(iter(dataset))

        self.assertEqual(dataset.wait_events, 1)
        self.assertEqual(dataset.poll_count, 0)

    def test_iterator_waits_for_new_chunk_and_for_seal(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 0, "chunk-a", [0, 1], expected=4)
            feed = _feed(root, expected=4)
            dataset = _dataset(feed)
            sleep_calls = 0

            def publish(_seconds: float) -> None:
                nonlocal sleep_calls
                sleep_calls += 1
                if sleep_calls == 1:
                    _write_snapshot(root, 1, "chunk-b", [2, 3], expected=4)
                    return
                if sleep_calls == 2:
                    self.assertEqual(dataset.read_position, 3)
                    _write_seal(root, feed)
                    return
                raise AssertionError("streaming iterator waited after the seal")

            iterator = iter(dataset)
            with patch(
                "speech_to_speech.datamodule.streaming.time.sleep",
                side_effect=publish,
            ), patch(
                "speech_to_speech.datamodule.streaming.time.perf_counter",
                side_effect=[10.0, 11.0, 12.0, 20.0, 23.0],
            ):
                actual = [_sample_index(next(iterator)) for _ in range(4)]
                with self.assertRaises(StopIteration):
                    next(iterator)

        self.assertEqual(actual, [0, 1, 2, 3])
        self.assertEqual(sleep_calls, 2)
        self.assertEqual(dataset.wait_events, 2)
        self.assertEqual(dataset.poll_count, 2)
        self.assertEqual(dataset.wait_seconds, 5.0)

    def test_two_ddp_ranks_partition_global_publication_positions(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 0, "chunk-a", [4, 0, 3], expected=6)
            _write_snapshot(root, 1, "chunk-b", [1, 5, 2], expected=6)
            feed = _feed(root, expected=6)
            _write_seal(root, feed)

            rank_zero = [
                _sample_index(sample)
                for sample in _dataset(_feed(root, expected=6), rank=0, world_size=2)
            ]
            rank_one = [
                _sample_index(sample)
                for sample in _dataset(_feed(root, expected=6), rank=1, world_size=2)
            ]

        self.assertEqual(rank_zero, [4, 3, 5])
        self.assertEqual(rank_one, [0, 1, 2])
        self.assertEqual(set(rank_zero) & set(rank_one), set())
        self.assertEqual(set(rank_zero) | set(rank_one), set(range(6)))

    def test_world_size_divisibility_fails_before_consuming_a_long_run(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 0, "chunk-a", [0, 1, 2], expected=3)
            feed = _feed(root, expected=3)
            _write_seal(root, feed)
            dataset = _dataset(feed, rank=0, world_size=2)

            with self.assertRaisesRegex(ValueError, "3 % 2"):
                next(iter(dataset))


class StreamingDataLoaderTest(unittest.TestCase):
    def test_batch_timing_is_checkpointed_without_replaying_the_last_batch(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 0, "chunk-a", [0, 1, 2, 3], expected=4)
            feed = _feed(root, expected=4)
            _write_seal(root, feed)
            loader = _loader(_dataset(feed))

            with patch(
                "speech_to_speech.datamodule.streaming.time.perf_counter",
                side_effect=[10.0, 10.5],
            ):
                self.assertEqual(_batch_indices(next(iter(loader))), [0, 1])

            telemetry = loader.telemetry()
            self.assertEqual(telemetry.batch_fetch_seconds, 0.5)
            self.assertEqual(telemetry.batch_wait_seconds, 0.0)
            self.assertEqual(telemetry.batch_load_seconds, 0.5)

            state = loader.state_dict()
            dataset_state = cast(dict[str, object], state["dataset"])
            self.assertEqual(dataset_state["wait_seconds"], 0.0)
            self.assertEqual(dataset_state["wait_events"], 0)
            self.assertEqual(dataset_state["poll_count"], 0)
            dataset_state["wait_seconds"] = 2.5
            dataset_state["wait_events"] = 2
            dataset_state["poll_count"] = 7
            restored = _loader(_dataset(_feed(root, expected=4)))
            restored.load_state_dict(state)
            resumed = restored.telemetry()

        self.assertEqual(resumed.batch_fetch_seconds, 0.0)
        self.assertEqual(resumed.total_fetch_seconds, 0.5)
        self.assertEqual(resumed.total_wait_seconds, 2.5)
        self.assertEqual(resumed.total_load_seconds, 0.5)
        self.assertEqual(resumed.wait_events, 2)
        self.assertEqual(resumed.poll_count, 7)

    def test_length_is_the_per_rank_logical_batch_count(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)

            full_batches = _loader(
                _dataset(_feed(root, expected=8), batch_size=1, world_size=2)
            )
            partial_tail = _loader(
                _dataset(_feed(root, expected=6), batch_size=2, world_size=2)
            )

            self.assertEqual(len(full_batches), 4)
            self.assertEqual(len(partial_tail), 2)

    def test_length_rejects_a_resumed_world_size_change(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 0, "chunk-a", [0, 1, 2, 3], expected=4)
            feed = _feed(root, expected=4)
            _write_seal(root, feed)
            original = _dataset(feed, batch_size=1, rank=0, world_size=2)
            next(iter(original))

            restored = _dataset(
                _feed(root, expected=4),
                batch_size=1,
                rank=0,
                world_size=1,
            )
            restored.load_state_dict(original.state_dict())

            with self.assertRaisesRegex(ValueError, "same DDP world size"):
                len(_loader(restored))

    def test_delivered_pending_and_committed_positions_are_distinct(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 0, "chunk-a", [0, 1, 2, 3], expected=4)
            feed = _feed(root, expected=4)
            _write_seal(root, feed)
            dataset = _dataset(feed)
            loader = _loader(dataset)
            loader.set_global_step(0)
            iterator = iter(loader)

            first = next(iterator)
            self.assertEqual(_batch_indices(first), [0, 1])
            self.assertEqual(dataset.read_position, 2)
            self.assertEqual(dataset.committed_position, 0)

            loader.acknowledge(0)
            self.assertEqual(dataset.committed_position, 0)
            self.assertEqual(
                cast(Mapping[str, object], loader.state_dict()["dataset"])[
                    "committed_batches"
                ],
                0,
            )

            second = next(iterator)
            self.assertEqual(_batch_indices(second), [2, 3])
            self.assertEqual(dataset.read_position, 4)
            loader.acknowledge(1)

            self.assertEqual(dataset.committed_position, 4)
            self.assertEqual(
                cast(Mapping[str, object], loader.state_dict()["dataset"])[
                    "committed_batches"
                ],
                2,
            )

    def test_checkpoint_roundtrip_resumes_from_committed_position(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 0, "chunk-a", [0, 1, 2], expected=6)
            _write_snapshot(root, 1, "chunk-b", [3, 4, 5], expected=6)
            feed = _feed(root, expected=6)
            _write_seal(root, feed)
            loader = _loader(_dataset(feed))
            iterator = iter(loader)

            self.assertEqual(_batch_indices(next(iterator)), [0, 1])
            loader.acknowledge(1)
            state = cast(
                dict[str, object],
                json.loads(json.dumps(loader.state_dict())),
            )

            restored_dataset = _dataset(_feed(root, expected=6))
            restored = _loader(restored_dataset)
            restored.load_state_dict(state)
            remaining = [_batch_indices(batch) for batch in restored]

        self.assertEqual(restored_dataset.committed_position, 2)
        self.assertEqual(remaining, [[2, 3], [4, 5]])

    def test_uncommitted_delivered_batch_is_replayed_after_restore(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 0, "chunk-a", [0, 1, 2, 3], expected=4)
            feed = _feed(root, expected=4)
            _write_seal(root, feed)
            loader = _loader(_dataset(feed))

            delivered = next(iter(loader))
            state = loader.state_dict()
            restored = _loader(_dataset(_feed(root, expected=4)))
            restored.load_state_dict(state)
            replayed = next(iter(restored))

        self.assertEqual(_batch_indices(delivered), [0, 1])
        self.assertEqual(_batch_indices(replayed), [0, 1])


class StreamingCheckpointCatalogTest(unittest.TestCase):
    def test_checkpoint_rejects_catalog_fork_before_committed_cursor(self) -> None:
        with TemporaryDirectory() as original_directory, TemporaryDirectory() as fork_directory:
            original = Path(original_directory)
            _write_snapshot(original, 0, "chunk-a", [0, 1], expected=4)
            _write_snapshot(original, 1, "chunk-b", [2, 3], expected=4)
            original_dataset = _dataset(_feed(original, expected=4))
            original_iterator = iter(original_dataset)
            next(original_iterator)
            next(original_iterator)
            original_dataset.commit(BatchSpan(0, 2), batches=1)
            state = original_dataset.state_dict()

            fork = Path(fork_directory)
            _write_snapshot(
                fork,
                0,
                "chunk-a",
                [0, 1],
                expected=4,
                revision="rewritten",
            )
            _write_snapshot(fork, 1, "chunk-b", [2, 3], expected=4)
            restored = _dataset(_feed(fork, expected=4))
            restored.load_state_dict(state)

            with self.assertRaisesRegex(RuntimeError, "catalog forked"):
                next(iter(restored))

    def test_checkpoint_rejects_removal_of_known_next_chunk(self) -> None:
        with TemporaryDirectory() as original_directory, TemporaryDirectory() as shrink_directory:
            original = Path(original_directory)
            _write_snapshot(original, 0, "chunk-a", [0, 1], expected=4)
            _write_snapshot(original, 1, "chunk-b", [2, 3], expected=4)
            original_dataset = _dataset(_feed(original, expected=4))
            original_iterator = iter(original_dataset)
            next(original_iterator)
            next(original_iterator)
            original_dataset.commit(BatchSpan(0, 2), batches=1)
            state = original_dataset.state_dict()

            shrink = Path(shrink_directory)
            _write_snapshot(shrink, 0, "chunk-a", [0, 1], expected=4)
            restored = _dataset(_feed(shrink, expected=4))
            restored.load_state_dict(state)

            with patch(
                "speech_to_speech.datamodule.streaming.time.sleep",
                side_effect=AssertionError("shrunken catalog entered the wait loop"),
            ):
                with self.assertRaisesRegex(RuntimeError, "missing snapshots"):
                    next(iter(restored))

    def test_checkpoint_at_frontier_accepts_later_append(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 0, "chunk-a", [0, 1], expected=4)
            initial = _dataset(_feed(root, expected=4))
            initial_iterator = iter(initial)
            next(initial_iterator)
            next(initial_iterator)
            initial.commit(BatchSpan(0, 2), batches=1)
            state = initial.state_dict()

            _write_snapshot(root, 1, "chunk-b", [2, 3], expected=4)
            appended_feed = _feed(root, expected=4)
            _write_seal(root, appended_feed)
            restored = _dataset(appended_feed)
            restored.load_state_dict(state)

            remaining = [_sample_index(sample) for sample in restored]

        self.assertEqual(remaining, [2, 3])

    def test_checkpoint_identity_mismatches_fail_during_restore(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, 0, "chunk-a", [0, 1], expected=4)
            state = _dataset(_feed(root, expected=4)).state_dict()

            with self.assertRaisesRegex(ValueError, "another stream"):
                _dataset(_feed(root, expected=4, stream_id="other-stream")).load_state_dict(
                    state
                )
            with self.assertRaisesRegex(ValueError, "expected_samples"):
                _dataset(_feed(root, expected=6)).load_state_dict(state)
            with self.assertRaisesRegex(ValueError, "batch_size"):
                _dataset(_feed(root, expected=4), batch_size=1).load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
