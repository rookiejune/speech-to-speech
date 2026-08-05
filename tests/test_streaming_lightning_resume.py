from __future__ import annotations

import json
import unittest
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest.mock import patch

import torch
from anydataset.types import Sample
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import Tensor, nn
from torch.optim import SGD
from torch.utils.data import Dataset

from speech_to_speech.callback.streaming import (
    StreamingSynthesis,
    StreamingTelemetryCallback,
)
from speech_to_speech.datamodule.streaming import (
    SnapshotFeed,
    StreamingDataLoader,
    StreamingSnapshotDataset,
    StreamingTelemetry,
)


_SNAPSHOT_SCHEMA = "speech-to-speech-stream-snapshot-v1"
_SEAL_SCHEMA = "speech-to-speech-stream-seal-v1"
_STREAM_ID = "lightning-resume-test"


class _Samples(Dataset[Sample]):
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Sample:
        return cast(Sample, cast(object, {"index": self.indices[index]}))


class _CountingStreamingDataLoader(StreamingDataLoader):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.restore_count = 0

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.restore_count += 1
        super().load_state_dict(state)


class _StreamingDataModule(LightningDataModule):
    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = root
        self.loader: _CountingStreamingDataLoader | None = None
        self._pending_state: Mapping[str, object] | None = None
        self.loaded_states = 0

    @property
    def streaming_enabled(self) -> bool:
        return True

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.loader is not None:
            return
        dataset = StreamingSnapshotDataset(
            SnapshotFeed(
                self.root,
                stream_id=_STREAM_ID,
                expected_samples=4,
                loader=_load_snapshot,
            ),
            batch_size=2,
            poll_seconds=0.001,
            status_seconds=60.0,
        )
        self.loader = _CountingStreamingDataLoader(
            dataset,
            collate_fn=_identity,
            pin_memory=False,
        )
        if self._pending_state is not None:
            self.loader.load_state_dict(self._pending_state)
            self._pending_state = None

    def train_dataloader(self) -> _CountingStreamingDataLoader:
        if self.loader is None:
            raise RuntimeError("setup() must run before train_dataloader().")
        return self.loader

    def state_dict(self) -> dict[str, object]:
        if self.loader is None:
            raise RuntimeError("streaming loader was not created before checkpointing.")
        return {"loader": self.loader.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state = state_dict.get("loader")
        if not isinstance(state, Mapping):
            raise TypeError("test checkpoint has no streaming loader state.")
        self.loaded_states += 1
        if self.loader is None:
            self._pending_state = dict(state)
            return
        self.loader.load_state_dict(cast(Mapping[str, object], state))

    def start_streaming_synthesis(self, *, owner: bool) -> None:
        del owner

    def check_streaming_synthesis(self, *, owner: bool) -> None:
        del owner

    def close_streaming_synthesis(self, *, owner: bool) -> None:
        del owner

    def set_streaming_global_step(self, step: int) -> None:
        self._loader().set_global_step(step)

    def acknowledge_streaming_batch(self, global_step: int) -> None:
        self._loader().acknowledge(global_step)

    def streaming_telemetry(
        self,
        *,
        loader_name: str | None = None,
    ) -> StreamingTelemetry:
        del loader_name
        return self._loader().telemetry()

    def _loader(self) -> _CountingStreamingDataLoader:
        if self.loader is None:
            raise RuntimeError("streaming loader was not created.")
        return self.loader


class _Module(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.seen: list[int] = []

    def training_step(self, batch: list[Sample], batch_idx: int) -> Tensor:
        del batch_idx
        self.seen.extend(_index(sample) for sample in batch)
        return self.weight * 0

    def configure_optimizers(self) -> SGD:
        return SGD(self.parameters(), lr=0.1)


class _AcknowledgedCheckpoint(ModelCheckpoint):
    def __init__(self, datamodule: _StreamingDataModule, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.datamodule = datamodule
        self.committed_positions: list[int] = []

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        self.committed_positions.append(self.datamodule._loader().dataset.committed_position)
        super()._save_checkpoint(trainer, filepath)


class StreamingLightningResumeTest(unittest.TestCase):
    def test_telemetry_writes_real_tensorboard_scalars_and_gpu_summary(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root)
            _write_seal(root)
            logger = TensorBoardLogger(save_dir=root / "logs", name="stream")
            data = _StreamingDataModule(root)
            callback = StreamingTelemetryCallback(
                gpu_sample_interval_seconds=0,
                log_every_n_steps=1,
            )

            _trainer(
                max_steps=2,
                callbacks=[StreamingSynthesis(), callback],
                logger=logger,
                default_root_dir=root,
            ).fit(_Module(), datamodule=data)

            accumulator = EventAccumulator(logger.log_dir)
            accumulator.Reload()
            tags = set(accumulator.Tags()["scalars"])
            summary = json.loads(
                (Path(logger.log_dir) / "streaming_gpu_summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertIn("streaming/batch_wait_seconds", tags)
        self.assertIn("streaming/step_seconds", tags)
        self.assertIn("streaming/committed_position", tags)
        self.assertFalse(summary["available"])
        self.assertEqual(summary["reason"], "disabled")

    def test_trains_published_batch_before_waiting_for_the_next_snapshot(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root, snapshot_id="first", indices=[0, 1])
            data = _StreamingDataModule(root)
            model = _Module()
            published = False

            def publish(_seconds: float) -> None:
                nonlocal published
                if published:
                    raise AssertionError("streaming loader waited after the seal.")
                self.assertEqual(model.seen, [0, 1])
                _write_snapshot(
                    root,
                    sequence=1,
                    snapshot_id="second",
                    indices=[2, 3],
                )
                _write_seal(root)
                published = True

            with patch(
                "speech_to_speech.datamodule.streaming.time.sleep",
                side_effect=publish,
            ):
                _trainer(
                    max_steps=2,
                    callbacks=[StreamingSynthesis()],
                    enable_checkpointing=False,
                ).fit(model, datamodule=data)

        self.assertTrue(published)
        self.assertEqual(model.seen, [0, 1, 2, 3])

    def test_checkpoint_resume_restores_only_unconsumed_sealed_samples(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _write_snapshot(root)
            _write_seal(root)

            first_data = _StreamingDataModule(root)
            checkpoint = _AcknowledgedCheckpoint(
                first_data,
                dirpath=root / "checkpoints",
                save_last=True,
                every_n_train_steps=1,
            )
            first_model = _Module()
            _trainer(max_steps=1, callbacks=[StreamingSynthesis(), checkpoint]).fit(
                first_model,
                datamodule=first_data,
            )

            path = Path(checkpoint.last_model_path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            datamodule_state = saved.get("_StreamingDataModule")
            self.assertIsInstance(datamodule_state, Mapping)
            loader_state = cast(Mapping[str, object], datamodule_state).get("loader")
            self.assertIsInstance(loader_state, Mapping)
            dataset_state = cast(Mapping[str, object], loader_state).get("dataset")
            self.assertIsInstance(dataset_state, Mapping)
            self.assertEqual(cast(Mapping[str, object], dataset_state)["committed_position"], 2)
            self.assertEqual(checkpoint.committed_positions, [2, 2])

            resumed_data = _StreamingDataModule(root)
            resumed_model = _Module()
            _trainer(
                max_steps=2,
                callbacks=[StreamingSynthesis()],
                enable_checkpointing=False,
            ).fit(
                resumed_model,
                datamodule=resumed_data,
                ckpt_path=path,
            )

        self.assertEqual(first_model.seen, [0, 1])
        self.assertEqual(resumed_model.seen, [2, 3])
        self.assertEqual(resumed_data.loaded_states, 1)
        self.assertIsNotNone(resumed_data.loader)
        self.assertGreaterEqual(cast(_CountingStreamingDataLoader, resumed_data.loader).restore_count, 2)


def _trainer(
    *,
    max_steps: int,
    callbacks: list[Any],
    enable_checkpointing: bool = True,
    logger: Any = False,
    default_root_dir: str | Path | None = None,
) -> Trainer:
    return Trainer(
        accelerator="cpu",
        devices=1,
        callbacks=callbacks,
        default_root_dir=default_root_dir,
        enable_checkpointing=enable_checkpointing,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=logger,
        max_steps=max_steps,
        num_sanity_val_steps=0,
    )


def _identity(samples: list[Sample]) -> list[Sample]:
    return samples


def _index(sample: Sample) -> int:
    value = cast(Mapping[str, object], cast(object, sample)).get("index")
    if type(value) is not int:
        raise TypeError("test sample has no integer index.")
    return value


def _load_snapshot(root: Path) -> Dataset[Sample]:
    payload = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    indices = payload.get("sample_indices")
    if not isinstance(indices, list) or any(type(index) is not int for index in indices):
        raise TypeError("test snapshot indices must be integers.")
    return _Samples(cast(list[int], indices))


def _write_snapshot(
    root: Path,
    *,
    sequence: int = 0,
    snapshot_id: str = "all",
    indices: list[int] | None = None,
) -> None:
    sample_indices = [0, 1, 2, 3] if indices is None else indices
    directory = root / "snapshots" / f"{sequence:06d}-{snapshot_id}"
    directory.mkdir(parents=True)
    payload = {
        "schema": _SNAPSHOT_SCHEMA,
        "stream_id": _STREAM_ID,
        "expected_samples": 4,
        "sequence": sequence,
        "snapshot_id": snapshot_id,
        "sample_indices": sample_indices,
        "sample_count": len(sample_indices),
    }
    (directory / "snapshot.json").write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def _write_seal(root: Path) -> None:
    feed = SnapshotFeed(root, stream_id=_STREAM_ID, expected_samples=4, loader=_load_snapshot)
    catalog = feed.status().catalog
    payload = {
        "schema": _SEAL_SCHEMA,
        "stream_id": _STREAM_ID,
        "expected_samples": 4,
        "snapshot_count": len(catalog.snapshots),
        "sample_count": catalog.sample_count,
        "catalog_sha256": catalog.sha256,
    }
    (root / "sealed.json").write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
