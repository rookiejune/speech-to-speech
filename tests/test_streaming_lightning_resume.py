from __future__ import annotations

import json
import unittest
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import torch
from anydataset.types import Sample
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor, nn
from torch.optim import SGD
from torch.utils.data import Dataset

from speech_to_speech.callback.streaming import StreamingSynthesis
from speech_to_speech.datamodule.streaming import (
    SnapshotFeed,
    StreamingDataLoader,
    StreamingSnapshotDataset,
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
) -> Trainer:
    return Trainer(
        accelerator="cpu",
        devices=1,
        callbacks=callbacks,
        default_root_dir=None,
        enable_checkpointing=enable_checkpointing,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
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


def _write_snapshot(root: Path) -> None:
    directory = root / "snapshots" / "000000-all"
    directory.mkdir(parents=True)
    payload = {
        "schema": _SNAPSHOT_SCHEMA,
        "stream_id": _STREAM_ID,
        "expected_samples": 4,
        "sequence": 0,
        "snapshot_id": "all",
        "sample_indices": [0, 1, 2, 3],
        "sample_count": 4,
    }
    (directory / "snapshot.json").write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def _write_seal(root: Path) -> None:
    feed = SnapshotFeed(root, stream_id=_STREAM_ID, expected_samples=4, loader=_load_snapshot)
    payload = {
        "schema": _SEAL_SCHEMA,
        "stream_id": _STREAM_ID,
        "expected_samples": 4,
        "snapshot_count": 1,
        "sample_count": 4,
        "catalog_sha256": feed.status().catalog.sha256,
    }
    (root / "sealed.json").write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
