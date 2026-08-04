from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from speech_to_speech.datamodule.config import DataLoaderConfig
from speech_to_speech.datamodule.mimo import (
    JsonlMimoSegmentDataset,
    MimoDataModule,
    MimoDatasetConfig,
    MimoTaskDataset,
)
from speech_to_speech.mimo import MimoSegment, MimoSpecialTokens, MimoTask


class JsonlMimoSegmentDatasetTest(unittest.TestCase):
    def test_reads_prepared_segments_and_optional_features(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "segments.jsonl"
            records = [
                {
                    "text_input_ids": [1, 2],
                    "audio_input_ids": [3, 4, 5],
                    "audio_features": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    "recording_id": "r0",
                    "segment_index": 0,
                },
                {
                    "text_input_ids": [6],
                    "audio_input_ids": [7, 8],
                    "recording_id": "r0",
                    "segment_index": 1,
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            dataset = JsonlMimoSegmentDataset(path)
            first = dataset[0]

            self.assertEqual(len(dataset), 2)
            self.assertTrue(torch.equal(first.text_input_ids, torch.tensor([1, 2])))
            self.assertEqual(tuple(first.audio_features.shape), (3, 2))
            self.assertEqual(first.recording_id, "r0")
            self.assertEqual(dataset[-1].segment_index, 1)

    def test_rejects_invalid_rows_when_materialized(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "segments.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "text_input_ids": [1],
                        "audio_input_ids": [2, 3],
                        "audio_features": [[0.1]],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = JsonlMimoSegmentDataset(path)
            with self.assertRaisesRegex(ValueError, "audio_features must have shape"):
                dataset[0]


class MimoTaskDatasetContractTest(unittest.TestCase):
    def test_context_tasks_require_explicit_recording_order(self) -> None:
        segments = [
            MimoSegment(torch.tensor([1]), torch.tensor([2])),
            MimoSegment(torch.tensor([3]), torch.tensor([4])),
        ]
        special = MimoSpecialTokens(1, 2, 0, 5, 6, 7)
        with self.assertRaisesRegex(ValueError, "recording_id and consecutive segment_index"):
            MimoTaskDataset(segments, special)

    def test_epoch_changes_deterministic_task_mapping(self) -> None:
        segments = [
            MimoSegment(
                torch.tensor([1]),
                torch.tensor([2]),
                recording_id="r",
                segment_index=index,
            )
            for index in range(3)
        ]
        special = MimoSpecialTokens(1, 2, 0, 5, 6, 7)
        dataset = MimoTaskDataset(
            segments,
            special,
            config=MimoDatasetConfig(seed=4, samples_per_epoch=32),
        )
        before = tuple(dataset.task_for_index(index) for index in range(32))
        dataset.set_epoch(1)
        after = tuple(dataset.task_for_index(index) for index in range(32))
        self.assertTrue(any(left != right for left, right in zip(before, after)))

    def test_context_windows_follow_metadata_not_manifest_order(self) -> None:
        segments = [
            MimoSegment(
                torch.tensor([index + 1]),
                torch.tensor([index + 2]),
                recording_id="r",
                segment_index=index,
            )
            for index in (1, 0)
        ]
        special = MimoSpecialTokens(1, 2, 0, 5, 6, 7)
        dataset = MimoTaskDataset(
            segments,
            special,
            config=MimoDatasetConfig(
                samples_per_epoch=1,
                task_weights={MimoTask.AUDIO_TO_NEXT_TEXT: 1.0},
            ),
        )
        sample = dataset[0]
        self.assertEqual(sample.recording_id, "r")

    def test_epoch_aware_dataset_rejects_persistent_workers(self) -> None:
        segments = [
            MimoSegment(
                torch.tensor([index + 1]),
                torch.tensor([index + 2]),
                recording_id="r",
                segment_index=index,
            )
            for index in range(2)
        ]
        dataset = MimoTaskDataset(
            segments,
            MimoSpecialTokens(1, 2, 0, 5, 6, 7),
        )

        with self.assertRaisesRegex(ValueError, "persistent_workers is incompatible"):
            MimoDataModule(
                dataset,
                dataloader=DataLoaderConfig(
                    batch_size=1,
                    num_workers=1,
                    persistent_workers=True,
                ),
                text_pad_token_id=0,
                audio_pad_token_id=7,
            )


if __name__ == "__main__":
    unittest.main()
