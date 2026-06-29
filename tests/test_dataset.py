from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from anydataset import AnyDataset

from speech_to_speech.config import DatasetFactoryConfig
from speech_to_speech.dataset import dataset_metadata, training_dataset
from helpers import write_toy_longcat_store


class DatasetTest(unittest.TestCase):
    def test_training_dataset_defaults_to_workspace_wmt19_longcat(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = write_toy_longcat_store(root / "store")
            expected = AnyDataset(f"store://{store}:train")

            zhuyin = types.ModuleType("zhuyin")
            datasets = types.ModuleType("zhuyin.datasets")
            wmt19_tts = types.ModuleType("zhuyin.datasets.wmt19_tts")
            wmt19_tts.wmt19_tts_longcat = lambda: expected

            with patch.dict(
                sys.modules,
                {
                    "zhuyin": zhuyin,
                    "zhuyin.datasets": datasets,
                    "zhuyin.datasets.wmt19_tts": wmt19_tts,
                },
            ):
                dataset = training_dataset(DatasetFactoryConfig())
                metadata = dataset_metadata(DatasetFactoryConfig())

        self.assertIs(dataset, expected)
        self.assertEqual(metadata, (expected.spec.to_dict(),))

    def test_training_dataset_rejects_unknown_factory(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported dataset factory"):
            training_dataset(DatasetFactoryConfig(name="unknown"))


if __name__ == "__main__":
    unittest.main()
