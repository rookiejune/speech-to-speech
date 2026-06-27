from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest
from pathlib import Path

import torch
from anydataset import AudioItem, AudioView, Modality, Role
from anydataset.store import DatasetWriter
from anytrain.idspace import IdSpace, IdSpaceEmbedding, Modality as IdModality, ModalityBlock

from speech_to_speech.config import DataConfig, TaskConfig
from speech_to_speech.datamodule import SpeechToSpeechDataModule
from speech_to_speech.types import CausalLMBatch, Task
from helpers import MockTokenizer


class MockBPE:
    def encode_units(self, units: list[int]) -> list[int]:
        return [int(unit) for unit in units]


class SpeechToSpeechDataModuleTest(unittest.TestCase):
    def test_data_config_defaults_to_parallel_workers(self) -> None:
        self.assertEqual(DataConfig(datasets=("store://unused",)).num_workers, 8)

    def test_loader_outputs_unified_causal_lm_batch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            store = _write_store(Path(tmpdir) / "store")
            module = SpeechToSpeechDataModule(
                DataConfig(
                    datasets=(f"store://{store}:train",),
                    cache_root=Path(tmpdir) / "cache",
                    batch_size=1,
                    num_workers=0,
                ),
                TaskConfig(
                    enabled=(Task.AUTOREGRESSION.value, Task.TRANSLATION.value)
                ),
                _embedding(),
                tokenizer=MockTokenizer(),
                bpe_tokenizer=MockBPE(),
            )

            loader = module.train_dataloader()
            batches = [batch for _, batch in zip(range(2), loader, strict=False)]

        self.assertNotIsInstance(loader, list)
        self.assertEqual(len(batches), 2)
        for batch in batches:
            self.assertIsInstance(batch, CausalLMBatch)
            self.assertEqual(batch.input_ids.size(0), 1)
            self.assertFalse(hasattr(batch, "task"))
            self.assertFalse(hasattr(batch, "schema"))

    def test_translation_loader_outputs_unified_causal_lm_batch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            store = _write_store(Path(tmpdir) / "store")
            module = SpeechToSpeechDataModule(
                DataConfig(
                    datasets=(f"store://{store}:train",),
                    cache_root=Path(tmpdir) / "cache",
                    batch_size=2,
                    num_workers=0,
                ),
                TaskConfig(enabled=(Task.TRANSLATION.value,)),
                _embedding(),
                tokenizer=MockTokenizer(),
                bpe_tokenizer=MockBPE(),
            )

            loader = module.train_dataloader()

            self.assertNotIsInstance(loader, list)
            batch = next(iter(loader))

        self.assertIsInstance(batch, CausalLMBatch)
        self.assertEqual(tuple(batch.input_ids.shape), (2, 13))
        self.assertEqual(
            batch.input_ids.tolist(),
            [
                [1, 10, 11, 16, 18, 19, 20, 17, 12, 13, 16, 20, 21],
                [1, 10, 11, 16, 20, 21, 17, 12, 13, 16, 18, 19, 20],
            ],
        )
        self.assertFalse(hasattr(batch, "schema"))

    def test_loader_supports_worker_task_stream(self) -> None:
        with TemporaryDirectory() as tmpdir:
            store = _write_store(Path(tmpdir) / "store")
            module = SpeechToSpeechDataModule(
                DataConfig(
                    datasets=(f"store://{store}:train",),
                    cache_root=Path(tmpdir) / "cache",
                    batch_size=1,
                    num_workers=2,
                ),
                TaskConfig(),
                _embedding(),
                tokenizer=MockTokenizer(),
                bpe_tokenizer=MockBPE(),
            )

            batch = next(iter(module.train_dataloader()))

        self.assertIsInstance(batch, CausalLMBatch)
        self.assertEqual(batch.input_ids.size(0), 1)


def _write_store(path: Path) -> Path:
    DatasetWriter(path, dataset_id="toy-s2s", split="train").write(
        [
            _sample(torch.tensor([0, 1, 2]), torch.tensor([2, 3])),
            _sample(torch.tensor([3]), torch.tensor([4, 0, 1])),
        ]
    )
    return path


def _sample(source: torch.Tensor, target: torch.Tensor):
    return {
        (Role.SOURCE, Modality.AUDIO): AudioItem(
            views={AudioView.LONGCAT: _longcat_view(source)}
        ),
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={AudioView.LONGCAT: _longcat_view(target)}
        ),
    }


def _longcat_view(semantic: torch.Tensor) -> dict[str, torch.Tensor]:
    length = int(semantic.numel())
    return {
        "semantic_codes": semantic,
        "acoustic_codes": torch.zeros((4, length), dtype=torch.long),
    }


def _embedding() -> IdSpaceEmbedding:
    space = IdSpace(
        {
            "<|endoftext|>": 0,
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "user": 3,
            "assistant": 4,
            "\n": 5,
            "<think>": 6,
            "</think>": 7,
            "boa": 16,
            "eoa": 17,
        },
        [
            ModalityBlock(IdModality.TEXT, 0, 16),
            ModalityBlock(IdModality.AUDIO, 18, 5),
        ],
    )
    return IdSpaceEmbedding(space, 4)


if __name__ == "__main__":
    unittest.main()
