from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest
from pathlib import Path
from unittest.mock import patch

from speech_to_speech.config import (
    DataLoaderConfig,
    DataModuleConfig,
    TaskConfig,
    TaskWeightsConfig,
)
from speech_to_speech.datamodule import SpeechToSpeechDataModule
from speech_to_speech.types import CausalLMBatch, Task, TaskFamily
from helpers import (
    MockFrameBPE,
    MockTokenizer,
    isolated_anydataset_home,
    patched_wmt19_longcat,
    toy_embedding,
    write_toy_longcat_store,
)


class SpeechToSpeechDataModuleTest(unittest.TestCase):
    def test_datamodule_config_defaults_to_parallel_workers(self) -> None:
        self.assertEqual(DataModuleConfig().dataloader.num_workers, 8)

    def test_loader_outputs_unified_causal_lm_batch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = write_toy_longcat_store(root / "store")
            module = SpeechToSpeechDataModule(
                DataModuleConfig(
                    dataloader=DataLoaderConfig(batch_size=1, num_workers=0),
                ),
                TaskConfig(
                    enabled=(Task.AUTOREGRESSION.value, Task.TRANSLATION.value)
                ),
                toy_embedding(),
                tokenizer=MockTokenizer(),
                bpe_tokenizer=MockFrameBPE(),
            )

            loader = module.train_dataloader()
            with isolated_anydataset_home(root):
                with patched_wmt19_longcat(store):
                    batches = [batch for _, batch in zip(range(2), loader, strict=False)]

        self.assertNotIsInstance(loader, list)
        self.assertEqual(len(batches), 2)
        for batch in batches:
            self.assertIsInstance(batch, CausalLMBatch)
            self.assertEqual(batch.input_ids.size(0), 1)
            self.assertFalse(hasattr(batch, "task"))
            self.assertFalse(hasattr(batch, "schema"))

    def test_loader_carries_task_family_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = write_toy_longcat_store(root / "store")
            module = SpeechToSpeechDataModule(
                DataModuleConfig(
                    dataloader=DataLoaderConfig(batch_size=4, num_workers=0),
                ),
                TaskConfig(
                    enabled=(Task.AUTOREGRESSION.value, Task.TRANSLATION.value)
                ),
                toy_embedding(),
                tokenizer=MockTokenizer(),
                bpe_tokenizer=MockFrameBPE(),
            )

            with isolated_anydataset_home(root):
                with patched_wmt19_longcat(store):
                    batch = next(iter(module.train_dataloader()))

        self.assertIsNotNone(batch.task_family)
        assert batch.task_family is not None
        self.assertEqual(
            batch.task_family.tolist(),
            [
                TaskFamily.SOURCE_AR.id,
                TaskFamily.TARGET_AR.id,
                TaskFamily.SOURCE_TO_TARGET.id,
                TaskFamily.TARGET_TO_SOURCE.id,
            ],
        )

    def test_loader_uses_task_sampling_weights(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = write_toy_longcat_store(root / "store")
            module = SpeechToSpeechDataModule(
                DataModuleConfig(
                    dataloader=DataLoaderConfig(batch_size=8, num_workers=0),
                ),
                TaskConfig(
                    enabled=(Task.AUTOREGRESSION.value, Task.TRANSLATION.value),
                    weights=TaskWeightsConfig(
                        source_ar=1.0,
                        target_ar=2.0,
                        source_to_target=0.5,
                        target_to_source=0.0,
                    ),
                ),
                toy_embedding(),
                tokenizer=MockTokenizer(),
                bpe_tokenizer=MockFrameBPE(),
            )

            with isolated_anydataset_home(root):
                with patched_wmt19_longcat(store):
                    batch = next(iter(module.train_dataloader()))

        self.assertIsNotNone(batch.task_family)
        assert batch.task_family is not None
        self.assertEqual(
            batch.task_family.tolist(),
            [
                TaskFamily.SOURCE_AR.id,
                TaskFamily.TARGET_AR.id,
                TaskFamily.TARGET_AR.id,
                TaskFamily.SOURCE_AR.id,
                TaskFamily.TARGET_AR.id,
                TaskFamily.TARGET_AR.id,
                TaskFamily.SOURCE_TO_TARGET.id,
            ],
        )

    def test_translation_loader_outputs_unified_causal_lm_batch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = write_toy_longcat_store(root / "store")
            module = SpeechToSpeechDataModule(
                DataModuleConfig(
                    dataloader=DataLoaderConfig(batch_size=2, num_workers=0),
                ),
                TaskConfig(enabled=(Task.TRANSLATION.value,)),
                toy_embedding(),
                tokenizer=MockTokenizer(),
                bpe_tokenizer=MockFrameBPE(),
            )

            loader = module.train_dataloader()

            self.assertNotIsInstance(loader, list)
            with isolated_anydataset_home(root):
                with patched_wmt19_longcat(store):
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

    def test_translation_loader_carries_raw_longcat_audio_sides(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = write_toy_longcat_store(root / "store")
            module = SpeechToSpeechDataModule(
                DataModuleConfig(
                    dataloader=DataLoaderConfig(batch_size=2, num_workers=0),
                ),
                TaskConfig(enabled=(Task.TRANSLATION.value,)),
                toy_embedding(),
                tokenizer=MockTokenizer(),
                bpe_tokenizer=MockFrameBPE(),
            )

            with isolated_anydataset_home(root):
                with patched_wmt19_longcat(store):
                    batch = next(iter(module.train_dataloader()))

        self.assertIsNotNone(batch.source_audio)
        self.assertIsNotNone(batch.target_audio)
        assert batch.source_audio is not None
        assert batch.target_audio is not None
        self.assertEqual(batch.source_audio.semantic_ids.tolist(), [[0, 1, 2], [2, 3, 0]])
        self.assertEqual(
            batch.source_audio.semantic_mask.tolist(),
            [[True, True, True], [True, True, False]],
        )
        self.assertEqual(batch.target_audio.semantic_ids.tolist(), [[2, 3, 0], [0, 1, 2]])
        self.assertEqual(
            batch.target_audio.semantic_mask.tolist(),
            [[True, True, False], [True, True, True]],
        )
        self.assertEqual(tuple(batch.source_audio.acoustic_ids.shape), (2, 4, 3))
        self.assertEqual(
            batch.source_audio.acoustic_mask.tolist(),
            [[True, True, True], [True, True, False]],
        )
        self.assertEqual(tuple(batch.target_audio.acoustic_ids.shape), (2, 4, 3))
        self.assertEqual(
            batch.target_audio.acoustic_mask.tolist(),
            [[True, True, False], [True, True, True]],
        )

    def test_loader_configures_worker_task_stream(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = write_toy_longcat_store(root / "store")
            module = SpeechToSpeechDataModule(
                DataModuleConfig(
                    dataloader=DataLoaderConfig(batch_size=1, num_workers=2),
                ),
                TaskConfig(),
                toy_embedding(),
                tokenizer=MockTokenizer(),
                bpe_tokenizer=MockFrameBPE(),
            )

            with patch(
                "speech_to_speech.datamodule.module.DataLoader",
                return_value=[],
            ) as loader:
                dataloader = module.train_dataloader()

        self.assertFalse(hasattr(dataloader, "__len__"))
        self.assertEqual(loader.call_args.kwargs["num_workers"], 2)
        self.assertEqual(loader.call_args.kwargs["batch_size"], 1)

if __name__ == "__main__":
    unittest.main()
