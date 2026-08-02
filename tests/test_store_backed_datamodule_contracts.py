from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _contracts_helpers import *


class StoreBackedDataModuleContractTest(unittest.TestCase):
    def test_datamodule_shards_child_loader_indices_across_ranks(self):
        runtime = _data_runtime()
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(2),
            dataset=DatasetConfig(
                name=DatasetName.TOY,
                toy_samples=6,
                toy_frames=3,
            ),
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )
        datamodule.setup()
        sampler = cast(Any, datamodule.train_dataloader()).batch_sampler

        rank_batches = []
        for rank in range(2):
            with patch(
                "anydataset.dataset.batching.rank",
                return_value=(2, rank),
            ):
                rank_batches.append(list(sampler))

        rank_indices = [
            {index for batch in batches for index in batch}
            for batches in rank_batches
        ]
        self.assertTrue(rank_indices[0].isdisjoint(rank_indices[1]))
        self.assertEqual(rank_indices[0] | rank_indices[1], set(range(6)))
        self.assertEqual(len(rank_batches[0]), len(rank_batches[1]))
    def test_datamodule_uses_anydataset_batches_for_store_backed_data(self):
        with _store_dataset([_raw_sample(index) for index in range(4)]) as dataset:
            runtime = _data_runtime()
            config = SpeechConfig(codec="longcat", dataloader=_loader(2))
            datamodule = _speech_datamodule(runtime, config, {Task.TTS: 1.0})

            loader = _store_train_loader(self, datamodule, dataset)

            _assert_store_sampler(self, loader, dataset)
            self.assertEqual(
                len(
                    datamodule.diagnostic_samples(
                        [0, 1],
                        split=SampleSplit.TRAIN,
                        loader_name="train",
                    )
                ),
                2,
            )
    def test_datamodule_smoke_uses_audio_frame_costs_for_store_backed_data(self):
        with _store_dataset([_raw_sample(index) for index in range(4)]) as dataset:
            runtime = _data_runtime()
            config = SpeechConfig(
                codec="longcat",
                dataloader=DataLoaderConfig(
                    batch_size=2,
                    num_workers=0,
                    costs=DataLoaderCostsConfig(
                        enabled=True,
                        max_batch_frames=8,
                        planning_window=4,
                    ),
                ),
            )
            datamodule = _speech_datamodule(runtime, config, {Task.TTS: 1.0})

            loader = _store_train_loader(self, datamodule, dataset)

            sampler = _assert_store_sampler(
                self,
                loader,
                dataset,
                max_batch_memory=8,
            )
            self.assertIsNotNone(sampler.costs)
            self.assertEqual(sampler.costs[0], 4)
            self.assertEqual(sampler.planning_window, 4)
    def test_datamodule_uses_store_backed_data_without_duration(self):
        samples = [_raw_sample_without_duration(index) for index in range(4)]
        with _store_dataset(samples) as dataset:
            runtime = _data_runtime()
            runtime.text_tokenizer = _ChatTokenizer(10)
            config = SpeechConfig(codec="longcat", dataloader=_loader(2))
            datamodule = _speech_datamodule(runtime, config, {Task.S2ST: 1.0})

            loader = _store_train_loader(self, datamodule, dataset)

            _assert_store_sampler(self, loader, dataset)
            batch = next(iter(loader))
            self.assertEqual(batch.tasks, [Task.S2ST, Task.S2ST])
            torch.testing.assert_close(
                batch.audio_seconds,
                torch.tensor([0.08, 0.08]),
            )
            self.assertEqual(
                len(
                    datamodule.diagnostic_samples(
                        [0, 1],
                        split=SampleSplit.TRAIN,
                        loader_name="train",
                    )
                ),
                2,
            )
    def test_datamodule_enabled_costs_require_audio_duration_metadata(self):
        samples = [_raw_sample_without_duration(index) for index in range(2)]
        with _store_dataset(samples) as dataset:
            runtime = _data_runtime()
            config = SpeechConfig(
                codec="longcat",
                dataloader=DataLoaderConfig(
                    batch_size=2,
                    num_workers=0,
                    costs=DataLoaderCostsConfig(
                        enabled=True,
                        max_batch_frames=8,
                    ),
                ),
            )
            datamodule = _speech_datamodule(runtime, config, {Task.TTS: 1.0})

            loader = _store_train_loader(self, datamodule, dataset)

            with self.assertRaisesRegex(ValueError, "duration metadata"):
                _ = loader.batch_sampler.costs[0]


if __name__ == "__main__":
    unittest.main()
