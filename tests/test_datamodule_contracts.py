from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _contracts_helpers import *


class DataModuleContractTest(unittest.TestCase):
    @patch("speech_to_speech.datamodule.module.load_dataset")
    def test_datamodule_setup_loads_dataset_once(self, load_dataset):
        load_dataset.return_value = []
        runtime = _data_runtime()
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(),
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )

        datamodule.setup()
        datamodule.setup()

        load_dataset.assert_called_once_with(config.dataset, runtime)

    @patch("speech_to_speech.datamodule.module.load_dataset")
    def test_multi_loader_shares_speech_dataset_and_worker_budget(
        self,
        load_dataset,
    ):
        class SharedDataset(MapStyleABC):
            def __len__(self):
                return 2

            def __getitem__(self, index):
                if index not in (0, 1):
                    raise IndexError(index)
                return _raw_sample()

        dataset = SharedDataset()
        load_dataset.return_value = dataset
        runtime = _data_runtime()
        config = SpeechConfig(
            codec="longcat",
            dataloader=DataLoaderConfig(batch_size=1, num_workers=4),
        )
        datamodule = DataModule(
            runtime,
            {
                "asr": LoaderSpec.speech(config, {Task.ASR: 1.0}),
                "tts": LoaderSpec.speech(config, {Task.TTS: 1.0}),
            },
            LoaderSchedule({"asr": 0.25, "tts": 0.75}),
        )

        datamodule.setup()

        loaders = cast(Any, datamodule)._loaders
        load_dataset.assert_called_once_with(config.dataset, runtime)
        self.assertIs(loaders["asr"]._dataset, dataset)
        self.assertIs(loaders["tts"]._dataset, dataset)
        self.assertEqual(loaders["asr"].num_workers, 1)
        self.assertEqual(loaders["tts"].num_workers, 3)
        self.assertEqual(
            sum(loader.num_workers for loader in loaders.values()),
            config.dataloader.num_workers,
        )
        scheduled = cast(Any, datamodule.train_dataloader())
        self.assertEqual(scheduled.loaders["asr"].num_workers, 1)
        self.assertEqual(scheduled.loaders["tts"].num_workers, 3)

    @patch("speech_to_speech.datamodule.module.load_dataset")
    def test_datamodule_keeps_standard_loader_for_non_store_dataset(
        self,
        load_dataset,
    ):
        load_dataset.return_value = [_raw_sample(), _raw_sample()]
        runtime = _data_runtime()
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(2),
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )

        datamodule.setup()
        loader = cast(Any, datamodule.train_dataloader())

        self.assertIs(loader.dataset, load_dataset.return_value)
        self.assertEqual(loader.batch_size, 2)

    @patch("speech_to_speech.datamodule.module.load_dataset")
    def test_datamodule_rejects_enabled_costs_for_non_mapstyle_dataset(
        self,
        load_dataset,
    ):
        load_dataset.return_value = [_raw_sample(), _raw_sample()]
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
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )
        datamodule.setup()
        with self.assertRaisesRegex(ValueError, "non-MapStyle"):
            datamodule.train_dataloader()

    def test_toy_dataset_uses_codec_shapes_and_value_ranges(self):
        cases = (
            (
                "longcat",
                SimpleNamespace(
                    sample_rate=16_000,
                    semantic_feature_dim=4,
                    semantic_codebook=torch.zeros(5, 4),
                    codebook_sizes=(5, 3, 7),
                    acoustic_feature_dim=4,
                    acoustic_codebook_sizes=(3, 7),
                    acoustic_codes_to_features=Mock(),
                    decode_features=Mock(),
                    frame_rate=50.0,
                ),
                AudioView.LONGCAT,
                (5, 3, 7),
            ),
            (
                "unicodec",
                SimpleNamespace(
                    semantic_feature_dim=4,
                    semantic_codebook=torch.zeros(11, 4),
                    codebook_sizes=(11,),
                    frame_rate=50.0,
                ),
                AudioView.UNICODEC,
                (11,),
            ),
        )

        for codec_name, codec, view, sizes in cases:
            with self.subTest(codec=codec_name):
                dataset = ToyDataset(codec_name, codec, samples=2, frames=3)
                first = dataset[0]
                again = dataset[0]
                self.assertEqual(len(dataset), 2)
                for role in (Role.SOURCE, Role.TARGET):
                    item = first[(role, Modality.AUDIO)]
                    codes = item.views[view]
                    self.assertEqual(tuple(codes.shape), (3, len(sizes)))
                    for codebook, size in enumerate(sizes):
                        self.assertTrue((codes[:, codebook] >= 0).all())
                        self.assertTrue((codes[:, codebook] < size).all())
                    self.assertTrue(
                        torch.equal(codes, again[(role, Modality.AUDIO)].views[view])
                    )
    def test_datamodule_loads_toy_data_without_prepared_dataset(self):
        runtime = _data_runtime()
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(),
            dataset=DatasetConfig(
                name=DatasetName.TOY,
                toy_samples=2,
                toy_frames=3,
            ),
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )

        datamodule.setup()

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
        loader = cast(Any, datamodule.train_dataloader())
        self.assertEqual(loader.batch_sampler.max_batch_samples, 1)
    def test_wmt19_loader_uses_default_filter(self):
        loaded, dataset, moss_tts, filtered, view = _load_wmt19_dataset(
            DatasetConfig()
        )

        self.assertIs(loaded, dataset)
        moss_tts.codec.assert_called_once_with(
            "longcat",
            root=None,
            split="train",
        )
        view.filter.assert_called_once_with("speech_translation_v1")
        filtered.load.assert_called_once_with()
    def test_wmt19_loader_can_disable_filter(self):
        loaded, dataset, moss_tts, filtered, view = _load_wmt19_dataset(
            DatasetConfig(filter=None)
        )

        self.assertIs(loaded, dataset)
        moss_tts.codec.assert_called_once_with(
            "longcat",
            root=None,
            split="train",
        )
        view.filter.assert_called_once_with(None)
        filtered.load.assert_called_once_with()
    def test_toy_settings_reject_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ToyConfig(hidden_size=7, heads=2)
        with self.assertRaisesRegex(ValueError, "toy_samples"):
            DatasetConfig(name=DatasetName.TOY, toy_samples=0)
        codec = SimpleNamespace(
            sample_rate=16_000,
            semantic_feature_dim=4,
            semantic_codebook=torch.zeros(5, 4),
            codebook_sizes=(5, 4),
            acoustic_feature_dim=4,
            acoustic_codebook_sizes=(3,),
            acoustic_codes_to_features=Mock(),
            decode_features=Mock(),
            frame_rate=50.0,
        )
        with self.assertRaisesRegex(ValueError, "LongCat"):
            ToyDataset("longcat", codec)
    def test_datamodule_rejects_runtime_codec_mismatch(self):
        config = SpeechConfig(
            codec="unicodec",
            dataloader=_loader(),
        )
        datamodule = DataModule(
            _data_runtime(),
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )

        with self.assertRaisesRegex(ValueError, "same codec"):
            datamodule.setup()


if __name__ == "__main__":
    unittest.main()
