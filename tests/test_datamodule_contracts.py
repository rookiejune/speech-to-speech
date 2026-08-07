from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _contracts_helpers import *
from torch.utils.data import IterableDataset
from speech_to_speech.datamodule.asset import AssetPhase, AssetResolution
from speech_to_speech.datamodule.config import AssetMaterializationConfig
from speech_to_speech.datamodule.dataset.speech import uses_distinct_audio_assets


class DataModuleContractTest(unittest.TestCase):
    def test_canonical_covost2_validation_uses_workspace_loader(self):
        dataset = [Mock()]
        covost2 = SimpleNamespace(load=Mock(return_value=dataset))
        datasets = ModuleType("zhuyin.datasets")
        datasets.covost2 = covost2
        config = DatasetConfig(
            name=DatasetName.COVOST2,
            split="validation",
            filter=None,
            source_lang="zh",
            target_lang="en",
        )

        with patch.dict(
            sys.modules,
            {
                "zhuyin": ModuleType("zhuyin"),
                "zhuyin.datasets": datasets,
            },
        ):
            loaded = load_dataset(config, _data_runtime())

        self.assertIs(loaded, dataset)
        covost2.load.assert_called_once_with(
            split="validation",
            source_lang="zh",
            target_lang="en",
        )

    def test_canonical_libritts_validation_uses_workspace_loader(self):
        dataset = [Mock()]
        libritts = SimpleNamespace(load=Mock(return_value=dataset))
        datasets = ModuleType("zhuyin.datasets")
        datasets.libritts = libritts
        config = DatasetConfig(
            name=DatasetName.LIBRITTS,
            split="dev-clean",
            filter=None,
        )

        with patch.dict(
            sys.modules,
            {
                "zhuyin": ModuleType("zhuyin"),
                "zhuyin.datasets": datasets,
            },
        ):
            loaded = load_dataset(config, _data_runtime())

        self.assertIs(loaded, dataset)
        libritts.load.assert_called_once_with(split="dev-clean")

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
    def test_datamodule_uses_injected_dataset_for_the_named_speech_loader(
        self,
        load_dataset,
    ):
        injected = [_raw_sample(), _raw_sample()]
        runtime = _data_runtime()
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(),
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
            training_datasets={"train": injected},
        )

        datamodule.setup()

        load_dataset.assert_not_called()
        loader = cast(Any, datamodule.train_dataloader())
        self.assertIs(loader.dataset, injected)

    def test_datamodule_rejects_unknown_or_text_training_dataset_injection(self):
        runtime = _data_runtime()
        speech = SpeechConfig(codec="longcat", dataloader=_loader())
        with self.assertRaisesRegex(ValueError, "unknown loaders: missing"):
            DataModule(
                runtime,
                {"train": LoaderSpec.speech(speech, {Task.TTS: 1.0})},
                training_datasets={"missing": []},
            )

        text = TextConfig(dataloader=_loader())
        with self.assertRaisesRegex(ValueError, "only be injected into speech"):
            DataModule(
                runtime,
                {"mt": LoaderSpec.text(text, {Task.MT: 1.0})},
                training_datasets={"mt": []},
            )

    def test_datamodule_keeps_and_closes_one_live_s2st_dataset(self):
        class LiveDataset(IterableDataset):
            lineage_id = "lineage"

            def __init__(self):
                self.closed = 0

            def __iter__(self):
                yield _raw_sample()

            def acknowledge(self):
                pass

            def state_dict(self):
                return {}

            def load_state_dict(self, _value):
                pass

            def set_stop_requested(self, _predicate):
                pass

            def close(self):
                self.closed += 1

        dataset = LiveDataset()
        runtime = _data_runtime()
        config = SpeechConfig(
            codec="longcat",
            dataloader=DataLoaderConfig(batch_size=2, num_workers=0),
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
            training_datasets={"train": dataset},
        )

        datamodule.setup()
        loader = cast(Any, datamodule.train_dataloader())

        self.assertIs(loader.dataset, dataset)
        self.assertEqual(loader.num_workers, 0)
        self.assertIs(cast(Any, datamodule)._loaders["train"]._dataset, dataset)
        datamodule.teardown()
        datamodule.teardown()
        self.assertEqual(dataset.closed, 1)

    def test_datamodule_rejects_workers_and_costs_for_live_s2st(self):
        class LiveDataset(IterableDataset):
            lineage_id = "lineage"

            def __iter__(self):
                return iter(())

            def acknowledge(self):
                pass

            def state_dict(self):
                return {}

            def load_state_dict(self, _value):
                pass

            def set_stop_requested(self, _predicate):
                pass

            def close(self):
                pass

        cases = (
            (
                DataLoaderConfig(batch_size=1, num_workers=1),
                "num_workers=0",
            ),
            (
                DataLoaderConfig(
                    batch_size=1,
                    num_workers=0,
                    persistent_workers=True,
                ),
                "persistent_workers=false",
            ),
            (
                DataLoaderConfig(
                    batch_size=1,
                    num_workers=0,
                    costs=DataLoaderCostsConfig(
                        enabled=True,
                        max_batch_frames=8,
                    ),
                ),
                "costs are unsupported",
            ),
        )
        for dataloader, message in cases:
            with self.subTest(message=message):
                config = SpeechConfig(codec="longcat", dataloader=dataloader)
                datamodule = DataModule(
                    _data_runtime(),
                    {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
                    training_datasets={"train": LiveDataset()},
                )
                with self.assertRaisesRegex(ValueError, message):
                    datamodule.setup()

    @patch("speech_to_speech.datamodule.module.load_dataset")
    def test_speech_validation_max_samples_builds_a_deterministic_prefix(self, load_dataset):
        samples = [_raw_sample() for _ in range(5)]
        load_dataset.return_value = samples
        runtime = _data_runtime()
        config = SpeechConfig(
            codec="longcat",
            dataloader=DataLoaderConfig(batch_size=2, num_workers=0),
        )
        spec = LoaderSpec.speech(
            config,
            {Task.TTS: 1.0},
            max_samples=3,
        )
        datamodule = DataModule(runtime, {"train": spec}, validation=spec)

        datamodule.setup()
        validation = cast(Any, datamodule.val_dataloader())

        self.assertEqual(len(validation.dataset), 3)
        self.assertEqual(list(validation.dataset.indices), [0, 1, 2])
        self.assertEqual(validation.batch_size, 2)

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
                    semantic_codebook_sizes=(5,),
                    codebook_sizes=(5, 3, 7),
                    acoustic_feature_dim=4,
                    acoustic_codebook_sizes=(3, 7),
                    acoustic_layout=AcousticLayout.FRAME_ALIGNED,
                    acoustic_unit_length=None,
                    acoustic_codes_to_features=Mock(),
                    tokenize=Mock(),
                    detokenize=Mock(),
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

    def test_toy_dataset_reuses_prepared_codes_for_same_codec_view_bpe_split(self):
        runtime = _data_runtime()
        runtime.input_audio_decoupled = True
        runtime.input_audio_tokenizer = NativeAudioTokenizer(vocab_size=4)

        dataset = cast(
            ToyDataset,
            load_dataset(
                DatasetConfig(
                    name=DatasetName.TOY,
                    toy_samples=2,
                    toy_frames=3,
                ),
                runtime,
            ),
        )

        self.assertFalse(uses_distinct_audio_assets(runtime))
        self.assertIsNone(dataset.input_view)
        source = cast(AudioItem, dataset[0][Role.SOURCE, Modality.AUDIO])
        self.assertEqual(set(source.views), {AudioView.LONGCAT})

    def test_dataset_rejects_distinct_codecs_with_the_same_audio_view(self):
        runtime = _data_runtime()
        runtime.input_audio_decoupled = True
        runtime.input_codec_name = "other-longcat"

        with self.assertRaisesRegex(
            ValueError,
            "distinct input/output codecs must use distinct audio views",
        ):
            load_dataset(DatasetConfig(name=DatasetName.TOY), runtime)

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
    def test_streaming_s2st_loader_uses_unfiltered_codec_resource(self):
        loaded, dataset, streaming_s2st, view = _load_streaming_s2st_dataset(
            DatasetConfig(name=DatasetName.STREAMING_S2ST, filter=None)
        )

        self.assertIs(loaded, dataset)
        self.assertEqual(len(loaded), 4)
        streaming_s2st.codec.assert_called_once_with(
            "longcat",
            root=None,
            split="train",
        )
        view.load.assert_called_once_with()
    def test_streaming_s2st_rejects_wmt19_filter(self):
        with self.assertRaisesRegex(ValueError, "does not accept a filter"):
            DatasetConfig(name=DatasetName.STREAMING_S2ST)
    def test_toy_settings_reject_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ToyConfig(hidden_size=7, heads=2)
        with self.assertRaisesRegex(ValueError, "toy_samples"):
            DatasetConfig(name=DatasetName.TOY, toy_samples=0)
        codec = SimpleNamespace(
            sample_rate=16_000,
            semantic_feature_dim=4,
            semantic_codebook=torch.zeros(5, 4),
            semantic_codebook_sizes=(5,),
            codebook_sizes=(5, 4),
            acoustic_feature_dim=4,
            acoustic_codebook_sizes=(3,),
            acoustic_layout=AcousticLayout.FRAME_ALIGNED,
            acoustic_unit_length=None,
            acoustic_codes_to_features=Mock(),
            tokenize=Mock(),
            detokenize=Mock(),
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

    @patch("speech_to_speech.datamodule.module.resolve_workspace_asset")
    def test_datamodule_refreshes_materialized_asset_for_next_epoch(
        self,
        resolve_workspace_asset,
    ):
        fallback = [_raw_sample()]
        ready = [_raw_sample(), _raw_sample()]

        class FakeJob:
            request_id = "asset-request-v1"

            def __init__(self):
                self.phase = AssetPhase.FALLBACK
                self.starts = []
                self.finishes = []
                self.loads = 0

            def start(self, *, owner):
                self.starts.append(owner)
                self.phase = AssetPhase.MATERIALIZING

            def finish(self, *, owner):
                self.finishes.append(owner)
                self.phase = AssetPhase.READY

            def load_ready(self):
                self.loads += 1
                return ready

            def close(self):
                pass

        job = FakeJob()
        resolve_workspace_asset.return_value = AssetResolution(
            fallback,
            request_id=job.request_id,
            job=cast(Any, job),
        )
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(),
            encode_missing_codes=True,
            materialization=AssetMaterializationConfig(
                enabled=True,
                output_root="/tmp/s2s-assets",
                device="cpu",
                provider_id="longcat-provider-v1",
            ),
        )
        datamodule = DataModule(
            _data_runtime(),
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )

        datamodule.setup()
        epoch_zero = cast(Any, datamodule.train_dataloader())
        self.assertIs(epoch_zero.dataset, fallback)
        self.assertTrue(datamodule.materialization_enabled)
        self.assertTrue(datamodule.has_pending_assets)

        datamodule.start_asset_materialization(owner=True)
        datamodule.finish_asset_materialization(owner=True)
        datamodule.refresh_materialized_assets()

        epoch_one = cast(Any, datamodule.train_dataloader())
        self.assertIs(epoch_one.dataset, ready)
        self.assertFalse(datamodule.has_pending_assets)
        self.assertEqual(job.starts, [True])
        self.assertEqual(job.finishes, [True])
        self.assertEqual(job.loads, 1)
        resolve_workspace_asset.assert_called_once_with(
            config.dataset,
            datamodule.runtime,
            config.materialization,
        )

    def test_datamodule_rejects_multiple_training_loaders_during_materialization(
        self,
    ):
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(),
            encode_missing_codes=True,
            materialization=AssetMaterializationConfig(
                enabled=True,
                output_root="/tmp/s2s-assets",
                device="cpu",
                provider_id="longcat-provider-v1",
            ),
        )

        with self.assertRaisesRegex(
            ValueError,
            "exactly one training speech loader",
        ):
            DataModule(
                _data_runtime(),
                {
                    "asr": LoaderSpec.speech(config, {Task.ASR: 1.0}),
                    "tts": LoaderSpec.speech(config, {Task.TTS: 1.0}),
                },
            )


if __name__ == "__main__":
    unittest.main()
