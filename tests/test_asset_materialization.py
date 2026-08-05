from __future__ import annotations

import sys
import time
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Iterator, cast
from unittest.mock import Mock, call, patch

from anydataset.dataset import IndexSelection, MapStyleABC
from anydataset.types import AudioView, Sample

from speech_to_speech.datamodule import asset
from speech_to_speech.datamodule.asset import (
    AssetPhase,
    AssetRequest,
    BackgroundAssetJob,
    BackgroundDualAssetJob,
    DualAssetRequest,
    DualWorkspaceCodecProducer,
    WorkspaceCodecProducer,
    resolve_workspace_asset,
)
from speech_to_speech.datamodule.config import (
    AssetMaterializationConfig,
    DataLoaderConfig,
    SpeechConfig,
)
from speech_to_speech.datamodule.dataset.speech import (
    DatasetConfig,
    DatasetName,
    DualAudioDataset,
)
from speech_to_speech.datamodule.contract import DatasetRuntime


class _TaggedDataset(MapStyleABC):
    def __init__(self, tag: str, samples: int = 2) -> None:
        self.tag = tag
        self.samples = samples

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> Sample:
        if index < 0:
            index += self.samples
        if index < 0 or index >= self.samples:
            raise IndexError(index)
        return cast(Sample, {})


class _Producer:
    def __init__(self, ready: _TaggedDataset, marker: Path) -> None:
        self.ready = ready
        self.marker = marker
        self.loads = 0

    def __call__(self) -> None:
        self.marker.touch()

    def load(self) -> _TaggedDataset:
        self.loads += 1
        return self.ready


class _BlockingProducer:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __call__(self) -> None:
        self.marker.touch()
        while True:
            time.sleep(1.0)

    def load(self) -> _TaggedDataset:
        raise AssertionError("a cancelled producer must not be loaded")


class _LargeFailureProducer:
    def __call__(self) -> None:
        raise RuntimeError("worker failure " + "x" * 1_000_000)

    def load(self) -> _TaggedDataset:
        raise AssertionError("a failed producer must not be loaded")


class AssetMaterializationTest(unittest.TestCase):
    def test_enabled_config_requires_waveform_fallback_and_one_device(self) -> None:
        with self.assertRaisesRegex(ValueError, "encode_missing_codes=true"):
            SpeechConfig(
                codec="longcat",
                dataloader=DataLoaderConfig(batch_size=1, num_workers=0),
                materialization=_materialization(Path("/tmp/output")),
            )
        with self.assertRaisesRegex(ValueError, "explicit device"):
            AssetMaterializationConfig(
                enabled=True,
                output_root="/tmp/output",
                device="auto",
                provider_id="provider-v1",
            )
        with self.assertRaisesRegex(ValueError, "unsupported.*codec_view"):
            AssetMaterializationConfig(codec_view="missing-codec-view")

    def test_codec_view_must_match_the_training_runtime(self) -> None:
        with self.assertRaisesRegex(ValueError, "codec_view must match"):
            resolve_workspace_asset(
                DatasetConfig(),
                _runtime(),
                _materialization(
                    Path("/tmp/output"),
                    codec_view=AudioView.STABLE.value,
                ),
            )

    def test_coupled_asset_request_id_remains_contract_v2(self) -> None:
        request = AssetRequest(
            dataset="wmt19_tts",
            source_root=Path("/source"),
            output_root=Path("/output"),
            split="train",
            codec="longcat",
            codec_view=AudioView.LONGCAT,
            filter_policy="speech_translation_v1",
            input_id="input-v1",
            provider_id="provider-v1",
        )

        self.assertEqual(
            request.id,
            "s2s-asset-3b8fa7f4109fa6d9533c585cd5eb8971"
            "5ddf2015181fd986aa7de66786473e12",
        )
        self.assertNotEqual(
            request.id,
            replace(
                request,
                codec="unicodec",
                codec_view=AudioView.UNICODEC,
            ).id,
        )
        self.assertNotEqual(
            request.id,
            replace(request, provider_id="provider-v2").id,
        )

    def test_unsupported_codec_view_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "supports BiCodec structured units.*frame-code codec views.*waveform",
        ):
            resolve_workspace_asset(
                DatasetConfig(),
                _runtime(codec_name="waveform", audio_view=AudioView.WAVEFORM),
                _materialization(
                    Path("/tmp/output"),
                    codec_view=AudioView.WAVEFORM.value,
                ),
            )

    def test_bicodec_workspace_hit_is_supported(self) -> None:
        existing = _TaggedDataset("workspace-bicodec")
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            with (
                _workspace(root),
                patch.object(asset, "_load_codec_dataset", return_value=existing) as load,
            ):
                resolution = resolve_workspace_asset(
                    DatasetConfig(root=str(root), filter="speech_translation_v1"),
                    _runtime(codec_name="bicodec", audio_view=AudioView.BICODEC),
                    _materialization(
                        Path(directory) / "output",
                        codec_view=AudioView.BICODEC.value,
                    ),
                )

        self.assertIs(resolution.dataset, existing)
        self.assertIsNone(resolution.job)
        load.assert_called_once_with(
            DatasetName.WMT19_TTS,
            root.resolve(),
            split="train",
            view=AudioView.BICODEC,
            filter_policy="speech_translation_v1",
            missing_ok=True,
        )

    def test_same_codec_view_with_independent_bpe_reuses_workspace_asset(self) -> None:
        existing = _TaggedDataset("shared-longcat")
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            with (
                _workspace(root),
                patch.object(asset, "_load_codec_dataset", return_value=existing) as load,
                patch.object(asset, "WorkspaceWaveformFactory") as waveform_factory,
            ):
                resolution = resolve_workspace_asset(
                    DatasetConfig(root=str(root), filter="speech_translation_v1"),
                    _same_codec_decoupled_runtime(input_bpe="input-bpe-v1"),
                    _materialization(Path(directory) / "output"),
                )

        self.assertIs(resolution.dataset, existing)
        self.assertIsNone(resolution.request_id)
        self.assertIsNone(resolution.job)
        load.assert_called_once_with(
            DatasetName.WMT19_TTS,
            root.resolve(),
            split="train",
            view=AudioView.LONGCAT,
            filter_policy="speech_translation_v1",
            missing_ok=True,
        )
        waveform_factory.assert_not_called()

    def test_distinct_codecs_cannot_share_one_audio_view(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "distinct input/output codecs must use distinct audio views",
        ):
            resolve_workspace_asset(
                DatasetConfig(),
                _decoupled_runtime(
                    input_codec_name="other-bicodec",
                    input_audio_view=AudioView.BICODEC,
                ),
                _materialization(
                    Path("/tmp/output"),
                    codec_view=AudioView.BICODEC.value,
                ),
            )

    def test_decoupled_workspace_hits_return_dual_dataset_without_job(self) -> None:
        input_dataset = _TaggedDataset("workspace-glm4")
        output_dataset = _TaggedDataset("workspace-bicodec")

        def load_codec(
            dataset_name: DatasetName,
            root: Path,
            *,
            split: str,
            view: AudioView,
            filter_policy: str | None,
            missing_ok: bool,
        ) -> _TaggedDataset:
            del dataset_name, root, split, filter_policy, missing_ok
            if view is AudioView.GLM4:
                return input_dataset
            if view is AudioView.BICODEC:
                return output_dataset
            raise AssertionError(view)

        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            with (
                _workspace(root),
                patch.object(asset, "_load_codec_dataset", side_effect=load_codec) as load,
                patch.object(asset, "WorkspaceWaveformFactory") as waveform_factory,
                patch.object(asset, "_load_materialized_dataset") as materialized,
                patch.object(asset, "_provider_factory") as provider,
            ):
                resolution = resolve_workspace_asset(
                    DatasetConfig(root=str(root), filter="speech_translation_v1"),
                    _decoupled_runtime(),
                    _materialization(
                        Path(directory) / "output",
                        codec_view=AudioView.BICODEC.value,
                    ),
                )

        dual = cast(DualAudioDataset, resolution.dataset)
        self.assertIsInstance(dual, DualAudioDataset)
        self.assertIs(dual.input_dataset, input_dataset)
        self.assertIs(dual.output_dataset, output_dataset)
        self.assertIs(dual.input_view, AudioView.GLM4)
        self.assertIsNone(resolution.request_id)
        self.assertIsNone(resolution.job)
        self.assertEqual(
            load.call_args_list,
            [
                call(
                    DatasetName.WMT19_TTS,
                    root.resolve(),
                    split="train",
                    view=AudioView.GLM4,
                    filter_policy="speech_translation_v1",
                    missing_ok=True,
                ),
                call(
                    DatasetName.WMT19_TTS,
                    root.resolve(),
                    split="train",
                    view=AudioView.BICODEC,
                    filter_policy="speech_translation_v1",
                    missing_ok=True,
                ),
            ],
        )
        waveform_factory.assert_not_called()
        materialized.assert_not_called()
        provider.assert_not_called()

    def test_glm4_workspace_store_uses_direct_anydataset_loader(self) -> None:
        base = _TaggedDataset("glm4", samples=3)
        selection = IndexSelection(_TaggedDataset("text", samples=3), [0, 2])
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            store = root / AudioView.GLM4.value
            store.mkdir(parents=True)
            with (
                _workspace(root) as moss_tts,
                patch.object(asset, "read_store_manifest"),
                patch.object(
                    asset.AnyDataset,
                    "from_store",
                    return_value=base,
                ) as from_store,
            ):
                moss_tts.text.return_value.filter.return_value.load.return_value = selection
                loaded = asset._load_codec_dataset(
                    DatasetName.WMT19_TTS,
                    root,
                    split="train",
                    view=AudioView.GLM4,
                    filter_policy="speech_translation_v1",
                    missing_ok=False,
                )

        selected = cast(IndexSelection, loaded)
        self.assertIsInstance(selected, IndexSelection)
        self.assertIs(selected.dataset, base)
        self.assertEqual(selected.indices, (0, 2))
        from_store.assert_called_once_with(store, split="train")
        moss_tts.text.assert_called_once_with(root=root, split="train")
        moss_tts.text.return_value.filter.assert_called_once_with(
            "speech_translation_v1"
        )
        moss_tts.codec.assert_not_called()

    def test_bicodec_request_uses_structured_provider_factory(self) -> None:
        request = AssetRequest(
            dataset="wmt19_tts",
            source_root=Path("/source"),
            output_root=Path("/output"),
            split="train",
            codec="bicodec",
            codec_view=AudioView.BICODEC,
            filter_policy="speech_translation_v1",
            input_id="input-v1",
            provider_id="provider-v1",
        )

        provider = asset._provider_factory(request)
        producer = WorkspaceCodecProducer(
            request,
            Mock(return_value=_TaggedDataset("source")),
            _materialization(Path("/output")),
        )

        self.assertIsInstance(provider, asset.BiCodecProviderFactory)
        self.assertEqual(provider.codec, "bicodec")
        self.assertEqual(producer.output_dir, request.asset_root / "bicodec")

    def test_glm4_request_uses_tokenizer_only_provider_factory(self) -> None:
        request = AssetRequest(
            dataset="wmt19_tts",
            source_root=Path("/source"),
            output_root=Path("/output"),
            split="train",
            codec="glm4",
            codec_view=AudioView.GLM4,
            filter_policy="speech_translation_v1",
            input_id="input-v1",
            provider_id="provider-v1",
        )
        tokenizer = SimpleNamespace(
            spec=SimpleNamespace(
                view="glm4",
                frame_codebook_sizes=(16_384,),
            ),
            backend=SimpleNamespace(),
            tokenize=Mock(),
        )

        factory = asset._provider_factory(request)
        self.assertIsInstance(factory, asset.AudioTokenizerProviderFactory)
        with patch.object(
            asset,
            "load_audio_tokenizer",
            return_value=tokenizer,
        ) as load:
            provider = factory("cpu")

        self.assertIsInstance(provider, asset.AudioTokenizerProvider)
        self.assertIs(provider.tokenizer, tokenizer)
        self.assertIs(provider.output, AudioView.GLM4)
        load.assert_called_once_with("glm4", device="cpu")

    def test_workspace_hit_reuses_filtered_codec_without_job(self) -> None:
        existing = _TaggedDataset("workspace")
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            materialization = _materialization(Path(directory) / "output")
            with (
                _workspace(root) as moss_tts,
                patch.object(asset, "_load_codec_dataset", return_value=existing) as load,
                patch.object(asset, "_source_factory") as source_factory,
                patch.object(asset, "_load_materialized_dataset") as load_materialized,
            ):
                resolution = resolve_workspace_asset(
                    DatasetConfig(root=str(root), filter="speech_translation_v1"),
                    _runtime(),
                    materialization,
                )

        self.assertIs(resolution.dataset, existing)
        self.assertIsNone(resolution.request_id)
        self.assertIsNone(resolution.job)
        moss_tts.dataset_root.assert_called_once_with(str(root))
        load.assert_called_once_with(
            DatasetName.WMT19_TTS,
            root.resolve(),
            split="train",
            view=AudioView.LONGCAT,
            filter_policy="speech_translation_v1",
            missing_ok=True,
        )
        source_factory.assert_not_called()
        load_materialized.assert_not_called()

    def test_workspace_miss_builds_filtered_composite_job(self) -> None:
        fallback = _TaggedDataset("waveform")
        source = Mock(return_value=fallback)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            output = Path(directory) / "output"
            materialization = _materialization(output)
            with (
                _workspace(root),
                patch.object(asset, "_load_codec_dataset", return_value=None),
                patch.object(
                    asset,
                    "WorkspaceWaveformFactory",
                    return_value=source,
                ) as waveform_factory,
                patch.object(
                    asset,
                    "_workspace_input_id",
                    return_value="workspace-input-v1",
                ) as input_id,
                patch.object(asset, "_load_materialized_dataset", return_value=None),
            ):
                resolution = resolve_workspace_asset(
                    DatasetConfig(root=str(root), filter="speech_translation_v1"),
                    _runtime(),
                    materialization,
                )

        self.assertIs(resolution.dataset, fallback)
        self.assertIsNotNone(resolution.request_id)
        job = cast(BackgroundAssetJob, resolution.job)
        self.assertIs(job.fallback, fallback)
        self.assertEqual(job.request.filter_policy, "speech_translation_v1")
        self.assertEqual(job.request.input_id, "workspace-input-v1")
        self.assertEqual(
            cast(WorkspaceCodecProducer, job.producer).output_dir,
            output.resolve() / cast(str, resolution.request_id) / "longcat",
        )
        waveform_factory.assert_called_once_with(
            DatasetName.WMT19_TTS,
            root.resolve(),
            "train",
            "speech_translation_v1",
        )
        source.assert_called_once_with()
        input_id.assert_called_once_with(
            root.resolve() / "base",
            fallback,
            dataset_name=DatasetName.WMT19_TTS,
            split="train",
            filter_policy="speech_translation_v1",
        )

    def test_bpe_change_keeps_one_background_asset_job_and_request_id(self) -> None:
        fallback = _TaggedDataset("waveform")
        source = Mock(return_value=fallback)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            materialization = _materialization(Path(directory) / "output")
            with (
                _workspace(root),
                patch.object(asset, "_load_codec_dataset", return_value=None) as load,
                patch.object(
                    asset,
                    "WorkspaceWaveformFactory",
                    return_value=source,
                ),
                patch.object(
                    asset,
                    "_workspace_input_id",
                    return_value="workspace-input-v1",
                ),
                patch.object(asset, "_load_materialized_dataset", return_value=None),
            ):
                first = resolve_workspace_asset(
                    DatasetConfig(root=str(root), filter="speech_translation_v1"),
                    _same_codec_decoupled_runtime(input_bpe="input-bpe-v1"),
                    materialization,
                )
                second = resolve_workspace_asset(
                    DatasetConfig(root=str(root), filter="speech_translation_v1"),
                    _same_codec_decoupled_runtime(input_bpe="input-bpe-v2"),
                    materialization,
                )

        first_job = cast(BackgroundAssetJob, first.job)
        second_job = cast(BackgroundAssetJob, second.job)
        self.assertIsInstance(first_job, BackgroundAssetJob)
        self.assertIsInstance(second_job, BackgroundAssetJob)
        self.assertNotIsInstance(first_job, BackgroundDualAssetJob)
        self.assertNotIsInstance(second_job, BackgroundDualAssetJob)
        self.assertEqual(first.request_id, second.request_id)
        self.assertEqual(first_job.request.id, second_job.request.id)
        self.assertEqual(load.call_count, 2)

    def test_partial_dual_hit_keeps_input_codes_in_waveform_fallback(self) -> None:
        input_dataset = _TaggedDataset("workspace-glm4")
        fallback = _TaggedDataset("waveform")
        source = Mock(return_value=fallback)

        def load_codec(
            dataset_name: DatasetName,
            root: Path,
            *,
            split: str,
            view: AudioView,
            filter_policy: str | None,
            missing_ok: bool,
        ) -> _TaggedDataset | None:
            del dataset_name, root, split, filter_policy, missing_ok
            return input_dataset if view is AudioView.GLM4 else None

        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            output = Path(directory) / "output"
            with (
                _workspace(root),
                patch.object(asset, "_load_codec_dataset", side_effect=load_codec),
                patch.object(
                    asset,
                    "WorkspaceWaveformFactory",
                    return_value=source,
                ),
                patch.object(
                    asset,
                    "_workspace_input_id",
                    return_value="workspace-input-v1",
                ),
                patch.object(asset, "_load_materialized_dataset", return_value=None),
                patch.object(
                    asset,
                    "_provider_factory",
                    wraps=asset._provider_factory,
                ) as provider,
            ):
                resolution = resolve_workspace_asset(
                    DatasetConfig(root=str(root), filter="speech_translation_v1"),
                    _decoupled_runtime(),
                    _materialization(
                        output,
                        codec_view=AudioView.BICODEC.value,
                    ),
                )

        fallback_dataset = cast(DualAudioDataset, resolution.dataset)
        self.assertIsInstance(fallback_dataset, DualAudioDataset)
        self.assertIs(fallback_dataset.input_dataset, input_dataset)
        self.assertIs(fallback_dataset.output_dataset, fallback)
        self.assertIs(fallback_dataset.input_view, AudioView.GLM4)
        job = cast(BackgroundDualAssetJob, resolution.job)
        self.assertIsInstance(job, BackgroundDualAssetJob)
        self.assertIs(job.fallback, fallback_dataset)
        self.assertEqual(job.request.input.codec, "glm4")
        self.assertIs(job.request.input.codec_view, AudioView.GLM4)
        self.assertEqual(job.request.output.codec, "bicodec")
        self.assertIs(job.request.output.codec_view, AudioView.BICODEC)
        self.assertEqual(job.request.input.input_id, "workspace-input-v1")
        self.assertEqual(job.request.output.input_id, "workspace-input-v1")
        producer = cast(DualWorkspaceCodecProducer, job.producer)
        self.assertTrue(producer.input_from_workspace)
        self.assertFalse(producer.output_from_workspace)
        self.assertIs(producer.source, source)
        provider.assert_called_once_with(job.request.output)
        source.assert_called_once_with()

    def test_corrupt_workspace_codec_is_not_treated_as_a_miss(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            with (
                _workspace(root),
                patch.object(
                    asset,
                    "_load_codec_dataset",
                    side_effect=ValueError("codec manifest mismatch"),
                ),
                patch.object(
                    asset,
                    "WorkspaceWaveformFactory",
                ) as waveform_factory,
            ):
                with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                    resolve_workspace_asset(
                        DatasetConfig(root=str(root)),
                        _runtime(),
                        _materialization(Path(directory) / "output"),
                    )

        waveform_factory.assert_not_called()

    def test_materialized_store_requires_exact_provenance(self) -> None:
        with TemporaryDirectory() as directory:
            request = AssetRequest(
                dataset="wmt19_tts",
                source_root=Path(directory) / "source",
                output_root=Path(directory) / "output",
                split="train",
                codec="longcat",
                codec_view=AudioView.LONGCAT,
                filter_policy="speech_translation_v1",
                input_id="input-v1",
                provider_id="provider-v1",
            )
            output = request.asset_root / "longcat"
            output.mkdir(parents=True)
            (output / ".ready").touch()
            manifest = SimpleNamespace(
                provenance={"input_id": "stale", "provider_id": "provider-v1"}
            )
            with (
                patch.object(asset, "read_store_manifest", return_value=manifest),
                patch.object(asset, "_load_codec_dataset") as load,
            ):
                with self.assertRaisesRegex(ValueError, "provenance"):
                    asset._load_materialized_dataset(request, missing_ok=True)

        load.assert_not_called()

    def test_concurrent_ready_publish_wins_over_local_writer_error(self) -> None:
        ready = _TaggedDataset("ready")
        with TemporaryDirectory() as directory:
            request = AssetRequest(
                dataset="wmt19_tts",
                source_root=Path(directory) / "source",
                output_root=Path(directory) / "output",
                split="train",
                codec="longcat",
                codec_view=AudioView.LONGCAT,
                filter_policy="speech_translation_v1",
                input_id="input-v1",
                provider_id="provider-v1",
            )
            producer = WorkspaceCodecProducer(
                request,
                Mock(return_value=_TaggedDataset("source")),
                _materialization(Path(directory) / "output"),
            )
            with (
                patch.object(
                    asset,
                    "_load_materialized_dataset",
                    side_effect=[None, ready],
                ) as load,
                patch.object(asset, "ViewMaterializer") as materializer,
            ):
                materializer.return_value.write.side_effect = ValueError(
                    "target directory became non-empty"
                )

                producer()

        self.assertEqual(load.call_count, 2)

    def test_dual_producer_validates_both_sides_before_publish(self) -> None:
        request = _dual_asset_request()
        input_dataset = _TaggedDataset("input")
        output_dataset: _TaggedDataset | None = None

        def load_materialized(
            child: AssetRequest,
            *,
            missing_ok: bool,
        ) -> _TaggedDataset | None:
            dataset = (
                input_dataset
                if child.codec_view is AudioView.GLM4
                else output_dataset
            )
            if dataset is None and not missing_ok:
                raise FileNotFoundError(child.codec_view.value)
            return dataset

        producer = DualWorkspaceCodecProducer(
            request,
            Mock(return_value=_TaggedDataset("source")),
            _materialization(Path("/output")),
            input_from_workspace=False,
            output_from_workspace=False,
        )
        with (
            patch.object(asset, "WorkspaceCodecProducer") as child_producer,
            patch.object(
                asset,
                "_load_materialized_dataset",
                side_effect=load_materialized,
            ),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "bicodec"):
                producer()

            output_dataset = _TaggedDataset("output")
            producer()
            loaded = producer.load()

        self.assertEqual(child_producer.call_count, 4)
        dual = cast(DualAudioDataset, loaded)
        self.assertIs(dual.input_dataset, input_dataset)
        self.assertIs(dual.output_dataset, output_dataset)

    def test_dual_producer_rejects_misaligned_store_lengths(self) -> None:
        request = _dual_asset_request()
        producer = DualWorkspaceCodecProducer(
            request,
            Mock(return_value=_TaggedDataset("source")),
            _materialization(Path("/output")),
            input_from_workspace=False,
            output_from_workspace=False,
        )

        def load_materialized(
            child: AssetRequest,
            *,
            missing_ok: bool,
        ) -> _TaggedDataset:
            del missing_ok
            if child.codec_view is AudioView.GLM4:
                return _TaggedDataset("input", samples=2)
            return _TaggedDataset("output", samples=3)

        with patch.object(
            asset,
            "_load_materialized_dataset",
            side_effect=load_materialized,
        ):
            with self.assertRaisesRegex(ValueError, "equal lengths"):
                producer.load()

    def test_missing_glm4_input_uses_its_tokenizer_provider(self) -> None:
        output_dataset = _TaggedDataset("workspace-bicodec")
        fallback = _TaggedDataset("waveform")

        def load_codec(
            dataset_name: DatasetName,
            root: Path,
            *,
            split: str,
            view: AudioView,
            filter_policy: str | None,
            missing_ok: bool,
        ) -> _TaggedDataset | None:
            del dataset_name, root, split, filter_policy, missing_ok
            return output_dataset if view is AudioView.BICODEC else None

        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            with (
                _workspace(root),
                patch.object(asset, "_load_codec_dataset", side_effect=load_codec),
                patch.object(
                    asset,
                    "WorkspaceWaveformFactory",
                    return_value=Mock(return_value=fallback),
                ),
                patch.object(
                    asset,
                    "_workspace_input_id",
                    return_value="workspace-input-v1",
                ),
                patch.object(asset, "_load_materialized_dataset", return_value=None),
                patch.object(
                    asset,
                    "AudioTokenizerProviderFactory",
                ) as input_provider,
                patch.object(asset, "BiCodecProviderFactory") as output_provider,
            ):
                resolution = resolve_workspace_asset(
                    DatasetConfig(root=str(root)),
                    _decoupled_runtime(),
                    _materialization(
                        Path(directory) / "output",
                        codec_view=AudioView.BICODEC.value,
                    ),
                )

        self.assertIsNotNone(resolution.job)
        input_provider.assert_called_once_with("glm4", AudioView.GLM4)
        output_provider.assert_not_called()

    def test_missing_workspace_filter_does_not_fall_back_to_unfiltered_data(self) -> None:
        source = Mock(side_effect=asset._WorkspaceFilterMissing("selection missing"))
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            with (
                _workspace(root),
                patch.object(asset, "_load_codec_dataset", return_value=None),
                patch.object(
                    asset,
                    "WorkspaceWaveformFactory",
                    return_value=source,
                ) as waveform_factory,
                patch.object(asset, "_load_materialized_dataset") as load_materialized,
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError,
                    "workspace filter 'missing-filter'.*not published.*unfiltered",
                ):
                    resolve_workspace_asset(
                        DatasetConfig(root=str(root), filter="missing-filter"),
                        _runtime(),
                        _materialization(Path(directory) / "output"),
                    )

        waveform_factory.assert_called_once_with(
            DatasetName.WMT19_TTS,
            root.resolve(),
            "train",
            "missing-filter",
        )
        source.assert_called_once_with()
        load_materialized.assert_not_called()

    def test_stream_source_can_supply_a_missing_filter_route(self) -> None:
        fallback = _TaggedDataset("stream")
        source = Mock(return_value=fallback)
        workspace_source = Mock(side_effect=asset._WorkspaceFilterMissing("selection missing"))
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            materialization = _materialization(
                Path(directory) / "output",
                source_factory="fixture.stream:build",
                input_id="stream-filter-v1",
            )
            with (
                _workspace(root),
                patch.object(asset, "_load_codec_dataset", return_value=None),
                patch.object(asset, "_source_factory", return_value=source) as source_factory,
                patch.object(asset, "_load_materialized_dataset", return_value=None),
                patch.object(
                    asset,
                    "WorkspaceWaveformFactory",
                    return_value=workspace_source,
                ) as waveform_factory,
            ):
                resolution = resolve_workspace_asset(
                    DatasetConfig(root=str(root), filter="stream-filter"),
                    _runtime(),
                    materialization,
                )

        job = cast(BackgroundAssetJob, resolution.job)
        self.assertIs(resolution.dataset, fallback)
        self.assertEqual(job.request.filter_policy, "stream-filter")
        self.assertEqual(job.request.input_id, "stream-filter-v1")
        self.assertEqual(job.request.source_factory, "fixture.stream:build")
        source_factory.assert_called_once_with(job.request)
        source.assert_called_once_with()
        waveform_factory.assert_called_once_with(
            DatasetName.WMT19_TTS,
            root.resolve(),
            "train",
            "stream-filter",
        )
        workspace_source.assert_called_once_with()

    def test_dual_source_factory_is_opened_once_and_shared(self) -> None:
        fallback = _TaggedDataset("stream")
        source = Mock(return_value=fallback)
        workspace_source = Mock(
            side_effect=asset._WorkspaceSourceMissing("base missing")
        )
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            materialization = _materialization(
                Path(directory) / "output",
                source_factory="fixture.stream:build",
                input_id="stream-filter-v1",
                codec_view=AudioView.BICODEC.value,
            )
            with (
                _workspace(root),
                patch.object(asset, "_load_codec_dataset", return_value=None),
                patch.object(asset, "_load_materialized_dataset", return_value=None),
                patch.object(asset, "_source_factory", return_value=source) as builder,
                patch.object(
                    asset,
                    "WorkspaceWaveformFactory",
                    return_value=workspace_source,
                ),
            ):
                resolution = resolve_workspace_asset(
                    DatasetConfig(root=str(root), filter="stream-filter"),
                    _decoupled_runtime(
                        input_codec_name="longcat",
                        input_audio_view=AudioView.LONGCAT,
                    ),
                    materialization,
                )

        job = cast(BackgroundDualAssetJob, resolution.job)
        producer = cast(DualWorkspaceCodecProducer, job.producer)
        builder.assert_called_once_with(job.request.output)
        source.assert_called_once_with()
        self.assertIs(producer.source, source)
        input_request = job.request.input
        output_request = job.request.output
        self.assertEqual(input_request.source_root, output_request.source_root)
        self.assertEqual(input_request.split, output_request.split)
        self.assertEqual(input_request.filter_policy, output_request.filter_policy)
        self.assertEqual(input_request.input_id, output_request.input_id)
        self.assertEqual(input_request.source_factory, output_request.source_factory)

        changed_input = DualAssetRequest(
            replace(
                input_request,
                codec="unicodec",
                codec_view=AudioView.UNICODEC,
            ),
            output_request,
        )
        changed_output = DualAssetRequest(
            input_request,
            replace(
                output_request,
                codec="longcat",
                codec_view=AudioView.LONGCAT,
            ),
        )
        self.assertNotEqual(job.request.id, changed_input.id)
        self.assertNotEqual(job.request.id, changed_output.id)

    def test_ready_stream_asset_does_not_reopen_its_source(self) -> None:
        ready = _TaggedDataset("ready")
        workspace_source = Mock(side_effect=asset._WorkspaceFilterMissing("selection missing"))
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            materialization = _materialization(
                Path(directory) / "output",
                source_factory="fixture.stream:build",
                input_id="stream-filter-v1",
            )
            with (
                _workspace(root),
                patch.object(asset, "_load_codec_dataset", return_value=None),
                patch.object(
                    asset,
                    "_load_materialized_dataset",
                    return_value=ready,
                ),
                patch.object(asset, "_source_factory") as source_factory,
                patch.object(
                    asset,
                    "WorkspaceWaveformFactory",
                    return_value=workspace_source,
                ),
            ):
                resolution = resolve_workspace_asset(
                    DatasetConfig(root=str(root), filter="stream-filter"),
                    _runtime(),
                    materialization,
                )

        self.assertIs(resolution.dataset, ready)
        self.assertIsNone(resolution.job)
        source_factory.assert_not_called()
        workspace_source.assert_called_once_with()

    def test_streaming_s2st_hit_uses_its_own_codec_resource(self) -> None:
        existing = _TaggedDataset("streaming-codec", samples=4)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "streaming"
            with (
                _streaming_workspace(root) as streaming_s2st,
                patch.object(asset, "_load_codec_dataset", return_value=existing) as load,
            ):
                resolution = resolve_workspace_asset(
                    DatasetConfig(
                        name=DatasetName.STREAMING_S2ST,
                        root=str(root),
                        filter=None,
                    ),
                    _runtime(),
                    _materialization(Path(directory) / "output"),
                )

        self.assertIs(resolution.dataset, existing)
        self.assertIsNone(resolution.job)
        streaming_s2st.dataset_root.assert_called_once_with(str(root))
        load.assert_called_once_with(
            DatasetName.STREAMING_S2ST,
            root.resolve(),
            split="train",
            view=AudioView.LONGCAT,
            filter_policy=None,
            missing_ok=True,
        )

    def test_streaming_s2st_source_factory_preserves_bidirectional_length(self) -> None:
        fallback = _TaggedDataset("bidirectional-waveform", samples=4)
        source = Mock(return_value=fallback)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "streaming"
            materialization = _materialization(
                Path(directory) / "output",
                source_factory="fixture.streaming_s2st:build",
                input_id="wmt19-bidirectional-v1",
            )
            with (
                _streaming_workspace(root),
                patch.object(asset, "_load_codec_dataset", return_value=None),
                patch.object(asset, "_source_factory", return_value=source),
                patch.object(asset, "_load_materialized_dataset", return_value=None),
            ):
                resolution = resolve_workspace_asset(
                    DatasetConfig(
                        name=DatasetName.STREAMING_S2ST,
                        root=str(root),
                        filter=None,
                    ),
                    _runtime(),
                    materialization,
                )

        job = cast(BackgroundAssetJob, resolution.job)
        self.assertEqual(len(cast(_TaggedDataset, resolution.dataset)), 4)
        self.assertEqual(job.request.dataset, DatasetName.STREAMING_S2ST.value)
        self.assertIsNone(job.request.filter_policy)
        self.assertEqual(job.request.input_id, "wmt19-bidirectional-v1")
        self.assertEqual(job.request.source_factory, "fixture.streaming_s2st:build")
        source.assert_called_once_with()

    def test_ready_streaming_asset_reopens_with_streaming_codec_resource(self) -> None:
        ready = _TaggedDataset("ready-bidirectional-codec", samples=4)
        with TemporaryDirectory() as directory:
            request = AssetRequest(
                dataset=DatasetName.STREAMING_S2ST.value,
                source_root=Path(directory) / "source",
                output_root=Path(directory) / "output",
                split="train",
                codec="longcat",
                codec_view=AudioView.LONGCAT,
                filter_policy=None,
                input_id="wmt19-bidirectional-v1",
                provider_id="longcat-v1",
            )
            output = request.asset_root / "longcat"
            output.mkdir(parents=True)
            (output / ".ready").touch()
            manifest = SimpleNamespace(
                provenance={
                    "input_id": request.input_id,
                    "provider_id": request.provider_id,
                }
            )
            with (
                _streaming_workspace(request.asset_root) as streaming_s2st,
                patch.object(asset, "read_store_manifest", return_value=manifest),
            ):
                streaming_s2st.codec.return_value.load.return_value = ready
                loaded = asset._load_materialized_dataset(request, missing_ok=False)

        self.assertIs(loaded, ready)
        self.assertEqual(len(cast(_TaggedDataset, loaded)), 4)
        streaming_s2st.codec.assert_called_once_with(
            "longcat",
            root=request.asset_root,
            split="train",
        )

    def test_background_job_finishes_and_loads_ready_dataset(self) -> None:
        fallback = _TaggedDataset("fallback")
        ready = _TaggedDataset("ready")
        with TemporaryDirectory() as directory:
            marker = Path(directory) / "produced"
            producer = _Producer(ready, marker)
            request = AssetRequest(
                dataset="wmt19_tts",
                source_root=Path("/source"),
                output_root=Path("/output"),
                split="train",
                codec="longcat",
                codec_view=AudioView.LONGCAT,
                filter_policy="speech_translation_v1",
                input_id="input-v1",
                provider_id="provider-v1",
            )
            job = BackgroundAssetJob(
                request,
                fallback,
                producer,
            )

            job.start(owner=True)
            self.assertIs(job.phase, AssetPhase.MATERIALIZING)
            self.assertTrue(job._process is not None and not job._process.daemon)
            job.finish(owner=True)
            loaded = job.load_ready()
            job.close()

            self.assertTrue(marker.is_file())

        self.assertIs(job.phase, AssetPhase.READY)
        self.assertIs(loaded, ready)
        self.assertEqual(producer.loads, 1)

    def test_background_job_close_terminates_running_worker(self) -> None:
        with TemporaryDirectory() as directory:
            marker = Path(directory) / "started"
            request = AssetRequest(
                dataset="wmt19_tts",
                source_root=Path("/source"),
                output_root=Path("/output"),
                split="train",
                codec="longcat",
                codec_view=AudioView.LONGCAT,
                filter_policy=None,
                input_id="input-v1",
                provider_id="provider-v1",
            )
            job = BackgroundAssetJob(
                request,
                _TaggedDataset("fallback"),
                _BlockingProducer(marker),
            )
            job.start(owner=True)
            deadline = time.monotonic() + 10.0
            while not marker.is_file() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(marker.is_file())

            started = time.monotonic()
            job.close()
            job.close()

        self.assertLess(time.monotonic() - started, 6.0)
        self.assertIs(job.phase, AssetPhase.FAILED)

    def test_background_job_drains_large_worker_error_before_join(self) -> None:
        request = AssetRequest(
            dataset="wmt19_tts",
            source_root=Path("/source"),
            output_root=Path("/output"),
            split="train",
            codec="longcat",
            codec_view=AudioView.LONGCAT,
            filter_policy=None,
            input_id="input-v1",
            provider_id="provider-v1",
        )
        job = BackgroundAssetJob(
            request,
            _TaggedDataset("fallback"),
            _LargeFailureProducer(),
        )

        started = time.monotonic()
        job.start(owner=True)
        with self.assertRaisesRegex(
            RuntimeError,
            "asset materialization worker failed: RuntimeError: worker failure",
        ):
            job.finish(owner=True)
        job.close()

        self.assertLess(time.monotonic() - started, 10.0)
        self.assertIs(job.phase, AssetPhase.FAILED)


def _runtime(
    *,
    codec_name: str = "longcat",
    audio_view: AudioView = AudioView.LONGCAT,
) -> DatasetRuntime:
    return cast(
        DatasetRuntime,
        SimpleNamespace(
            input_audio_decoupled=False,
            input_codec_name=codec_name,
            input_audio_view=audio_view,
            codec_name=codec_name,
            audio_view=audio_view,
        ),
    )


def _same_codec_decoupled_runtime(*, input_bpe: str) -> DatasetRuntime:
    return cast(
        DatasetRuntime,
        SimpleNamespace(
            input_audio_decoupled=True,
            input_codec_name="longcat",
            input_audio_view=AudioView.LONGCAT,
            input_audio_tokenizer=SimpleNamespace(identity=input_bpe),
            codec_name="longcat",
            audio_view=AudioView.LONGCAT,
            audio_tokenizer=SimpleNamespace(identity="output-bpe"),
        ),
    )


def _decoupled_runtime(
    *,
    input_codec_name: str = "glm4",
    input_audio_view: AudioView = AudioView.GLM4,
    codec_name: str = "bicodec",
    audio_view: AudioView = AudioView.BICODEC,
) -> DatasetRuntime:
    return cast(
        DatasetRuntime,
        SimpleNamespace(
            input_audio_decoupled=True,
            input_codec_name=input_codec_name,
            input_audio_view=input_audio_view,
            codec_name=codec_name,
            audio_view=audio_view,
        ),
    )


def _dual_asset_request() -> DualAssetRequest:
    common = {
        "dataset": "wmt19_tts",
        "source_root": Path("/source"),
        "output_root": Path("/output"),
        "split": "train",
        "filter_policy": "speech_translation_v1",
        "input_id": "input-v1",
        "provider_id": "provider-v1",
    }
    return DualAssetRequest(
        input=AssetRequest(
            **common,
            codec="glm4",
            codec_view=AudioView.GLM4,
        ),
        output=AssetRequest(
            **common,
            codec="bicodec",
            codec_view=AudioView.BICODEC,
        ),
    )


def _materialization(
    output_root: Path,
    *,
    codec_view: str | None = None,
    source_factory: str | None = None,
    input_id: str | None = None,
) -> AssetMaterializationConfig:
    return AssetMaterializationConfig(
        enabled=True,
        codec_view=codec_view,
        output_root=str(output_root),
        device="cpu",
        provider_id="longcat-test-provider-v1",
        input_id=input_id,
        source_factory=source_factory,
    )


@contextmanager
def _workspace(root: Path) -> Iterator[Mock]:
    moss_tts = Mock()
    moss_tts.dataset_root.return_value = root
    wmt19 = ModuleType("zhuyin.datasets.wmt19")
    wmt19.__dict__["moss_tts"] = moss_tts
    with patch.dict(
        sys.modules,
        {
            "zhuyin": ModuleType("zhuyin"),
            "zhuyin.datasets": ModuleType("zhuyin.datasets"),
            "zhuyin.datasets.wmt19": wmt19,
        },
    ):
        yield moss_tts


@contextmanager
def _streaming_workspace(root: Path) -> Iterator[Mock]:
    streaming_s2st = Mock()
    streaming_s2st.dataset_root.return_value = root
    wmt19 = ModuleType("zhuyin.datasets.wmt19")
    wmt19.__dict__["streaming_s2st"] = streaming_s2st
    with patch.dict(
        sys.modules,
        {
            "zhuyin": ModuleType("zhuyin"),
            "zhuyin.datasets": ModuleType("zhuyin.datasets"),
            "zhuyin.datasets.wmt19": wmt19,
        },
    ):
        yield streaming_s2st


if __name__ == "__main__":
    unittest.main()
