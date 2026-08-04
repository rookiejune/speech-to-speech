from __future__ import annotations

import sys
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Iterator, cast
from unittest.mock import Mock, patch

from anydataset.dataset import MapStyleABC
from anydataset.types import AudioView, Sample

from speech_to_speech.datamodule import asset
from speech_to_speech.datamodule.asset import (
    AssetPhase,
    AssetRequest,
    BackgroundAssetJob,
    WorkspaceCodecProducer,
    resolve_workspace_asset,
)
from speech_to_speech.datamodule.config import (
    AssetMaterializationConfig,
    DataLoaderConfig,
    SpeechConfig,
)
from speech_to_speech.datamodule.dataset.speech import DatasetConfig
from speech_to_speech.datamodule.protocol import DatasetRuntime


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

    def test_non_frame_codec_view_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "supports only frame-code codec views.*waveform",
        ):
            resolve_workspace_asset(
                DatasetConfig(),
                _runtime(codec_name="waveform", audio_view=AudioView.WAVEFORM),
                _materialization(
                    Path("/tmp/output"),
                    codec_view=AudioView.WAVEFORM.value,
                ),
            )

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
            root.resolve(),
            "train",
            "speech_translation_v1",
        )
        source.assert_called_once_with()
        input_id.assert_called_once_with(
            root.resolve() / "base",
            fallback,
            split="train",
            filter_policy="speech_translation_v1",
        )

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
            root.resolve(),
            "train",
            "stream-filter",
        )
        workspace_source.assert_called_once_with()

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
            self.assertTrue(job._process is not None and job._process.daemon)
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
            codec_name=codec_name,
            audio_view=audio_view,
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


if __name__ == "__main__":
    unittest.main()
