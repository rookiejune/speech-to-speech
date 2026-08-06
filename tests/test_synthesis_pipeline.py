from __future__ import annotations

import json
import os
import unittest
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import torch
from anydataset.types import (
    AudioItem,
    AudioView,
    Lang,
    Modality,
    Role,
    Sample,
    TextItem,
    TextMeta,
    TextView,
)

from speech_to_speech.datamodule.streaming import WorkspaceSnapshotLoader
from speech_to_speech.synthesis import InputCodec
from speech_to_speech.synthesis.pipeline import (
    CodecPair,
    Components,
    PipelineConfig,
    StagePlacement,
    StreamingSynthesisPipeline,
)
from speech_to_speech.synthesis.publisher import SnapshotPublisher
from speech_to_speech.synthesis.cache import SynthesisStageCache
from speech_to_speech.synthesis.telemetry import SynthesisTelemetry


class _Seeds:
    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Sample:
        source_language, target_language = _languages(index)
        return {
            (Role.SOURCE, Modality.TEXT): TextItem(
                views={TextView.TEXT: f"source {index}"},
                meta={TextMeta.LANG: source_language},
            ),
            (Role.TARGET, Modality.TEXT): TextItem(
                meta={TextMeta.LANG: target_language},
            ),
        }

    def reference_translation(self, index: int) -> TextItem:
        _, language = _languages(index)
        return TextItem(
            views={TextView.TEXT: f"dataset translation {index}"},
            meta={TextMeta.LANG: language},
        )


class _SourceTTS:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: Sequence[TextItem]) -> Sequence[AudioItem]:
        values = [_text(item) for item in texts]
        self.calls.append(values)
        return [_waveform(index + 1) for index in range(len(values))]


class _Translation:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self,
        texts: Sequence[TextItem],
        target_languages: Sequence[Lang],
    ) -> Sequence[TextItem]:
        values = [_text(item) for item in texts]
        self.calls.append(values)
        return [
            TextItem(
                views={TextView.TEXT: f"model translation of {text}"},
                meta={TextMeta.LANG: language},
            )
            for text, language in zip(values, target_languages)
        ]


class _TargetTTS:
    def __call__(
        self,
        texts: Sequence[TextItem],
        references: Sequence[AudioItem],
    ) -> Sequence[AudioItem]:
        assert len(texts) == len(references)
        return [_waveform(index + 11) for index in range(len(texts))]


class _Codec:
    def __init__(self, *, fail_call: int | None = None) -> None:
        self.calls = 0
        self.fail_call = fail_call

    def __call__(
        self,
        sources: Sequence[AudioItem],
        targets: Sequence[AudioItem],
    ) -> Sequence[CodecPair]:
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("injected codec interruption")
        return [
            CodecPair(_longcat(index), _longcat(index + 101))
            for index in range(len(sources))
        ]


class _BiCodec:
    def __call__(
        self,
        sources: Sequence[AudioItem],
        targets: Sequence[AudioItem],
    ) -> Sequence[CodecPair]:
        del targets
        return [
            CodecPair(_bicodec(index), _bicodec(index + 101))
            for index in range(len(sources))
        ]


class _InputCodec:
    def __init__(self, *, fail_call: int | None = None) -> None:
        self.calls = 0
        self.fail_call = fail_call

    def __call__(self, sources: Sequence[AudioItem]) -> Sequence[AudioItem]:
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("injected input codec interruption")
        return [_glm4(index) for index in range(len(sources))]


class _WrongViewInputCodec:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, sources: Sequence[AudioItem]) -> Sequence[AudioItem]:
        self.calls += 1
        return [_longcat(index) for index in range(len(sources))]


class StreamingSynthesisPipelineTest(unittest.TestCase):
    def test_input_codec_is_public_and_cacheable(self) -> None:
        self.assertIsNotNone(InputCodec)

    def test_resume_keeps_dataset_translation_out_of_training_label(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_home = os.environ.get("ANYDATASET_HOME")
            self.addCleanup(_restore_environment, "ANYDATASET_HOME", previous_home)
            os.environ["ANYDATASET_HOME"] = str(root / "anydataset")
            publisher = SnapshotPublisher(
                root,
                stream_id="pipeline-test",
                expected_samples=4,
                codec="longcat",
                split="train",
                loader=WorkspaceSnapshotLoader(codec="longcat", split="train"),
            )
            first_source = _SourceTTS()
            first = _pipeline(
                publisher,
                first_source,
                _Translation(),
                _Codec(fail_call=2),
                cache=_cache(root),
            )
            with self.assertRaisesRegex(RuntimeError, "injected codec interruption"):
                with SynthesisTelemetry(root, gpu_sample_interval_seconds=0) as telemetry:
                    first.run(telemetry)

            self.assertEqual(publisher.feed.status().catalog.sample_count, 2)
            resumed_source = _SourceTTS()
            resumed_translation = _Translation()
            resumed = _pipeline(
                publisher,
                resumed_source,
                resumed_translation,
                _Codec(),
                cache=_cache(root),
            )
            with SynthesisTelemetry(root, gpu_sample_interval_seconds=0) as telemetry:
                status = resumed.run(telemetry)

            self.assertIsNotNone(status.seal)
            self.assertEqual(status.catalog.sample_count, 4)
            self.assertEqual(
                first_source.calls,
                [["source 0", "source 1"], ["source 2", "source 3"]],
            )
            self.assertEqual(resumed_source.calls, [])
            self.assertEqual(resumed_translation.calls, [])
            published = publisher.feed.published([0, 3])
            self.assertEqual(
                [item.reference_translation for item in published],
                ["dataset translation 0", "dataset translation 3"],
            )
            self.assertEqual(
                [_target_text(item.sample) for item in published],
                ["model translation of source 0", "model translation of source 3"],
            )
            self.assertTrue(
                all(
                    _target_text(item.sample) != item.reference_translation
                    for item in published
                )
            )
            events = [
                json.loads(line)
                for line in (root / "producer_telemetry.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            stages = [
                event["stage"]
                for event in events
                if event["event"] == "stage_started"
            ]
            waits = [
                event["wait"]
                for event in events
                if event["event"] == "wait_finished"
            ]

        self.assertLessEqual(
            {"source_tts", "translation", "target_tts", "codec", "snapshot_publish"},
            set(stages),
        )
        self.assertLessEqual({"source_tts_join", "translation_join"}, set(waits))

    def test_decoupled_pipeline_publishes_composed_source_views(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_home = os.environ.get("ANYDATASET_HOME")
            self.addCleanup(_restore_environment, "ANYDATASET_HOME", previous_home)
            os.environ["ANYDATASET_HOME"] = str(root / "anydataset")
            publisher = SnapshotPublisher(
                root,
                stream_id="pipeline-composed",
                expected_samples=2,
                codec="bicodec",
                input_codec="glm4",
                split="train",
                loader=WorkspaceSnapshotLoader(
                    codec="bicodec",
                    input_codec="glm4",
                    split="train",
                ),
            )
            placement = StagePlacement(device="cpu")
            pipeline = StreamingSynthesisPipeline(
                _Seeds(2),
                Components(
                    source_tts=_SourceTTS(),
                    translation=_Translation(),
                    target_tts=_TargetTTS(),
                    codec=_BiCodec(),
                    input_codec=_InputCodec(),
                ),
                publisher,
                PipelineConfig(
                    batch_size=2,
                    source_tts=placement,
                    translation=placement,
                    target_tts=placement,
                    codec=placement,
                    input_codec=placement,
                ),
            )

            with SynthesisTelemetry(root, gpu_sample_interval_seconds=0) as telemetry:
                status = pipeline.run(telemetry)

            self.assertIsNotNone(status.seal)
            sample = publisher.feed.published([0])[0].sample
            source = sample[Role.SOURCE, Modality.AUDIO]
            target = sample[Role.TARGET, Modality.AUDIO]
            assert isinstance(source, AudioItem)
            assert isinstance(target, AudioItem)
            self.assertEqual(set(source.views), {AudioView.GLM4, AudioView.BICODEC})
            self.assertEqual(set(target.views), {AudioView.BICODEC})

    def test_decoupled_pipeline_requires_input_codec_component(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            publisher = SnapshotPublisher(
                root,
                stream_id="pipeline-composed",
                expected_samples=1,
                codec="bicodec",
                input_codec="glm4",
                split="train",
                loader=WorkspaceSnapshotLoader(
                    codec="bicodec",
                    input_codec="glm4",
                    split="train",
                ),
            )
            with self.assertRaisesRegex(ValueError, "input codec component"):
                StreamingSynthesisPipeline(
                    _Seeds(1),
                    Components(
                        source_tts=_SourceTTS(),
                        translation=_Translation(),
                        target_tts=_TargetTTS(),
                        codec=_BiCodec(),
                    ),
                    publisher,
                    PipelineConfig(batch_size=1),
                )

    def test_decoupled_pipeline_resumes_with_input_codec_stage_cache(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_home = os.environ.get("ANYDATASET_HOME")
            self.addCleanup(_restore_environment, "ANYDATASET_HOME", previous_home)
            os.environ["ANYDATASET_HOME"] = str(root / "anydataset")
            publisher = SnapshotPublisher(
                root,
                stream_id="pipeline-composed-resume",
                expected_samples=4,
                codec="bicodec",
                input_codec="glm4",
                split="train",
                loader=WorkspaceSnapshotLoader(
                    codec="bicodec",
                    input_codec="glm4",
                    split="train",
                ),
            )
            first_source = _SourceTTS()
            first_translation = _Translation()
            first_input = _InputCodec(fail_call=2)
            first = _composed_pipeline(
                publisher,
                first_source,
                first_translation,
                first_input,
                cache=_composed_cache(root),
            )
            with self.assertRaisesRegex(RuntimeError, "input codec interruption"):
                with SynthesisTelemetry(root, gpu_sample_interval_seconds=0) as telemetry:
                    first.run(telemetry)

            self.assertEqual(publisher.feed.status().catalog.sample_count, 2)
            resumed_source = _SourceTTS()
            resumed_translation = _Translation()
            resumed_input = _InputCodec()
            resumed = _composed_pipeline(
                publisher,
                resumed_source,
                resumed_translation,
                resumed_input,
                cache=_composed_cache(root),
            )
            with SynthesisTelemetry(root, gpu_sample_interval_seconds=0) as telemetry:
                status = resumed.run(telemetry)

            self.assertIsNotNone(status.seal)
            self.assertEqual(status.catalog.sample_count, 4)
            self.assertEqual(resumed_source.calls, [])
            self.assertEqual(resumed_translation.calls, [])
            self.assertEqual(resumed_input.calls, 1)
            sample = publisher.feed.published([3])[0].sample
            source = sample[Role.SOURCE, Modality.AUDIO]
            assert isinstance(source, AudioItem)
            self.assertEqual(set(source.views), {AudioView.GLM4, AudioView.BICODEC})

    def test_wrong_input_codec_view_is_not_cached_and_can_be_retried(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_home = os.environ.get("ANYDATASET_HOME")
            self.addCleanup(_restore_environment, "ANYDATASET_HOME", previous_home)
            os.environ["ANYDATASET_HOME"] = str(root / "anydataset")
            publisher = SnapshotPublisher(
                root,
                stream_id="pipeline-composed-resume",
                expected_samples=4,
                codec="bicodec",
                input_codec="glm4",
                split="train",
                loader=WorkspaceSnapshotLoader(
                    codec="bicodec",
                    input_codec="glm4",
                    split="train",
                ),
            )
            wrong_input = _WrongViewInputCodec()
            broken = _composed_pipeline(
                publisher,
                _SourceTTS(),
                _Translation(),
                wrong_input,
                cache=_composed_cache(root),
            )

            with self.assertRaisesRegex(ValueError, "exactly AudioView.GLM4"):
                with SynthesisTelemetry(root, gpu_sample_interval_seconds=0) as telemetry:
                    broken.run(telemetry)

            first_batch = root / ".stage-cache" / "samples-000000000000-000000000002"
            self.assertEqual(wrong_input.calls, 1)
            self.assertFalse((first_batch / "input_codec").exists())
            self.assertTrue((first_batch / "codec").is_dir())
            self.assertEqual(publisher.feed.status().catalog.sample_count, 0)

            fixed_input = _InputCodec()
            resumed = _composed_pipeline(
                publisher,
                _SourceTTS(),
                _Translation(),
                fixed_input,
                cache=_composed_cache(root),
            )
            with SynthesisTelemetry(root, gpu_sample_interval_seconds=0) as telemetry:
                status = resumed.run(telemetry)

            self.assertIsNotNone(status.seal)
            self.assertEqual(status.catalog.sample_count, 4)
            self.assertEqual(fixed_input.calls, 2)


def _pipeline(
    publisher: SnapshotPublisher,
    source_tts: _SourceTTS,
    translation: _Translation,
    codec: _Codec,
    *,
    cache: SynthesisStageCache | None = None,
) -> StreamingSynthesisPipeline:
    placement = StagePlacement(device="cuda:0", gpu_ids=(0,))
    return StreamingSynthesisPipeline(
        _Seeds(4),
        Components(
            source_tts=source_tts,
            translation=translation,
            target_tts=_TargetTTS(),
            codec=codec,
        ),
        publisher,
        PipelineConfig(
            batch_size=2,
            source_tts=placement,
            translation=placement,
            target_tts=placement,
            codec=placement,
        ),
        cache=cache,
    )


def _cache(root: Path) -> SynthesisStageCache:
    return SynthesisStageCache(
        root,
        stream_id="pipeline-test",
        split="train",
        identity_sha256="a" * 64,
    )


def _composed_pipeline(
    publisher: SnapshotPublisher,
    source_tts: _SourceTTS,
    translation: _Translation,
    input_codec: _InputCodec,
    *,
    cache: SynthesisStageCache | None = None,
) -> StreamingSynthesisPipeline:
    placement = StagePlacement(device="cpu")
    return StreamingSynthesisPipeline(
        _Seeds(4),
        Components(
            source_tts=source_tts,
            translation=translation,
            target_tts=_TargetTTS(),
            codec=_BiCodec(),
            input_codec=input_codec,
        ),
        publisher,
        PipelineConfig(
            batch_size=2,
            source_tts=placement,
            translation=placement,
            target_tts=placement,
            codec=placement,
            input_codec=placement,
        ),
        cache=cache,
    )


def _composed_cache(root: Path) -> SynthesisStageCache:
    return SynthesisStageCache(
        root,
        stream_id="pipeline-composed-resume",
        split="train",
        identity_sha256="b" * 64,
    )


def _languages(index: int) -> tuple[Lang, Lang]:
    return (Lang.ZH, Lang.EN) if index % 2 == 0 else (Lang.EN, Lang.ZH)


def _waveform(value: int) -> AudioItem:
    return AudioItem(
        views={
            AudioView.WAVEFORM: (
                torch.full((1, 32), float(value) / 100.0),
                16_000,
            )
        }
    )


def _longcat(value: int) -> AudioItem:
    columns = [
        torch.full((4,), value % 8192, dtype=torch.long),
        *(torch.full((4,), value % 8100, dtype=torch.long) for _ in range(3)),
    ]
    return AudioItem(views={AudioView.LONGCAT: torch.stack(columns, dim=1)})


def _bicodec(value: int) -> AudioItem:
    semantic = torch.tensor(
        [[value % 1024], [(value + 1) % 1024]],
        dtype=torch.long,
    )
    global_codes = torch.tensor(
        [[value % 2048], [(value + 1) % 2048]],
        dtype=torch.long,
    )
    return AudioItem(
        views={
            AudioView.BICODEC: {
                "semantic": semantic,
                "global": global_codes,
            }
        }
    )


def _glm4(value: int) -> AudioItem:
    return AudioItem(
        views={AudioView.GLM4: torch.tensor([[value], [value + 1]], dtype=torch.long)}
    )


def _text(item: TextItem) -> str:
    return cast(str, item.views[TextView.TEXT])


def _target_text(sample: Sample) -> str:
    item = sample[Role.TARGET, Modality.TEXT]
    assert isinstance(item, TextItem)
    return _text(item)


def _restore_environment(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous
