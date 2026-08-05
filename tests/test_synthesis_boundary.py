from __future__ import annotations

import json
import os
import sys
import threading
import time
import unittest
from collections.abc import Sequence, Sized
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast, overload

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
from torch.utils.data import Dataset

from speech_to_speech.datamodule.streaming import (
    SynthesisRequest,
    WorkspaceSnapshotLoader,
)
from speech_to_speech.synthesis.publisher import (
    SnapshotPublisher,
    TranslationReference,
)
from speech_to_speech.synthesis.process import controller
from speech_to_speech.synthesis.subprocess import SubprocessController


class _Samples(Dataset[Sample]):
    def __init__(self, count: int) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Sample:
        return cast(Sample, cast(object, {"index": index}))


class _FailingSamples(Sequence[Sample]):
    def __len__(self) -> int:
        return 1

    @overload
    def __getitem__(self, index: int) -> Sample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Sample]: ...

    def __getitem__(self, index: int | slice) -> Sample | Sequence[Sample]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if index != 0:
            raise IndexError(index)
        raise RuntimeError("output codec failed")


def _loader(root: Path) -> Dataset[Sample]:
    payload = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    indices = payload["sample_indices"]
    if not isinstance(indices, list):
        raise TypeError("test snapshot indices must be a list.")
    return _Samples(len(indices))


def _store_sample(index: int, view: AudioView) -> Sample:
    if view is AudioView.WAVEFORM:
        audio: object = (torch.full((1, 8), float(index)), 16_000)
    elif view is AudioView.BICODEC:
        audio = {
            "semantic": torch.tensor([[index, index + 1]], dtype=torch.long),
            "global": torch.tensor([[index + 2]], dtype=torch.long),
        }
    else:
        audio = torch.tensor([[index, index + 1]], dtype=torch.long)
    return cast(
        Sample,
        {
            (Role.SOURCE, Modality.TEXT): TextItem(
                views={TextView.TEXT: f"source {index}"},
                meta={TextMeta.LANG: Lang.ZH},
            ),
            (Role.TARGET, Modality.TEXT): TextItem(
                views={TextView.TEXT: f"target {index}"},
                meta={TextMeta.LANG: Lang.EN},
            ),
            (Role.SOURCE, Modality.AUDIO): AudioItem(views={view: audio}),
            (Role.TARGET, Modality.AUDIO): AudioItem(views={view: audio}),
        },
    )


def _references(*indices: int) -> list[TranslationReference]:
    return [
        TranslationReference(index, f"dataset translation {index}")
        for index in indices
    ]


def _restore_environment(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
        return
    os.environ[name] = previous


class SnapshotPublisherTest(unittest.TestCase):
    def test_real_store_roundtrip_is_idempotent_and_seals_exact_coverage(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_home = os.environ.get("ANYDATASET_HOME")
            self.addCleanup(_restore_environment, "ANYDATASET_HOME", previous_home)
            os.environ["ANYDATASET_HOME"] = str(root / "anydataset")
            publisher = SnapshotPublisher(
                root,
                stream_id="stream-a",
                expected_samples=2,
                codec="longcat",
                split="train",
                loader=WorkspaceSnapshotLoader(codec="longcat", split="train"),
            )
            base = [_store_sample(index, AudioView.WAVEFORM) for index in range(2)]
            codec = [_store_sample(index, AudioView.LONGCAT) for index in range(2)]
            first = publisher.publish(
                snapshot_id="first",
                sample_indices=[0],
                translation_references=_references(0),
                base_samples=base[:1],
                codec_samples=codec[:1],
            )
            repeated = publisher.publish(
                snapshot_id="first",
                sample_indices=[0],
                translation_references=_references(0),
                base_samples=base[:1],
                codec_samples=codec[:1],
            )
            second = publisher.publish(
                snapshot_id="second",
                sample_indices=[1],
                translation_references=_references(1),
                base_samples=base[1:],
                codec_samples=codec[1:],
            )

            self.assertEqual(first, repeated)
            self.assertTrue((first / "base").is_dir())
            self.assertTrue((first / "longcat").is_dir())
            self.assertTrue((first / "translation_references.jsonl").is_file())
            self.assertTrue((second / "snapshot.json").is_file())
            self.assertTrue((root / "sealed.json").is_file())
            manifest = json.loads(
                (first / "snapshot.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["schema"],
                "speech-to-speech-stream-snapshot-v2",
            )
            self.assertEqual(
                manifest["translation_references"]["file"],
                "translation_references.jsonl",
            )
            snapshot = publisher.feed.status().catalog.snapshots[0]
            loaded = publisher.feed.load(snapshot)
            self.assertEqual(len(cast(Sized, cast(object, loaded))), 1)
            sample = cast(dict[object, object], cast(object, loaded[0]))
            text = cast(TextItem, sample[Role.SOURCE, Modality.TEXT])
            audio = cast(AudioItem, sample[Role.TARGET, Modality.AUDIO])
            self.assertEqual(
                text.views[TextView.TEXT],
                "source 0",
            )
            self.assertTrue(
                torch.equal(
                    cast(torch.Tensor, audio.views[AudioView.LONGCAT]),
                    torch.tensor([[0, 1]], dtype=torch.long),
                )
            )
            comparison = publisher.feed.published([0])[0]
            self.assertEqual(comparison.reference_translation, "dataset translation 0")
            with self.assertRaisesRegex(RuntimeError, "after sealing"):
                publisher.publish(
                    snapshot_id="third",
                    sample_indices=[0],
                    translation_references=_references(0),
                    base_samples=base[:1],
                    codec_samples=codec[:1],
                )

    def test_idempotent_publish_rejects_changed_translation_reference(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_home = os.environ.get("ANYDATASET_HOME")
            self.addCleanup(_restore_environment, "ANYDATASET_HOME", previous_home)
            os.environ["ANYDATASET_HOME"] = str(root / "anydataset")
            publisher = SnapshotPublisher(
                root,
                stream_id="stream-a",
                expected_samples=2,
                codec="longcat",
                split="train",
                loader=WorkspaceSnapshotLoader(codec="longcat", split="train"),
            )
            base = [_store_sample(0, AudioView.WAVEFORM)]
            codec = [_store_sample(0, AudioView.LONGCAT)]
            publisher.publish(
                snapshot_id="first",
                sample_indices=[0],
                translation_references=_references(0),
                base_samples=base,
                codec_samples=codec,
            )

            with self.assertRaisesRegex(ValueError, "other translation references"):
                publisher.publish(
                    snapshot_id="first",
                    sample_indices=[0],
                    translation_references=[TranslationReference(0, "changed")],
                    base_samples=base,
                    codec_samples=codec,
                )

    def test_reference_sidecar_is_lazy_and_digest_checked(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_home = os.environ.get("ANYDATASET_HOME")
            self.addCleanup(_restore_environment, "ANYDATASET_HOME", previous_home)
            os.environ["ANYDATASET_HOME"] = str(root / "anydataset")
            publisher = SnapshotPublisher(
                root,
                stream_id="stream-a",
                expected_samples=1,
                codec="longcat",
                split="train",
                loader=WorkspaceSnapshotLoader(codec="longcat", split="train"),
            )
            published = publisher.publish(
                snapshot_id="only",
                sample_indices=[0],
                translation_references=_references(0),
                base_samples=[_store_sample(0, AudioView.WAVEFORM)],
                codec_samples=[_store_sample(0, AudioView.LONGCAT)],
            )
            catalog = publisher.feed.status().catalog
            publisher.feed.sample_at(0, catalog)
            (published / "translation_references.jsonl").write_text(
                '{"sample_index":0,"text":"tampered"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "sidecar digest mismatch"):
                publisher.feed.published([0])

    def test_translation_reference_indices_must_match_snapshot_indices(self) -> None:
        with TemporaryDirectory() as directory:
            publisher = SnapshotPublisher(
                Path(directory),
                stream_id="stream-a",
                expected_samples=2,
                codec="longcat",
                split="train",
                loader=_loader,
            )
            with self.assertRaisesRegex(ValueError, "exactly match sample_indices"):
                publisher.publish(
                    snapshot_id="first",
                    sample_indices=[0],
                    translation_references=_references(1),
                    base_samples=[_store_sample(0, AudioView.WAVEFORM)],
                    codec_samples=[_store_sample(0, AudioView.LONGCAT)],
                )

    def test_decoupled_snapshot_joins_glm4_source_and_bicodec_target(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_home = os.environ.get("ANYDATASET_HOME")
            self.addCleanup(_restore_environment, "ANYDATASET_HOME", previous_home)
            os.environ["ANYDATASET_HOME"] = str(root / "anydataset")
            publisher = SnapshotPublisher(
                root,
                stream_id="stream-dual",
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
            published = publisher.publish(
                snapshot_id="dual",
                sample_indices=[0],
                translation_references=_references(0),
                base_samples=[_store_sample(0, AudioView.WAVEFORM)],
                input_codec_samples=[_store_sample(0, AudioView.GLM4)],
                codec_samples=[_store_sample(0, AudioView.BICODEC)],
            )

            self.assertTrue((published / "base").is_dir())
            self.assertTrue((published / "glm4").is_dir())
            self.assertTrue((published / "bicodec").is_dir())
            manifest = json.loads(
                (published / "snapshot.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["input_codec"], "glm4")
            self.assertEqual(manifest["codec"], "bicodec")
            snapshot = publisher.feed.status().catalog.snapshots[0]
            sample = cast(dict[object, object], cast(object, publisher.feed.load(snapshot)[0]))
            source = cast(AudioItem, sample[Role.SOURCE, Modality.AUDIO])
            target = cast(AudioItem, sample[Role.TARGET, Modality.AUDIO])
            self.assertEqual(set(source.views), {AudioView.GLM4})
            self.assertEqual(set(target.views), {AudioView.BICODEC})
            self.assertTrue(
                torch.equal(
                    cast(torch.Tensor, source.views[AudioView.GLM4]),
                    torch.tensor([[0, 1]], dtype=torch.long),
                )
            )
            target_codes = cast(dict[str, torch.Tensor], target.views[AudioView.BICODEC])
            self.assertTrue(
                torch.equal(
                    target_codes["semantic"],
                    torch.tensor([[0, 1]], dtype=torch.long),
                )
            )
            seal = json.loads((root / "sealed.json").read_text(encoding="utf-8"))
            self.assertEqual(seal["input_codec"], "glm4")
            self.assertEqual(seal["codec"], "bicodec")

    def test_decoupled_snapshot_rejects_misaligned_stores_before_visibility(
        self,
    ) -> None:
        for case, message in (
            ("text", "disagree on aligned text"),
            ("direction", "missing target audio"),
        ):
            with self.subTest(case=case), TemporaryDirectory() as directory:
                root = Path(directory)
                previous_home = os.environ.get("ANYDATASET_HOME")
                self.addCleanup(_restore_environment, "ANYDATASET_HOME", previous_home)
                os.environ["ANYDATASET_HOME"] = str(root / "anydataset")
                publisher = SnapshotPublisher(
                    root,
                    stream_id="stream-dual",
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
                output = _store_sample(0, AudioView.BICODEC)
                output_mapping = cast(
                    dict[object, object],
                    cast(object, output),
                )
                if case == "text":
                    output_mapping[Role.SOURCE, Modality.TEXT] = TextItem(
                        views={TextView.TEXT: "other source"},
                        meta={TextMeta.LANG: Lang.ZH},
                    )
                else:
                    del output_mapping[Role.TARGET, Modality.AUDIO]

                with self.assertRaisesRegex((ValueError, KeyError), message):
                    publisher.publish(
                        snapshot_id="dual",
                        sample_indices=[0],
                        translation_references=_references(0),
                        base_samples=[_store_sample(0, AudioView.WAVEFORM)],
                        input_codec_samples=[_store_sample(0, AudioView.GLM4)],
                        codec_samples=[output],
                    )

                self.assertEqual(
                    list((root / "snapshots").glob("*/snapshot.json")),
                    [],
                )
                self.assertFalse((root / "sealed.json").exists())
                self.assertEqual(publisher.feed.status().catalog.sample_count, 0)

    def test_decoupled_snapshot_is_not_visible_until_both_codecs_finish(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            previous_home = os.environ.get("ANYDATASET_HOME")
            self.addCleanup(_restore_environment, "ANYDATASET_HOME", previous_home)
            os.environ["ANYDATASET_HOME"] = str(root / "anydataset")
            publisher = SnapshotPublisher(
                root,
                stream_id="stream-dual",
                expected_samples=1,
                codec="bicodec",
                input_codec="glm4",
                split="train",
                loader=_loader,
            )

            with self.assertRaisesRegex(RuntimeError, "output codec failed"):
                publisher.publish(
                    snapshot_id="dual",
                    sample_indices=[0],
                    translation_references=_references(0),
                    base_samples=[_store_sample(0, AudioView.WAVEFORM)],
                    input_codec_samples=[_store_sample(0, AudioView.GLM4)],
                    codec_samples=_FailingSamples(),
                )

            self.assertEqual(list((root / "snapshots").glob("*/snapshot.json")), [])
            self.assertEqual(publisher.feed.status().catalog.sample_count, 0)

    def test_rejects_path_segments_that_escape_the_snapshot_root(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "safe path segment"):
                SnapshotPublisher(
                    Path(directory),
                    stream_id="stream-a",
                    expected_samples=2,
                    codec="../longcat",
                    split="train",
                    loader=_loader,
                )


class SubprocessControllerTest(unittest.TestCase):
    def test_coupled_v1_seal_without_input_codec_is_accepted(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sealed.json").write_text(
                json.dumps(
                    {
                        "schema": "speech-to-speech-stream-seal-v1",
                        "stream_id": "stream-a",
                        "expected_samples": 2,
                        "codec": "longcat",
                    }
                ),
                encoding="utf-8",
            )
            instance = controller(
                SynthesisRequest(
                    root=root,
                    stream_id="stream-a",
                    expected_samples=2,
                    codec="longcat",
                    split="train",
                    options={"command": [sys.executable, "-c", "pass"]},
                )
            )

            instance.start()
            instance.close()

    def test_existing_seal_must_match_both_codecs_before_noop(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sealed.json").write_text(
                json.dumps(
                    {
                        "schema": "speech-to-speech-stream-seal-v1",
                        "stream_id": "stream-a",
                        "expected_samples": 2,
                        "input_codec": "longcat",
                        "codec": "bicodec",
                    }
                ),
                encoding="utf-8",
            )
            instance = controller(
                SynthesisRequest(
                    root=root,
                    stream_id="stream-a",
                    expected_samples=2,
                    codec="bicodec",
                    split="train",
                    options={"command": [sys.executable, "-c", "pass"]},
                    input_codec="glm4",
                )
            )

            with self.assertRaisesRegex(ValueError, "seal input_codec"):
                instance.start()

    def test_coupled_v1_producer_metadata_without_input_codec_is_reused(
        self,
    ) -> None:
        command = [sys.executable, "-c", "pass"]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "producer.json").write_text(
                json.dumps(
                    {
                        "schema": "speech-to-speech-stream-producer-v1",
                        "stream_id": "stream-a",
                        "expected_samples": 2,
                        "codec": "longcat",
                        "split": "train",
                        "pid": os.getpid(),
                        "command": command,
                    }
                ),
                encoding="utf-8",
            )
            instance = controller(
                SynthesisRequest(
                    root=root,
                    stream_id="stream-a",
                    expected_samples=2,
                    codec="longcat",
                    split="train",
                    options={"command": command, "monitor_seconds": 0.01},
                )
            )

            instance.start()
            self.assertEqual(instance._pid, os.getpid())
            instance.close()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "producer.json").write_text(
                json.dumps(
                    {
                        "schema": "speech-to-speech-stream-producer-v1",
                        "stream_id": "stream-a",
                        "expected_samples": 2,
                        "codec": "bicodec",
                        "split": "train",
                        "pid": os.getpid(),
                        "command": command,
                    }
                ),
                encoding="utf-8",
            )
            instance = controller(
                SynthesisRequest(
                    root=root,
                    stream_id="stream-a",
                    expected_samples=2,
                    codec="bicodec",
                    input_codec="glm4",
                    split="train",
                    options={"command": command},
                )
            )

            with self.assertRaisesRegex(ValueError, "metadata input_codec"):
                instance.start()

    def test_concurrent_starts_launch_one_child_and_share_its_pid(self) -> None:
        with TemporaryDirectory() as directory:
            request = SynthesisRequest(
                root=Path(directory),
                stream_id="stream-a",
                expected_samples=2,
                codec="longcat",
                split="train",
                options={"command": [sys.executable, "-c", "import time; time.sleep(30)"]},
            )
            controllers = (controller(request), controller(request))
            ready = threading.Barrier(3)
            errors: list[BaseException] = []

            def start(instance: SubprocessController) -> None:
                try:
                    ready.wait()
                    instance.start()
                except BaseException as error:
                    errors.append(error)

            threads = [threading.Thread(target=start, args=(instance,)) for instance in controllers]
            for thread in threads:
                thread.start()
            ready.wait()
            for thread in threads:
                thread.join()
            try:
                self.assertEqual(errors, [])
                pids = {item._pid for item in controllers}
                self.assertEqual(len(pids), 1)
                self.assertNotIn(None, pids)
            finally:
                for instance in controllers:
                    instance.close()

    def test_reuses_live_pid_metadata_without_duplicate_child(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            request = SynthesisRequest(
                root=root,
                stream_id="stream-a",
                expected_samples=2,
                codec="longcat",
                split="train",
                options={"command": [sys.executable, "-c", "import time; time.sleep(30)"]},
            )
            owner = controller(request)
            reused = controller(request)
            owner.start()
            try:
                first = json.loads((root / "producer.json").read_text(encoding="utf-8"))
                reused.start()
                second = json.loads((root / "producer.json").read_text(encoding="utf-8"))

                self.assertEqual(first["pid"], second["pid"])
                reused.close()
                owner.check()
            finally:
                owner.close()

    def test_monitor_records_failure_and_passes_stream_environment(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            command = [
                sys.executable,
                "-c",
                "import os, pathlib; pathlib.Path(os.environ['S2S_SYNTHESIS_ROOT']).joinpath('env.txt').write_text('|'.join((os.environ['S2S_SYNTHESIS_STREAM_ID'], os.environ['S2S_SYNTHESIS_INPUT_CODEC'], os.environ['S2S_SYNTHESIS_OUTPUT_CODEC'], os.environ['S2S_SYNTHESIS_CODEC']))); raise SystemExit(7)",
            ]
            instance = controller(
                SynthesisRequest(
                    root=root,
                    stream_id="stream-a",
                    expected_samples=2,
                    codec="bicodec",
                    split="train",
                    options={
                        "command": json.dumps(command),
                        "environment": {"S2S_TEST_PRODUCER_DEVICE": "cuda:3"},
                        "monitor_seconds": 0.01,
                    },
                    input_codec="glm4",
                )
            )
            instance.start()
            failed = root / "failed.json"
            deadline = time.monotonic() + 3.0
            while not failed.exists() and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertTrue(failed.is_file())
            payload = json.loads(failed.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], "speech-to-speech-stream-failure-v1")
            self.assertEqual(payload["stream_id"], "stream-a")
            self.assertEqual(payload["expected_samples"], 2)
            self.assertEqual(payload["input_codec"], "glm4")
            self.assertEqual(payload["codec"], "bicodec")
            self.assertEqual(payload["exit_code"], 7)
            self.assertEqual(
                (root / "env.txt").read_text(encoding="utf-8"),
                "stream-a|glm4|bicodec|bicodec",
            )
            metadata = json.loads((root / "producer.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["environment"],
                {"S2S_TEST_PRODUCER_DEVICE": "cuda:3"},
            )
            with self.assertRaisesRegex(RuntimeError, "exited before"):
                instance.check()
            instance.close()

    def test_rejects_reserved_stream_environment_override(self) -> None:
        with TemporaryDirectory() as directory:
            instance = controller(
                SynthesisRequest(
                    root=Path(directory),
                    stream_id="stream-a",
                    expected_samples=2,
                    codec="longcat",
                    split="train",
                    options={
                        "command": [sys.executable, "-c", "pass"],
                        "environment": {"S2S_SYNTHESIS_STREAM_ID": "other"},
                    },
                )
            )

            with self.assertRaisesRegex(ValueError, "cannot override"):
                instance.start()
