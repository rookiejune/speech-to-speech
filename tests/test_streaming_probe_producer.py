from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch

import torch
from anydataset.types import AudioItem, AudioView, Modality, Role, TextItem, TextView

from scripts.streaming_probe_producer import ProbeConfig, run
from speech_to_speech.datamodule.streaming import WorkspaceSnapshotLoader


class StreamingProbeProducerTest(unittest.TestCase):
    def test_publishes_two_real_longcat_stores_and_exact_seal(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            environment = _environment(root, expected_samples=4, delay_seconds="0")
            with patch.dict(os.environ, {"ANYDATASET_HOME": str(root / "anydataset")}):
                status = run(environment)
                loader = WorkspaceSnapshotLoader(codec="longcat", split="train")
                first = loader(status.catalog.snapshots[0].root)[0]

            self.assertIsNotNone(status.seal)
            self.assertEqual(status.catalog.sample_count, 4)
            self.assertEqual(set(status.catalog.locations), set(range(4)))
            self.assertEqual(
                [snapshot.snapshot_id for snapshot in status.catalog.snapshots],
                ["probe-first-half", "probe-second-half"],
            )
            source_text = cast(TextItem, first[Role.SOURCE, Modality.TEXT])
            target_text = cast(TextItem, first[Role.TARGET, Modality.TEXT])
            target_audio = cast(AudioItem, first[Role.TARGET, Modality.AUDIO])
            self.assertEqual(source_text.views[TextView.TEXT], "流式探针源句 0")
            self.assertEqual(
                target_text.views[TextView.TEXT],
                "streaming probe generated translation 0",
            )
            reference = json.loads(
                (
                    status.catalog.snapshots[0].root
                    / "translation_references.jsonl"
                )
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(
                reference,
                {
                    "sample_index": 0,
                    "text": "streaming probe dataset translation 0",
                },
            )
            codes = cast(torch.Tensor, target_audio.views[AudioView.LONGCAT])
            self.assertEqual(tuple(codes.shape), (4, 4))
            self.assertTrue(bool((codes >= 0).all()))

    def test_interrupted_delay_resumes_without_repeating_delay_or_first_batch(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            environment = _environment(root, expected_samples=6, delay_seconds="30")
            interruptions: list[float] = []

            def interrupt(seconds: float) -> None:
                interruptions.append(seconds)
                raise _Interrupted

            with patch.dict(os.environ, {"ANYDATASET_HOME": str(root / "anydataset")}):
                with self.assertRaises(_Interrupted):
                    run(environment, sleep=interrupt)
                first_snapshot = root / "snapshots" / "000000-probe-first-half"
                self.assertTrue((first_snapshot / "snapshot.json").is_file())
                self.assertFalse((root / "sealed.json").exists())

                resumed_sleeps: list[float] = []
                status = run(environment, sleep=resumed_sleeps.append)

            self.assertEqual(interruptions, [30.0])
            self.assertEqual(resumed_sleeps, [])
            self.assertEqual(len(status.catalog.snapshots), 2)
            self.assertEqual(status.catalog.sample_count, 6)
            self.assertIsNotNone(status.seal)

    def test_sealed_rerun_is_idempotent(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            environment = _environment(root, expected_samples=2, delay_seconds="0")
            with patch.dict(os.environ, {"ANYDATASET_HOME": str(root / "anydataset")}):
                first = run(environment)
                second = run(environment, sleep=self.fail)

            self.assertEqual(first.catalog.sha256, second.catalog.sha256)
            self.assertEqual(len(second.catalog.snapshots), 2)

    def test_requires_even_longcat_sample_count(self) -> None:
        with TemporaryDirectory() as directory:
            environment = _environment(Path(directory), expected_samples=3)
            with self.assertRaisesRegex(ValueError, "even for bidirectional 2N"):
                ProbeConfig.from_environment(environment)
            environment["S2S_SYNTHESIS_EXPECTED_SAMPLES"] = "4"
            environment["S2S_SYNTHESIS_CODEC"] = "bicodec"
            with self.assertRaisesRegex(ValueError, "only supports codec='longcat'"):
                ProbeConfig.from_environment(environment)


class _Interrupted(Exception):
    pass


def _environment(
    root: Path,
    *,
    expected_samples: int,
    delay_seconds: str = "0",
) -> dict[str, str]:
    return {
        "S2S_SYNTHESIS_ROOT": str(root),
        "S2S_SYNTHESIS_STREAM_ID": "probe-stream",
        "S2S_SYNTHESIS_EXPECTED_SAMPLES": str(expected_samples),
        "S2S_SYNTHESIS_CODEC": "longcat",
        "S2S_SYNTHESIS_SPLIT": "train",
        "S2S_SYNTHESIS_PROBE_DELAY_SECONDS": delay_seconds,
    }
