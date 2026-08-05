from __future__ import annotations

import csv
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest.mock import Mock, patch

from speech_to_speech.callback.gpu import GpuTelemetrySampler
from speech_to_speech.callback.streaming import StreamingTelemetryCallback
from speech_to_speech.datamodule.streaming import StreamingTelemetry
from speech_to_speech.synthesis.telemetry import stage


class GpuTelemetrySamplerTest(unittest.TestCase):
    def test_writes_visible_gpu_samples_and_returns_means(self) -> None:
        output = "\n".join(
            (
                "2026/08/05 12:00:00.000, 5, GPU A, 20, 10, 100, 1000, 50, 100",
                "2026/08/05 12:00:00.000, 6, GPU B, 80, 30, 300, 1000, 70, 100",
                "2026/08/05 12:00:00.000, 7, OTHER, 100, 90, 900, 1000, 90, 100",
            )
        )
        completed = SimpleNamespace(returncode=0, stdout=output, stderr="")
        with TemporaryDirectory() as directory, patch.dict(
            "os.environ",
            {"CUDA_VISIBLE_DEVICES": "5,6"},
        ), patch(
            "speech_to_speech.callback.gpu.shutil.which",
            return_value="/usr/bin/nvidia-smi",
        ), patch(
            "speech_to_speech.callback.gpu.subprocess.run",
            return_value=completed,
        ):
            path = Path(directory) / "gpu.csv"
            sampler = GpuTelemetrySampler(path, interval_seconds=60.0)
            sampler.start()
            latest = sampler.latest()
            sampler.stop()
            summary = sampler.summary()
            resumed = GpuTelemetrySampler(path, interval_seconds=60.0)
            resumed.start()
            resumed.stop()
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual([row["index"] for row in rows], ["5", "6", "5", "6"])
        self.assertEqual(latest["utilization_gpu_percent"], 50.0)
        self.assertEqual(latest["memory_used_mb"], 200.0)
        self.assertEqual(summary["scope"], "current_process")
        self.assertTrue(summary["available"])
        self.assertEqual(summary["samples"], 1)
        self.assertEqual(summary["utilization_gpu_percent"], 50.0)


class StreamingTelemetryCallbackTest(unittest.TestCase):
    def test_logs_batch_wait_step_cursor_and_latest_gpu_metrics(self) -> None:
        telemetry = StreamingTelemetry(
            batch_fetch_seconds=3.0,
            batch_wait_seconds=2.0,
            batch_load_seconds=1.0,
            total_fetch_seconds=7.0,
            total_wait_seconds=5.0,
            total_load_seconds=2.0,
            wait_events=2,
            poll_count=5,
            read_position=8,
            committed_position=6,
            committed_batches=3,
            published_samples=10,
            expected_samples=20,
        )
        datamodule = SimpleNamespace(
            streaming_enabled=True,
            streaming_telemetry=Mock(return_value=telemetry),
        )
        trainer = SimpleNamespace(
            datamodule=datamodule,
            global_step=2,
            is_global_zero=True,
        )
        module = Mock()
        sampler = Mock()
        sampler.latest.return_value = {
            "utilization_gpu_percent": 75.0,
            "memory_used_mb": 1234.0,
        }
        callback = StreamingTelemetryCallback(
            loader_name="s2st",
            gpu_sample_interval_seconds=1.0,
            log_every_n_steps=1,
        )
        cast(Any, callback)._sampler = sampler
        writer = Mock()

        with patch(
            "speech_to_speech.callback.streaming.time.perf_counter",
            side_effect=[10.0, 12.0],
        ), patch(
            "speech_to_speech.callback.streaming.experiment.scalar",
            return_value=writer,
        ):
            callback.on_train_batch_start(
                cast(Any, trainer),
                cast(Any, module),
                None,
                0,
            )
            callback.on_train_batch_end(
                cast(Any, trainer),
                cast(Any, module),
                None,
                None,
                0,
            )

        datamodule.streaming_telemetry.assert_called_once_with(loader_name="s2st")
        values = {call.args[0]: call.args[1] for call in writer.add_scalar.call_args_list}
        self.assertEqual(values["streaming/batch_wait_seconds"], 2.0)
        self.assertEqual(values["streaming/step_seconds"], 2.0)
        self.assertEqual(values["streaming/wait_seconds_total"], 5.0)
        self.assertEqual(values["streaming/wait_ratio"], 0.4)
        self.assertEqual(values["streaming/gpu_utilization_percent"], 75.0)
        self.assertEqual(values["streaming/gpu_memory_used_mb"], 1234.0)


class SynthesisStageTelemetryTest(unittest.TestCase):
    def test_stage_emits_structured_start_and_elapsed_finish_events(self) -> None:
        events: list[dict[str, object]] = []

        with patch(
            "speech_to_speech.synthesis.telemetry.time.perf_counter",
            side_effect=[10.0, 13.5],
        ):
            with stage(
                "text_translation",
                sample_count=4,
                device="cuda:1",
                gpu_ids=[1],
                sink=lambda payload: events.append(dict(payload)),
            ):
                pass

        self.assertEqual([event["event"] for event in events], ["stage_started", "stage_finished"])
        self.assertEqual(events[0]["stage"], "text_translation")
        self.assertEqual(events[0]["sample_count"], 4)
        self.assertEqual(events[0]["gpu_ids"], [1])
        self.assertEqual(events[1]["elapsed_seconds"], 3.5)


if __name__ == "__main__":
    unittest.main()
