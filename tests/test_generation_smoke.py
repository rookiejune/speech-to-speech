from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import call, patch

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ValidationError
import torch

from scripts import _generation_benchmark as benchmark
from scripts import _generation_probe as probe
from scripts._generation_smoke_config import generation_smoke
from scripts.generation_smoke import (
    _seed,
    _validate,
)
from speech_to_speech.generation import Request, Result
from speech_to_speech.model.acoustic.flow import FlowModel
from speech_to_speech.task import Task


class GenerationSmokeTest(unittest.TestCase):
    def test_hydra_config_accepts_smoke_overrides(self) -> None:
        with initialize_config_dir(
            config_dir=str(Path(__file__).resolve().parents[1] / "configs"),
            version_base=None,
        ):
            config = compose(
                config_name="generation_smoke",
                overrides=[
                    "repo_output_root=/tmp/generation-smoke-test",
                    "runtime.audio_tokenizer=/tmp/audio-tokenizer",
                    "runtime.device=cpu",
                    "batch_sizes=[1]",
                    "datamodule.dataset.filter=null",
                    "datamodule.encode_missing_codes=true",
                ],
            )
            experiment_config = compose(
                config_name="generation_smoke",
                overrides=[
                    "repo_output_root=/tmp/generation-smoke-test",
                    "runtime.audio_tokenizer=/tmp/audio-tokenizer",
                    "runtime.device=cpu",
                    "experiment=generation/online_encode_smoke",
                ],
            )

        parsed = generation_smoke(config)
        experiment = generation_smoke(experiment_config)

        self.assertEqual(parsed.sample_index, 0)
        self.assertEqual(parsed.batch_sizes, [1])
        self.assertEqual(parsed.max_new_tokens, 2)
        self.assertIsNone(parsed.datamodule.dataset.filter)
        self.assertTrue(parsed.datamodule.encode_missing_codes)
        self.assertEqual(experiment.batch_sizes, [1])
        self.assertIsNone(experiment.datamodule.dataset.filter)
        self.assertTrue(experiment.datamodule.encode_missing_codes)

    def test_config_validates_entry_budgets_before_execution(self) -> None:
        parsed = generation_smoke(_config())

        self.assertEqual(parsed.sample_index, 0)
        self.assertEqual(parsed.batch_sizes, [1, 2, 4])
        self.assertEqual(parsed.max_new_tokens, 2)

        cases = (
            {"sample_index": -1},
            {"batch_sizes": [1, 0]},
            {"batch_sizes": []},
            {"max_new_tokens": 0},
        )
        for override in cases:
            with self.subTest(override=override):
                with self.assertRaises((TypeError, ValueError)):
                    generation_smoke(_config(override))

    def test_integer_config_rejects_booleans(self) -> None:
        cases = (
            {"sample_index": True},
            {"seed": True},
            {"max_new_tokens": True},
            {"batch_sizes": [True]},
        )
        for override in cases:
            with self.subTest(override=override):
                with self.assertRaises((TypeError, ValidationError)):
                    generation_smoke(_config(override))

    def test_cpu_generation_does_not_call_cuda_apis(self) -> None:
        model = cast(FlowModel, cast(object, _Model(torch.device("cpu"))))
        result = _result(0.0)

        with (
            patch.object(torch.cuda, "manual_seed_all") as manual_seed_all,
            patch.object(torch.cuda, "empty_cache") as empty_cache,
            patch.object(torch.cuda, "reset_peak_memory_stats") as reset_peak,
            patch.object(torch.cuda, "synchronize") as synchronize,
            patch.object(torch.cuda, "max_memory_allocated") as max_memory,
            patch("scripts._generation_probe.generate_responses", return_value=[result]),
            patch(
                "scripts._generation_benchmark.generate_responses",
                return_value=[result],
            ),
        ):
            _seed(0, torch.device("cpu"))
            probe_output = probe.run(
                model,
                _request(),
                seed=0,
                max_new_tokens=1,
                use_cache=True,
            )
            benchmark_output = benchmark.timed_generate(
                model,
                [_request()],
                seed=0,
                max_new_tokens=1,
            )

        for cuda_api in (
            manual_seed_all,
            empty_cache,
            reset_peak,
            synchronize,
            max_memory,
        ):
            cuda_api.assert_not_called()
        self.assertEqual(probe_output["peak_cuda_bytes"], 0)
        self.assertEqual(benchmark_output["peak_cuda_bytes"], 0)

    def test_cuda_generation_synchronizes_the_model_device(self) -> None:
        model = cast(FlowModel, cast(object, _Model(torch.device("cpu"))))
        device = torch.device("cuda:3")

        with (
            patch(
                "scripts._generation_benchmark._model_device",
                return_value=device,
            ),
            patch.object(torch.cuda, "manual_seed_all") as manual_seed_all,
            patch.object(torch.cuda, "empty_cache") as empty_cache,
            patch.object(torch.cuda, "reset_peak_memory_stats") as reset_peak,
            patch.object(torch.cuda, "synchronize") as synchronize,
            patch.object(torch.cuda, "max_memory_allocated", return_value=123) as peak,
            patch(
                "scripts._generation_benchmark.generate_responses",
                return_value=[_result(0.0)],
            ),
        ):
            output = benchmark.timed_generate(
                model,
                [_request()],
                seed=7,
                max_new_tokens=1,
            )

        manual_seed_all.assert_called_once_with(7)
        empty_cache.assert_called_once_with()
        reset_peak.assert_called_once_with(device)
        self.assertEqual(synchronize.call_args_list, [call(device), call(device)])
        peak.assert_called_once_with(device)
        self.assertEqual(output["peak_cuda_bytes"], 123)

    def test_batch_and_serial_finiteness_both_control_smoke_success(self) -> None:
        model = cast(FlowModel, cast(object, _Model(torch.device("cpu"))))
        cases = (
            (0.0, float("nan"), True, False),
            (float("nan"), 0.0, False, True),
        )
        for batch_value, serial_value, batch_finite, serial_finite in cases:
            with self.subTest(
                batch_value=batch_value,
                serial_value=serial_value,
            ):
                outputs = (
                    _timed_output(_result(batch_value)),
                    _timed_output(_result(serial_value)),
                )
                with patch(
                    "scripts._generation_benchmark.timed_generate",
                    side_effect=outputs,
                ):
                    report = benchmark.benchmark_batch(
                        model,
                        [_request()],
                        seed=0,
                        max_new_tokens=1,
                    )

                self.assertIs(report["batch_finite"], batch_finite)
                self.assertIs(report["serial_finite"], serial_finite)
                self.assertFalse(report["finite"])
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    _validate(_comparison(), [report])


class _Model:
    def __init__(self, device: torch.device) -> None:
        weight = torch.zeros(1, device=device)
        self.backbone = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=weight)
        )
        self.runtime = SimpleNamespace(audio_generation_allowed_ids=(1,))

    def generation_step(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


def _config(override: dict[str, object] | None = None) -> DictConfig:
    base = OmegaConf.create(
        {
            "task": "s2st",
            "run_name": "generation_smoke",
            "repo_output_root": "/tmp/generation-smoke-test",
            "output_subdir": "004-real-cached-generation",
            "output_dir": "/tmp/generation-smoke-test/004-real-cached-generation",
            "audio_sequence_layout": "semantic",
            "datamodule": {
                "codec": "longcat",
                "dataloader": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "pin_memory": False,
                    "persistent_workers": False,
                },
                "shape": "pair",
                "encode_missing_codes": False,
                "dataset": {
                    "name": "wmt19_tts",
                    "root": None,
                    "split": "train",
                    "filter": "speech_translation_v1",
                    "split_manifest": None,
                    "split_label": "train",
                    "toy_samples": 8,
                    "toy_frames": 4,
                },
            },
            "sample_index": 0,
            "batch_sizes": [1, 2, 4],
            "max_new_tokens": 2,
            "seed": 0,
        }
    )
    if override is None:
        return cast(DictConfig, base)
    return cast(DictConfig, OmegaConf.merge(base, override))


def _request() -> Request:
    return Request(
        prompt_ids=torch.tensor([1]),
        task=Task.S2ST,
        audio_input_positions=None,
        audio_context=None,
    )


def _result(waveform: float) -> Result:
    return Result(
        response_ids=torch.tensor([1]),
        audio={
            "features": torch.zeros(1, 1),
            "codes": None,
            "waveform": torch.tensor([waveform]),
            "sample_rate": 24_000,
        },
    )


def _timed_output(result: Result) -> dict[str, Any]:
    return {
        "results": [result],
        "elapsed_seconds": 1.0,
        "peak_cuda_bytes": 0,
    }


def _comparison() -> dict[str, bool]:
    return {
        "tokens_equal": True,
        "cached_finite": True,
        "full_finite": True,
    }


if __name__ == "__main__":
    unittest.main()
