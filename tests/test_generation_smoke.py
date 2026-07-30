from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import call, patch

import torch

from scripts import _generation_benchmark as benchmark
from scripts import _generation_probe as probe
from scripts.generation_smoke import (
    _batch_sizes,
    _non_negative_int,
    _positive_int,
    _seed,
    _validate,
    parser,
)
from speech_to_speech.generation import Request, Result
from speech_to_speech.model.acoustic import FlowModel
from speech_to_speech.task import Task


class GenerationSmokeTest(unittest.TestCase):
    def test_parser_validates_entry_budgets_before_execution(self) -> None:
        parsed = parser().parse_args(_required_args())

        self.assertEqual(parsed.sample_index, 0)
        self.assertEqual(parsed.batch_sizes, [1, 2, 4])
        self.assertEqual(parsed.max_new_tokens, 2)

        cases = (
            ("--sample-index", "-1"),
            ("--batch-sizes", "1,0"),
            ("--batch-sizes", "1,,2"),
            ("--max-new-tokens", "0"),
        )
        for option, value in cases:
            with self.subTest(option=option, value=value):
                with self.assertRaises(SystemExit):
                    parser().parse_args([*_required_args(), option, value])

    def test_integer_validators_reject_boole(self) -> None:
        for validator in (_positive_int, _non_negative_int, _batch_sizes):
            with self.subTest(validator=validator.__name__):
                with self.assertRaises(TypeError):
                    validator(True)

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


def _required_args() -> list[str]:
    return [
        "--output-dir",
        "/tmp/generation-smoke-test",
        "--audio-tokenizer",
        "/tmp/audio-tokenizer",
        "--device",
        "cpu",
    ]


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
