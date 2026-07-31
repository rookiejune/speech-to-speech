from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from anytrain.codec import SemanticAcousticCodes

from speech_to_speech.callback import OOMDiagnostics
from speech_to_speech.callback._oom import generation_report, report_oom
from speech_to_speech.callback.logging.acoustic import AcousticEvaluation
from speech_to_speech.datamodule.types import (
    Language,
    ModelBatch,
    RawSpeech,
    RawSpeechBatch,
    SpeechTaskSample,
    Text,
)
from speech_to_speech.generation import Request
from speech_to_speech.task import Task


class OOMDiagnosticsTest(unittest.TestCase):
    def test_model_batch_report_survives_until_backward_oom(self) -> None:
        callback = OOMDiagnostics()
        trainer = _trainer(callback)
        module = SimpleNamespace(device=torch.device("cpu"))
        batch = _model_batch()

        callback.on_train_batch_start(trainer, module, batch, 7)
        json.dumps(callback._context)
        callback.on_before_backward(trainer, module, torch.tensor(1.0))

        payload = _exception_payload(
            callback,
            trainer,
            module,
            torch.OutOfMemoryError("CUDA out of memory"),
        )

        self.assertEqual(payload["event"], "out_of_memory")
        self.assertEqual(payload["global_rank"], 1)
        self.assertIsNone(payload["cuda"])
        context = payload["context"]
        self.assertEqual(context["phase"], "train_backward")
        self.assertEqual(context["batch_idx"], 7)
        inputs = context["inputs"]
        self.assertEqual(inputs["tasks"], [Task.S2ST.value])
        self.assertEqual(inputs["input_ids"]["shape"], [1, 5])
        self.assertEqual(
            inputs["acoustic_target"]["codes"]["shape"],
            [1, 2, 2],
        )
        self.assertEqual(
            inputs["audio_contexts"][0]["acoustic"]["shape"],
            [3, 2],
        )

    def test_raw_batch_report_contains_role_specific_waveform_shape(self) -> None:
        callback = OOMDiagnostics()
        trainer = _trainer(callback)
        module = SimpleNamespace(device=torch.device("cpu"))
        batch = RawSpeechBatch(
            samples=(
                SpeechTaskSample(
                    source=Text(torch.tensor([1, 2]), Language.EN),
                    target=RawSpeech(
                        text_token_ids=torch.tensor([3]),
                        waveform=torch.zeros(2, 160),
                        sample_rate=16_000,
                        language=Language.EN,
                    ),
                    task=Task.TTS,
                    prediction=Task.TTS.prediction_modality,
                ),
            ),
            pad_token_id=0,
        )

        callback.on_train_batch_start(trainer, module, batch, 2)
        payload = _exception_payload(
            callback,
            trainer,
            module,
            torch.OutOfMemoryError("codec allocation failed"),
        )

        sample = payload["context"]["inputs"]["samples"][0]
        self.assertEqual(sample["target"]["waveform"]["shape"], [2, 160])
        self.assertEqual(sample["target"]["sample_rate"], 16_000)

    def test_generation_context_replaces_outer_train_batch(self) -> None:
        callback = OOMDiagnostics()
        trainer = _trainer(callback)
        module = SimpleNamespace(device=torch.device("cpu"))
        callback.on_train_batch_start(trainer, module, _model_batch(), 4)
        requests = [
            Request(
                prompt_ids=torch.tensor([10, 101, 102, 198]),
                task=Task.S2ST,
                audio_input_positions=torch.tensor([1, 2]),
                audio_context=None,
            ),
            Request(
                prompt_ids=torch.tensor([20, 103, 198]),
                task=Task.S2ST,
                audio_input_positions=torch.tensor([1]),
                audio_context=None,
            ),
        ]
        error = torch.OutOfMemoryError("generation allocation failed")

        captured = report_oom(
            trainer,
            module,
            error,
            phase="task_sample_generation",
            inputs=generation_report(
                requests,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True,
            ),
        )
        payload = _exception_payload(callback, trainer, module, error)

        self.assertTrue(captured)
        context = payload["context"]
        self.assertEqual(context["phase"], "task_sample_generation")
        self.assertEqual(context["inputs"]["padded_prompt_shape"], [2, 4])
        self.assertEqual(context["inputs"]["max_new_tokens"], 512)
        self.assertEqual(
            context["inputs"]["padded_audio_input_positions_shape"],
            [2, 2],
        )

    def test_text_generation_report_has_no_audio_position_padding(self) -> None:
        requests = [
            Request(
                prompt_ids=torch.tensor([1, 2]),
                task=Task.T2TT,
                audio_input_positions=None,
                audio_context=None,
            ),
            Request(
                prompt_ids=torch.tensor([3]),
                task=Task.T2TT,
                audio_input_positions=None,
                audio_context=None,
            ),
        ]

        report = generation_report(
            requests,
            max_new_tokens=32,
            do_sample=False,
            use_cache=True,
        )

        self.assertIsNone(report["padded_audio_input_positions_shape"])

    def test_post_backward_and_optimizer_phases_are_distinct(self) -> None:
        callback = OOMDiagnostics()
        trainer = _trainer(callback)
        module = SimpleNamespace(device=torch.device("cpu"))
        batch = _model_batch()
        callback.on_train_batch_start(trainer, module, batch, 1)
        callback.on_before_backward(trainer, module, torch.tensor(1.0))
        callback.on_after_backward(trainer, module)

        post_backward = _exception_payload(
            callback,
            trainer,
            module,
            torch.OutOfMemoryError("gradient clipping failed"),
        )
        self.assertEqual(post_backward["context"]["phase"], "train_post_backward")

        parameter = torch.nn.Parameter(torch.zeros(()))
        callback.on_before_optimizer_step(
            trainer,
            module,
            torch.optim.SGD([parameter], lr=0.1),
        )
        optimizer = _exception_payload(
            callback,
            trainer,
            module,
            torch.OutOfMemoryError("optimizer allocation failed"),
        )
        self.assertEqual(optimizer["context"]["phase"], "train_optimizer")

    def test_validation_report_contains_dataloader_index(self) -> None:
        callback = OOMDiagnostics()
        trainer = _trainer(callback)
        module = SimpleNamespace(device=torch.device("cpu"))

        callback.on_validation_batch_start(
            trainer,
            module,
            _model_batch(),
            batch_idx=6,
            dataloader_idx=2,
        )
        payload = _exception_payload(
            callback,
            trainer,
            module,
            torch.OutOfMemoryError("validation allocation failed"),
        )

        self.assertEqual(payload["context"]["phase"], "validation_step")
        self.assertEqual(payload["context"]["batch_idx"], 6)
        self.assertEqual(payload["context"]["dataloader_idx"], 2)

    def test_acoustic_evaluation_reports_its_fixed_batch(self) -> None:
        oom = OOMDiagnostics()
        trainer = _trainer(oom)
        module = SimpleNamespace(device=torch.device("cpu"))
        evaluation = AcousticEvaluation(
            model=SimpleNamespace(),
            batch=_model_batch(),
            codec=SimpleNamespace(),
            output_dir=Path("/tmp"),
            every_n_steps=1,
            seeds=(),
        )
        error = torch.OutOfMemoryError("evaluation allocation failed")

        with patch(
            "speech_to_speech.callback.logging.acoustic.evaluate",
            side_effect=error,
        ):
            with self.assertRaises(torch.OutOfMemoryError) as raised:
                evaluation.on_fit_start(trainer, module)

        self.assertIs(raised.exception, error)
        payload = _exception_payload(oom, trainer, module, error)
        self.assertEqual(payload["context"]["phase"], "acoustic_evaluation")
        self.assertEqual(
            payload["context"]["inputs"]["input_ids"]["shape"],
            [1, 5],
        )

    def test_cuda_stats_failure_is_reported_without_masking_oom(self) -> None:
        callback = OOMDiagnostics()
        trainer = _trainer(callback)
        module = SimpleNamespace(device=torch.device("cuda:0"))
        error = torch.OutOfMemoryError("CUDA out of memory")

        with patch(
            "speech_to_speech.callback._oom.torch.cuda.memory_allocated",
            side_effect=RuntimeError("stats unavailable"),
        ):
            payload = _exception_payload(callback, trainer, module, error)

        self.assertEqual(payload["error"]["message"], str(error))
        self.assertEqual(payload["cuda"]["device"], "cuda:0")
        self.assertIn("stats unavailable", payload["cuda"]["error"])

    def test_non_oom_is_not_reported(self) -> None:
        callback = OOMDiagnostics()
        trainer = _trainer(callback)
        module = SimpleNamespace(device=torch.device("cpu"))
        output = io.StringIO()

        with redirect_stderr(output):
            callback.on_exception(trainer, module, RuntimeError("boom"))

        self.assertEqual(output.getvalue(), "")

    def test_completed_batch_is_not_reported_as_active(self) -> None:
        callback = OOMDiagnostics()
        trainer = _trainer(callback)
        module = SimpleNamespace(device=torch.device("cpu"))
        batch = _model_batch()
        callback.on_train_batch_start(trainer, module, batch, 3)
        callback.on_train_batch_end(trainer, module, None, batch, 3)

        payload = _exception_payload(
            callback,
            trainer,
            module,
            torch.OutOfMemoryError("outside batch"),
        )

        self.assertEqual(payload["context"]["phase"], "outside_batch")
        self.assertIsNone(payload["context"]["inputs"])


def _trainer(callback: OOMDiagnostics):
    return SimpleNamespace(
        callbacks=[callback],
        current_epoch=2,
        global_step=11,
        global_rank=1,
        local_rank=0,
        is_global_zero=True,
    )


def _model_batch() -> ModelBatch:
    return ModelBatch(
        input_ids=torch.tensor([[1, 8, 9, 11, 2]]),
        token_labels=torch.tensor([[-100, -100, -100, 11, 2]]),
        acoustic_target={
            "semantic_codes": torch.tensor([[[1], [2]]]),
            "codes": torch.tensor([[[1, 2], [3, 4]]]),
            "token_positions": torch.tensor([[3, 4]]),
        },
        tasks=[Task.S2ST],
        predictions=[Task.S2ST.prediction_modality],
        pad_token_id=0,
        audio_input_positions=torch.tensor([[1, 2]]),
        audio_contexts=(
            SemanticAcousticCodes(
                semantic=torch.zeros(2, 1, dtype=torch.long),
                acoustic=torch.zeros(3, 2, dtype=torch.long),
            ),
        ),
    )


def _exception_payload(
    callback: OOMDiagnostics,
    trainer,
    module,
    error: BaseException,
):
    output = io.StringIO()
    with redirect_stderr(output):
        callback.on_exception(trainer, module, error)
    lines = output.getvalue().splitlines()
    if len(lines) != 1:
        raise AssertionError(f"expected one diagnostic line, received {lines!r}")
    return json.loads(lines[0])


if __name__ == "__main__":
    unittest.main()
