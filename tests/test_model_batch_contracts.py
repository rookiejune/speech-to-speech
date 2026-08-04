from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _contracts_helpers import *


class ModelBatchContractTest(unittest.TestCase):
    def test_model_batch_preserves_typed_input_hints_across_transfer(self):
        batch = ModelBatch.from_samples([_sample(Task.ASR)], pad_token_id=99)
        modalities = frozenset({Modality.TEXT, Modality.AUDIO})
        batch.set_input_hints(
            modalities,
            audio_input_positions_validated=True,
        )

        moved = batch.to(torch.device("cpu"))

        self.assertEqual(moved.input_modalities, modalities)
        self.assertTrue(moved.audio_input_positions_validated)

    def test_model_batch_rejects_invalid_input_hints(self):
        batch = ModelBatch.from_samples([_sample(Task.ASR)], pad_token_id=99)

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            batch.set_input_hints(
                frozenset(),
                audio_input_positions_validated=True,
            )
        with self.assertRaisesRegex(TypeError, "Modality values"):
            batch.set_input_hints(
                cast(Any, frozenset({"text"})),
                audio_input_positions_validated=True,
            )

    def test_model_batch_transfer_reuses_validated_unit_metadata(self):
        batch = ModelBatch.from_samples([_sample(Task.ASR)], pad_token_id=99)
        expected = batch.training_units("tokens")

        with patch(
            "speech_to_speech.datamodule.types._validate_batch_tensors",
            side_effect=AssertionError("trusted transfer must not revalidate"),
        ):
            moved = batch.to(torch.device("cpu"), non_blocking=True)

        self.assertEqual(moved.training_units("tokens"), expected)

    def test_model_batch_rejects_mixed_execution_signatures(self):
        samples = [
            _sample(Task.ASR),
            _sample(Task.TEXT_AR),
        ]
        with self.assertRaisesRegex(ValueError, "same execution signature"):
            ModelBatch.from_samples(samples, pad_token_id=99)

    def test_model_batch_direct_constructor_maintains_batch_task_invariants(self):
        def batch(tasks: list[Task]) -> ModelBatch:
            predictions = [
                (
                    task.prediction_modality
                    if isinstance(task, Task)
                    else cast(object, task)
                )
                for task in tasks
            ]
            return ModelBatch(
                input_ids=torch.ones(2, 2, dtype=torch.long),
                token_labels=torch.ones(2, 2, dtype=torch.long),
                acoustic_target=None,
                tasks=tasks,
                predictions=predictions,  # type: ignore[arg-type]
                pad_token_id=99,
                generation_prompt_lengths=torch.ones(2, dtype=torch.long),
            )

        cases = (
            ([], ValueError, "one Task per row"),
            ([Task.ASR], ValueError, "one Task per row"),
            (
                [Task.ASR, Task.TEXT_AR],
                ValueError,
                "same execution signature",
            ),
        )

        for tasks, error, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(error, message):
                batch(tasks)

        with self.assertRaisesRegex(TypeError, "Task values"):
            batch([Task.ASR, cast(Task, "asr")])

        with self.assertRaisesRegex(ValueError, "at least one row"):
            ModelBatch(
                input_ids=torch.empty(0, 2, dtype=torch.long),
                token_labels=torch.empty(0, 2, dtype=torch.long),
                acoustic_target=None,
                tasks=[],
                predictions=[],
                pad_token_id=99,
            )

        with self.assertRaisesRegex(TypeError, "signed integer"):
            ModelBatch(
                input_ids=torch.ones(1, 2, dtype=torch.uint64),
                token_labels=torch.ones(1, 2, dtype=torch.long),
                acoustic_target=None,
                tasks=[Task.ASR],
                predictions=[Task.ASR.prediction_modality],
                pad_token_id=99,
            )

    def test_model_batch_accepts_unified_audio_target(self):
        batch = ModelBatch.from_samples([_sample(Task.TTS)], pad_token_id=99)

        self.assertIsNone(batch.acoustic_target)

    def test_model_batch_row_preserves_one_acoustic_target(self):
        batch = ModelBatch.from_samples(
            [
                _target_sample(torch.tensor([[1, 2]])),
                _target_sample(torch.tensor([[3, 4], [5, 6]])),
            ],
            pad_token_id=99,
        )

        row = batch.row(1)

        self.assertEqual(row.tasks, [Task.TTS])
        self.assertTrue(torch.equal(row.input_ids, batch.input_ids[1:2]))
        self.assertIsNotNone(row.acoustic_target)
        if row.acoustic_target is None or batch.acoustic_target is None:
            self.fail("acoustic target was dropped while selecting a row")
        self.assertTrue(
            torch.equal(
                row.acoustic_target["codes"],
                batch.acoustic_target["codes"][1:2],
            )
        )
        with self.assertRaises(IndexError):
            batch.row(2)

    def test_model_batch_owns_acoustic_target_position_constraints(self):
        def batch(position: int, codes: torch.Tensor | None = None) -> ModelBatch:
            return ModelBatch(
                input_ids=torch.tensor([[1, 4]]),
                token_labels=torch.tensor([[-100, 4]]),
                acoustic_target={
                    "semantic_codes": torch.tensor([[[1]]]),
                    "codes": (torch.tensor([[[1, 2]]]) if codes is None else codes),
                    "token_positions": torch.tensor([[position]]),
                },
                tasks=[Task.TTS],
                predictions=[Task.TTS.prediction_modality],
                pad_token_id=99,
            )

        with self.assertRaisesRegex(ValueError, "at least 1"):
            batch(0)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            batch(2)
        with self.assertRaisesRegex(ValueError, "whole padded frame"):
            batch(-1, torch.tensor([[[-1, 2]]]))

    def test_model_batch_rejects_padding_ids_inside_unpadded_acoustic_fields(self):
        samples = {
            "acoustic target codes": _target_sample(torch.tensor([[-1, 2]])),
            "target semantic codes": _target_sample(
                torch.tensor([[1, 2]]),
                semantic_codes=torch.tensor([[-1]]),
            ),
        }

        for name, sample in samples.items():
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    ValueError, f"{name} must contain non-negative codec IDs"
                ),
            ):
                ModelBatch.from_samples([sample], pad_token_id=99)

    def test_model_batch_rejects_malformed_acoustic_code_tensors(self):
        cases = (
            (
                _target_sample(torch.tensor([1, 2])),
                ValueError,
                "acoustic target codes must have shape",
            ),
            (
                _target_sample(torch.empty((0, 2), dtype=torch.long)),
                ValueError,
                "acoustic target codes must contain at least one frame",
            ),
            (
                _target_sample(torch.tensor([[1.0, 2.0]])),
                TypeError,
                "acoustic target codes must contain integer codec IDs",
            ),
            (
                _target_sample(
                    torch.tensor([[1, 2]]),
                    semantic_codes=torch.tensor([1]),
                ),
                ValueError,
                "target semantic codes must have shape",
            ),
            (
                _target_sample(
                    torch.tensor([[1, 2], [2, 1]]),
                    semantic_codes=torch.tensor([[1]]),
                ),
                ValueError,
                "semantic and acoustic codes must share the frame axis",
            ),
        )

        for sample, error, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(error, message):
                ModelBatch.from_samples([sample], pad_token_id=99)


if __name__ == "__main__":
    unittest.main()
