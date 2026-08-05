from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _contracts_helpers import *


class ModelBatchContractTest(unittest.TestCase):
    def test_model_batch_preserves_normalized_target_languages(self):
        samples = [
            ModelSample.from_sequence(
                torch.tensor([1, 2]),
                torch.tensor([-100, 2]),
                task=Task.MT,
                target_language="EN-US",
            ),
            ModelSample.from_sequence(
                torch.tensor([1, 3]),
                torch.tensor([-100, 3]),
                task=Task.MT,
                target_language="Chinese",
            ),
        ]

        batch = ModelBatch.from_samples(samples, pad_token_id=99)

        self.assertEqual(batch.target_languages, ["en", "zh"])
        self.assertEqual(batch.row(1).target_languages, ["zh"])
        self.assertEqual(
            batch.to(torch.device("cpu")).target_languages,
            ["en", "zh"],
        )

    def test_model_batch_rejects_misaligned_target_languages(self):
        with self.assertRaisesRegex(ValueError, "one value per row"):
            ModelBatch(
                input_ids=torch.ones(2, 2, dtype=torch.long),
                token_labels=torch.ones(2, 2, dtype=torch.long),
                acoustic_target=None,
                tasks=[Task.ASR, Task.ASR],
                target_languages=["en"],
                pad_token_id=99,
                generation_prompt_lengths=torch.ones(2, dtype=torch.long),
            )

    def test_model_batch_rejects_supervision_outside_layout(self):
        sample = ModelSample.from_sequence(
            torch.tensor([1, 999]),
            torch.tensor([-100, 999]),
            task=Task.ASR,
        )

        with self.assertRaisesRegex(ValueError, "outside the supervised layout"):
            ModelBatch.from_samples(
                [sample],
                pad_token_id=0,
                layout=Layout(text=(0, 10), audio=(10, 20)),
            )

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
            "speech_to_speech.datamodule._batch_ops._validate_batch_tensors",
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
            return ModelBatch(
                input_ids=torch.ones(2, 2, dtype=torch.long),
                token_labels=torch.ones(2, 2, dtype=torch.long),
                acoustic_target=None,
                tasks=tasks,
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
                pad_token_id=99,
            )

        with self.assertRaisesRegex(TypeError, "signed integer"):
            ModelBatch(
                input_ids=torch.ones(1, 2, dtype=torch.uint64),
                token_labels=torch.ones(1, 2, dtype=torch.long),
                acoustic_target=None,
                tasks=[Task.ASR],
                pad_token_id=99,
            )

    def test_model_batch_accepts_unified_audio_target(self):
        batch = ModelBatch.from_samples([_sample(Task.TTS)], pad_token_id=99)

        self.assertIsNone(batch.acoustic_target)

    def test_model_batch_preserves_source_and_target_ctc_contracts(self):
        samples = [
            ModelSample.from_sequence(
                torch.tensor([10, 11, 12, 13, 14]),
                torch.tensor([-100, -100, -100, 13, 14]),
                source_ctc={
                    "token_positions": torch.tensor([0, 1]),
                    "text_token_ids": torch.tensor([1]),
                },
                target_ctc={
                    "token_positions": torch.tensor([3, 4]),
                    "text_token_ids": torch.tensor([2]),
                },
                task=Task.S2ST,
            ),
            ModelSample.from_sequence(
                torch.tensor([10, 11, 12, 13, 14, 15, 16]),
                torch.tensor([-100, -100, -100, -100, 14, 15, 16]),
                source_ctc={
                    "token_positions": torch.tensor([0, 1, 2]),
                    "text_token_ids": torch.tensor([1, 2]),
                },
                target_ctc={
                    "token_positions": torch.tensor([4, 5, 6]),
                    "text_token_ids": torch.tensor([2, 3]),
                },
                task=Task.S2ST,
            ),
        ]

        batch = ModelBatch.from_samples(samples, pad_token_id=99)

        self.assertIsNotNone(batch.source_ctc)
        self.assertIsNotNone(batch.target_ctc)
        assert batch.source_ctc is not None and batch.target_ctc is not None
        self.assertTrue(
            torch.equal(batch.source_ctc["token_positions"][0], torch.tensor([0, 1, -1]))
        )
        self.assertTrue(
            torch.equal(batch.target_ctc["text_token_ids"][0], torch.tensor([2, -1]))
        )
        row = batch.row(1)
        self.assertIsNotNone(row.source_ctc)
        assert row.source_ctc is not None
        self.assertTrue(
            torch.equal(
                row.source_ctc["token_positions"],
                batch.source_ctc["token_positions"][1:2],
            )
        )
        moved = batch.to(torch.device("cpu"))
        self.assertIsNotNone(moved.target_ctc)
        assert moved.target_ctc is not None
        self.assertTrue(
            torch.equal(
                moved.target_ctc["text_token_ids"],
                batch.target_ctc["text_token_ids"],
            )
        )

    def test_model_batch_rejects_invalid_ctc_positions_and_lengths(self):
        def sample(*, positions: list[int], labels: list[int]) -> ModelSample:
            return ModelSample.from_sequence(
                torch.tensor([10, 11, 12]),
                torch.tensor([-100, 11, 12]),
                target_ctc={
                    "token_positions": torch.tensor(positions),
                    "text_token_ids": torch.tensor(labels),
                },
                task=Task.T2ST,
            )

        with self.assertRaisesRegex(ValueError, "positive valid sequence positions"):
            ModelBatch.from_samples(
                [sample(positions=[0, 1], labels=[1])],
                pad_token_id=99,
            )
        with self.assertRaisesRegex(ValueError, "requires at least 3 audio positions"):
            ModelBatch.from_samples(
                [sample(positions=[1, 2], labels=[1, 1])],
                pad_token_id=99,
            )

        with self.assertRaisesRegex(ValueError, "transcript visibility route"):
            ModelBatch.from_samples(
                [
                    ModelSample.from_sequence(
                        torch.tensor([0, 3, 4]),
                        torch.tensor([-100, 3, 4]),
                        target_ctc={
                            "token_positions": torch.tensor([1, 2]),
                            "text_token_ids": torch.tensor([1]),
                        },
                        task=Task.TTS,
                    )
                ],
                pad_token_id=99,
            )

        with self.assertRaisesRegex(ValueError, "transcript visibility route"):
            ModelBatch.from_samples(
                [
                    ModelSample.from_sequence(
                        torch.tensor([0, 3, 4]),
                        torch.tensor([-100, 3, 4]),
                        source_ctc={
                            "token_positions": torch.tensor([0, 1]),
                            "text_token_ids": torch.tensor([1]),
                        },
                        task=Task.T2ST,
                    )
                ],
                pad_token_id=99,
            )

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
