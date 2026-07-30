from __future__ import annotations

import unittest

import torch

from speech_to_speech.datamodule.types import ModelBatch, ModelSample
from speech_to_speech.generation.batch import requests_from_batch
from speech_to_speech.task import Task


class AudioInputContractTest(unittest.TestCase):
    def test_source_positions_are_padded_and_preserved_for_generation(self):
        batch = ModelBatch.from_samples(
            [
                ModelSample.from_sequence(
                    torch.tensor([1, 8, 9, 11, 2]),
                    torch.tensor([-100, -100, -100, -100, 2]),
                    task=Task.S2ST,
                    prediction=Task.S2ST.prediction_modality,
                    audio_input_positions=torch.tensor([1, 2]),
                ),
                ModelSample.from_sequence(
                    torch.tensor([1, 8, 11, 2]),
                    torch.tensor([-100, -100, -100, 2]),
                    task=Task.S2ST,
                    prediction=Task.S2ST.prediction_modality,
                    audio_input_positions=torch.tensor([1]),
                ),
            ],
            pad_token_id=0,
        )

        if batch.audio_input_positions is None:
            self.fail("batch did not retain source audio positions")
        torch.testing.assert_close(
            batch.audio_input_positions,
            torch.tensor([[1, 2], [1, -1]]),
        )
        requests = requests_from_batch(batch)
        torch.testing.assert_close(
            requests[0]["audio_input_positions"],
            torch.tensor([1, 2]),
        )
        torch.testing.assert_close(
            requests[1]["audio_input_positions"],
            torch.tensor([1]),
        )
        for request, prompt_length in zip(
            requests,
            batch.generation_prompt_lengths,
        ):
            self.assertEqual(
                int(request["prompt_ids"].numel()),
                int(prompt_length.item()),
            )
            self.assertEqual(request.get("prediction"), Task.S2ST.prediction_modality)

    def test_source_positions_reject_duplicates_and_out_of_range_values(self):
        sample = ModelSample.from_sequence(
            torch.tensor([1, 2]),
            torch.tensor([-100, 2]),
            task=Task.S2ST,
            prediction=Task.S2ST.prediction_modality,
            audio_input_positions=torch.tensor([1, 1]),
        )
        with self.assertRaisesRegex(ValueError, "must not repeat"):
            ModelBatch.from_samples([sample], pad_token_id=0)

        sample = ModelSample.from_sequence(
            torch.tensor([1, 2]),
            torch.tensor([-100, 2]),
            task=Task.S2ST,
            prediction=Task.S2ST.prediction_modality,
            audio_input_positions=torch.tensor([2]),
        )
        with self.assertRaisesRegex(ValueError, "valid sequence positions"):
            ModelBatch.from_samples([sample], pad_token_id=0)


if __name__ == "__main__":
    unittest.main()
