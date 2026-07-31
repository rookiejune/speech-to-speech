from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout

from speech_to_speech.generation.rollout import (
    generate_rollouts,
    write_rollouts_jsonl,
)
from speech_to_speech.model._generation import GenerationOutput
from speech_to_speech.prediction import PredictionModality
from speech_to_speech.task import Task


class _Runtime:
    layout = Layout(text=(0, 5), audio=(5, 8))
    pad_token_id = 0
    eos_token_id = 3


class _RolloutModel:
    def __init__(self) -> None:
        self.runtime = _Runtime()
        self.prompt_ids: torch.Tensor | None = None
        self.prompt_attention_mask: torch.Tensor | None = None

    def generate_tokens_with_logprobs(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        prompt_attention_mask: torch.Tensor | None = None,
        audio_input_positions: torch.Tensor | None = None,
        stop_token_id: int | None = None,
        generation_modality: Modality | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> GenerationOutput:
        del max_new_tokens, temperature, top_p, audio_input_positions, do_sample, use_cache
        self.prompt_ids = prompt_ids.clone()
        self.prompt_attention_mask = None if prompt_attention_mask is None else prompt_attention_mask.clone()
        if stop_token_id != self.runtime.eos_token_id:
            raise AssertionError("rollout must use runtime EOS as stop token")
        if generation_modality is not Modality.TEXT:
            raise AssertionError("rollout must use text generation modality")
        return GenerationOutput(
            sequences=torch.tensor([[0, 1, 3, 3], [1, 2, 4, 3]]),
            token_logprobs=torch.tensor([[-0.1, -9.0], [-0.2, -0.3]]),
            token_logprob_mask=torch.tensor([[True, False], [True, True]]),
        )


class GenerationRolloutTest(unittest.TestCase):
    def test_generate_rollouts_records_response_logprobs(self):
        model = _RolloutModel()
        rows = generate_rollouts(
            [
                {"prompt_ids": torch.tensor([1]), "task": Task.T2TT, "audio_input_positions": None, "audio_context": None},
                {"prompt_ids": torch.tensor([1, 2]), "task": Task.T2TT, "audio_input_positions": None, "audio_context": None},
            ],
            model,
            max_new_tokens=2,
            do_sample=False,
        )

        self.assertTrue(torch.equal(model.prompt_ids, torch.tensor([[0, 1], [1, 2]])))
        self.assertTrue(
            torch.equal(
                model.prompt_attention_mask,
                torch.tensor([[False, True], [True, True]]),
            )
        )
        self.assertEqual(rows[0]["response_ids"], [])
        self.assertEqual(rows[0]["response_logprobs"], [])
        self.assertEqual(rows[0]["finish_reason"], "eos")
        self.assertEqual(rows[1]["version"], 1)
        self.assertEqual(rows[1]["index"], 1)
        self.assertEqual(rows[1]["task"], Task.T2TT.value)
        self.assertEqual(rows[1]["prediction"], PredictionModality.TEXT.value)
        self.assertEqual(rows[1]["prompt_ids"], [1, 2])
        self.assertEqual(rows[1]["response_ids"], [4])
        self.assertEqual(rows[1]["finish_reason"], "eos")
        self.assertEqual(len(rows[1]["response_logprobs"]), 1)
        self.assertAlmostEqual(rows[1]["response_logprobs"][0], -0.2)

    def test_write_rollouts_jsonl_writes_plain_json_rows(self):
        rows = [
            {
                "version": 1,
                "index": 0,
                "task": Task.T2TT.value,
                "prediction": PredictionModality.TEXT.value,
                "prompt_ids": [1],
                "response_ids": [2],
                "response_logprobs": [-0.5],
                "finish_reason": "length",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.jsonl"
            write_rollouts_jsonl(path, rows)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]), rows[0])

    def test_generate_rollouts_rejects_non_text_prediction(self):
        with self.assertRaisesRegex(ValueError, "text prediction only"):
            generate_rollouts(
                [
                    {
                        "prompt_ids": torch.tensor([1]),
                        "task": Task.T2TT,
                        "audio_input_positions": None,
                        "audio_context": None,
                        "prediction": PredictionModality.AUDIO,
                    }
                ],
                _RolloutModel(),
                max_new_tokens=1,
            )


if __name__ == "__main__":
    unittest.main()
