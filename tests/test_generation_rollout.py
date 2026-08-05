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
from speech_to_speech.model.generation import GenerationOutput
from speech_to_speech.task import ControlToken, FULL_COT, PredictionModality, Task


class _Runtime:
    layout = Layout(text=(0, 11), audio=(11, 14))
    lexical_text_vocab_size = 5
    control_token_ids = tuple(range(5, 11))
    pad_token_id = 0
    eos_token_id = 3

    def control_token_id(self, token: ControlToken) -> int:
        return self.control_token_ids[list(ControlToken).index(token)]

    def generation_allowed_ids(self, modality: Modality) -> tuple[int, ...]:
        if modality is not Modality.TEXT:
            raise ValueError("rollout test runtime only supports text generation.")
        return (1, 2, 3, 4)


class _RolloutModel:
    def __init__(self) -> None:
        self.runtime = _Runtime()
        self.prompt_ids: torch.Tensor | None = None
        self.prompt_attention_mask: torch.Tensor | None = None
        self.stop_token_id: int | None = None
        self.allowed_token_ids: tuple[int, ...] | None = None

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
        allowed_token_ids=None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> GenerationOutput:
        del max_new_tokens, temperature, top_p, audio_input_positions, do_sample, use_cache
        self.prompt_ids = prompt_ids.clone()
        self.prompt_attention_mask = None if prompt_attention_mask is None else prompt_attention_mask.clone()
        self.stop_token_id = stop_token_id
        self.allowed_token_ids = tuple(int(value) for value in allowed_token_ids)
        if generation_modality is not None:
            raise AssertionError("selected-id rollout must not force one modality head")
        if prompt_ids.size(0) == 1 and stop_token_id != self.runtime.eos_token_id:
            response = prompt_ids.new_tensor([[2, stop_token_id]])
            return GenerationOutput(
                sequences=torch.cat((prompt_ids, response), dim=1),
                token_logprobs=torch.tensor([[-0.2, -0.3]]),
                token_logprob_mask=torch.tensor([[True, True]]),
            )
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
                {"prompt_ids": torch.tensor([1]), "task": Task.TEXT_AR, "audio_input_positions": None},
                {"prompt_ids": torch.tensor([1, 2]), "task": Task.TEXT_AR, "audio_input_positions": None},
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
        self.assertEqual(rows[0]["response_ids"], [3])
        self.assertEqual(len(rows[0]["response_logprobs"]), 1)
        self.assertAlmostEqual(rows[0]["response_logprobs"][0], -0.1)
        self.assertEqual(rows[0]["finish_reason"], "eos")
        self.assertEqual(rows[1]["version"], 1)
        self.assertEqual(rows[1]["index"], 1)
        self.assertEqual(rows[1]["task"], Task.TEXT_AR.value)
        self.assertEqual(rows[1]["prediction"], PredictionModality.TEXT.value)
        self.assertEqual(rows[1]["prompt_ids"], [1, 2])
        self.assertEqual(rows[1]["response_ids"], [4, 3])
        self.assertEqual(rows[1]["finish_reason"], "eos")
        self.assertEqual(len(rows[1]["response_logprobs"]), 2)
        self.assertAlmostEqual(rows[1]["response_logprobs"][0], -0.2)
        self.assertEqual(model.stop_token_id, model.runtime.eos_token_id)
        self.assertEqual(set(model.allowed_token_ids or ()), {1, 2, 3, 4})

    def test_generate_rollouts_uses_mt_end_control(self):
        model = _RolloutModel()
        rows = generate_rollouts(
            [
                {
                    "prompt_ids": torch.tensor([7]),
                    "task": Task.T2TT,
                    "target_language": "English",
                    "audio_input_positions": None,
                }
            ],
            model,
            max_new_tokens=2,
            do_sample=False,
        )

        self.assertEqual(model.stop_token_id, 8)
        self.assertEqual(set(model.allowed_token_ids or ()), {1, 2, 4, 7, 8, 9})
        self.assertEqual(rows[0]["response_ids"], [2, 8])
        self.assertEqual(rows[0]["finish_reason"], "control")

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

    def test_generate_rollouts_rejects_prediction_override(self):
        with self.assertRaisesRegex(
            ValueError,
            "prediction override is not supported",
        ):
            generate_rollouts(
                [
                    {
                        "prompt_ids": torch.tensor([1]),
                        "task": Task.TEXT_AR,
                        "audio_input_positions": None,
                        "prediction": PredictionModality.AUDIO,
                    }
                ],
                _RolloutModel(),
                max_new_tokens=1,
            )

    def test_generate_rollouts_rejects_multi_step_text_trace(self):
        with self.assertRaisesRegex(
            ValueError,
            "single-step text responses only",
        ):
            generate_rollouts(
                [
                    {
                        "prompt_ids": torch.tensor([1]),
                        "task": Task.S2TT,
                        "trace": FULL_COT,
                        "target_language": "en",
                        "audio_input_positions": None,
                    }
                ],
                _RolloutModel(),
                max_new_tokens=1,
            )


if __name__ == "__main__":
    unittest.main()
