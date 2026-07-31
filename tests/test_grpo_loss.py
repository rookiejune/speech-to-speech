from __future__ import annotations

import unittest

import torch
from anytrain.framework.rl import GRPOLoss, gather_token_logps

from speech_to_speech.datamodule.rollout import GRPOBatch
from speech_to_speech.datamodule.types import ModelBatch, PredictionModality
from speech_to_speech.loss.rollout import GRPOObjective
from speech_to_speech.task import Task


class _GRPOModel:
    def token_hidden_states(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        audio_input_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask, audio_input_positions
        return input_ids

    def token_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = hidden_states.shape
        logits = hidden_states.new_zeros(
            batch_size,
            sequence_length,
            5,
            dtype=torch.float32,
        )
        logits[:, 0, 1] = 1.0
        logits[:, 1, 2] = 2.0
        logits[:, 1, 3] = -1.0
        return logits


def _batch() -> ModelBatch:
    input_ids = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.long)
    token_labels = input_ids.clone()
    token_labels[:, 0] = -100
    return ModelBatch(
        input_ids=input_ids,
        token_labels=token_labels,
        acoustic_target=None,
        tasks=(Task.T2TT, Task.T2TT),
        predictions=(PredictionModality.TEXT, PredictionModality.TEXT),
        pad_token_id=0,
    )


def _policy_logps(batch: ModelBatch, model: _GRPOModel) -> torch.Tensor:
    logits = model.token_logits(model.token_hidden_states(batch.input_ids))
    return gather_token_logps(logits[:, :-1], batch.input_ids[:, 1:]).view(1, 2, 2)


class GRPOLossTest(unittest.TestCase):
    def test_grpo_objective_scores_teacher_forced_rollouts(self):
        model = _GRPOModel()
        sequences = _batch()
        old_token_logps = torch.zeros(1, 2, 2)
        rewards = torch.tensor([[2.0, 0.0]])
        objective = GRPOObjective(clip_range=0.2)

        outputs = objective(
            GRPOBatch(
                sequences=sequences,
                old_token_logps=old_token_logps,
                rewards=rewards,
            ),
            model,
        )

        expected, _ = GRPOLoss(clip_range=0.2)(
            policy_token_logps=_policy_logps(sequences, model),
            old_token_logps=old_token_logps,
            rewards=rewards,
            response_mask=sequences.token_labels[:, 1:].ne(-100).view(1, 2, 2),
        )
        torch.testing.assert_close(outputs["loss"], expected)
        grpo = outputs["grpo"]
        self.assertIn("preferences", grpo.details or {})
        self.assertEqual(float((grpo.details or {})["candidate_count"]), 2.0)

    def test_grpo_batch_validates_rollout_shape(self):
        with self.assertRaisesRegex(ValueError, "batch times group"):
            GRPOBatch(
                sequences=_batch(),
                old_token_logps=torch.zeros(2, 2, 2),
                rewards=torch.zeros(2, 2),
            )

    def test_grpo_objective_requires_grpo_batch(self):
        with self.assertRaisesRegex(TypeError, "GRPOBatch"):
            GRPOObjective()(_batch(), _GRPOModel())


if __name__ == "__main__":
    unittest.main()
