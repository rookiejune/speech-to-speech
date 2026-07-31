from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from anytrain.framework.rl import gather_token_logps, sequence_logps

from speech_to_speech.datamodule.preference import PreferenceBatch
from speech_to_speech.datamodule.types import ModelBatch, PredictionModality
from speech_to_speech.loss.preference import DPOObjective
from speech_to_speech.task import Task


class _DPOModel:
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
        logits[:, 1, 2] = 3.0
        logits[:, 1, 3] = -1.0
        return logits


def _batch(token_ids: list[int], *, supervised: bool = True) -> ModelBatch:
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    token_labels = input_ids.clone()
    token_labels[:, 0] = -100
    if not supervised:
        token_labels[:] = -100
        token_labels[:, 0] = input_ids[:, 0]
    return ModelBatch(
        input_ids=input_ids,
        token_labels=token_labels,
        acoustic_target=None,
        tasks=(Task.T2TT,),
        predictions=(PredictionModality.TEXT,),
        pad_token_id=0,
    )


def _logps(batch: ModelBatch, model: _DPOModel) -> torch.Tensor:
    logits = model.token_logits(model.token_hidden_states(batch.input_ids))
    token_logps = gather_token_logps(logits[:, :-1], batch.input_ids[:, 1:])
    return sequence_logps(token_logps, batch.token_labels[:, 1:].ne(-100))


class PreferenceLossTest(unittest.TestCase):
    def test_dpo_objective_scores_teacher_forced_preferences(self):
        model = _DPOModel()
        chosen = _batch([0, 1, 2])
        rejected = _batch([0, 1, 3])
        objective = DPOObjective(beta=0.5, reference_free=True)

        outputs = objective(PreferenceBatch(chosen=chosen, rejected=rejected), model)

        margin = _logps(chosen, model) - _logps(rejected, model)
        expected = -F.logsigmoid(0.5 * margin).mean()
        torch.testing.assert_close(outputs["loss"], expected)
        dpo = outputs["dpo"]
        self.assertIn("preferences", dpo.details or {})
        self.assertEqual(float((dpo.details or {})["accuracy"]), 1.0)

    def test_preference_batch_validates_reference_logp_shape(self):
        chosen = _batch([0, 1, 2])
        rejected = _batch([0, 1, 3])

        with self.assertRaisesRegex(ValueError, "shape"):
            PreferenceBatch(
                chosen=chosen,
                rejected=rejected,
                ref_chosen_logps=torch.zeros(2),
                ref_rejected_logps=torch.zeros(2),
            )

    def test_dpo_objective_requires_preference_batch(self):
        with self.assertRaisesRegex(TypeError, "PreferenceBatch"):
            DPOObjective(reference_free=True)(_batch([0, 1, 2]), _DPOModel())


if __name__ == "__main__":
    unittest.main()
