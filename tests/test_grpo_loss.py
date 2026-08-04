from __future__ import annotations

import unittest

import torch
from anydataset.types import Modality
from anytrain.framework.rl import GRPOLoss, gather_token_logps
from anytrain.module.idspace import Layout

from speech_to_speech.rl import GRPOBatch
from speech_to_speech.datamodule.types import ModelBatch, PredictionModality
from speech_to_speech.loss.rollout import GRPOObjective
from speech_to_speech.task import Task


class _GRPOModel:
    def __init__(self) -> None:
        self.layout = Layout(text=(0, 5), audio=(5, 7))
        self.hidden_calls = 0
        self.full_vocab_calls = 0
        self.logit_modalities: list[Modality] = []
        self.logit_rows: list[int] = []
        self.predictions: list[PredictionModality | None] = []
        self.input_hints: list[tuple[frozenset[Modality] | None, bool, bool]] = []

    def token_hidden_states(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        audio_input_positions: torch.Tensor | None = None,
        input_modalities: frozenset[Modality] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
        prediction: PredictionModality | None = None,
    ) -> torch.Tensor:
        del attention_mask, audio_input_positions
        self.hidden_calls += 1
        self.predictions.append(prediction)
        self.input_hints.append(
            (input_modalities, validate_input, validate_audio_input_positions)
        )
        return input_ids[..., None].to(dtype=torch.float32)

    def token_logits(
        self,
        hidden_states: torch.Tensor,
        modality: Modality | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
        if modality is None:
            self.full_vocab_calls += 1
            raise AssertionError("GRPO loss must not build global vocabulary logits")
        self.logit_modalities.append(modality)
        self.logit_rows.append(hidden_states.size(0))
        start, end = self.layout.blocks[modality.value]
        logits = hidden_states.new_zeros(hidden_states.size(0), end - start)
        values = hidden_states[:, 0]
        if modality is Modality.TEXT:
            logits[values.eq(0), 1] = 1.0
            logits[values.eq(1), 2] = 2.0
            logits[values.eq(1), 3] = -1.0
        else:
            logits[values.eq(1), 0] = 2.0
            logits[values.eq(1), 1] = -1.0
        return logits

    def project_audio_hidden(
        self,
        hidden_state: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        selection_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, object | None]:
        del attention_mask, past_key_values, use_cache
        if selection_mask is None:
            return hidden_state, None
        return hidden_state[selection_mask], None


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
    hidden = model.token_hidden_states(batch.input_ids)
    logits = model.token_logits(hidden[:, :-1].flatten(0, 1), Modality.TEXT)
    return gather_token_logps(logits, batch.input_ids[:, 1:].flatten()).view(1, 2, 2)


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
            policy_token_logps=_policy_logps(sequences, _GRPOModel()),
            old_token_logps=old_token_logps,
            rewards=rewards,
            response_mask=sequences.token_labels[:, 1:].ne(-100).view(1, 2, 2),
        )
        torch.testing.assert_close(outputs["loss"], expected)
        grpo = outputs["grpo"]
        self.assertIn("preferences", grpo.details or {})
        self.assertEqual(float((grpo.details or {})["candidate_count"]), 2.0)
        self.assertEqual(model.hidden_calls, 1)
        self.assertEqual(model.predictions, [PredictionModality.TEXT])
        self.assertEqual(model.full_vocab_calls, 0)
        self.assertEqual(model.logit_modalities, [Modality.TEXT])
        self.assertEqual(model.logit_rows, [4])

    def test_grpo_routes_mixed_targets_to_local_heads(self):
        input_ids = torch.tensor([[0, 1, 5], [0, 1, 6]], dtype=torch.long)
        token_labels = input_ids.clone()
        token_labels[:, 0] = -100
        sequences = ModelBatch(
            input_ids=input_ids,
            token_labels=token_labels,
            acoustic_target=None,
            tasks=[Task.INTERLEAVED_AR, Task.INTERLEAVED_AR],
            predictions=[PredictionModality.INTERLEAVED, PredictionModality.INTERLEAVED],
            pad_token_id=9,
        )
        model = _GRPOModel()

        outputs = GRPOObjective()(
            GRPOBatch(
                sequences=sequences,
                old_token_logps=torch.zeros(1, 2, 2),
                rewards=torch.tensor([[1.0, 0.0]]),
            ),
            model,
        )

        self.assertTrue(torch.isfinite(outputs["loss"]))
        self.assertEqual(model.full_vocab_calls, 0)
        self.assertEqual(
            model.logit_modalities,
            [Modality.AUDIO, Modality.TEXT],
        )
        self.assertEqual(model.logit_rows, [2, 2])

    def test_grpo_forwards_validated_input_modality_hints(self):
        sequences = _batch()
        sequences.set_input_hints(
            frozenset({Modality.TEXT}),
            audio_input_positions_validated=True,
        )
        model = _GRPOModel()

        GRPOObjective()(
            GRPOBatch(
                sequences=sequences,
                old_token_logps=torch.zeros(1, 2, 2),
                rewards=torch.tensor([[1.0, 0.0]]),
            ),
            model,
        )

        self.assertEqual(
            model.input_hints,
            [(frozenset({Modality.TEXT}), False, False)],
        )

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
