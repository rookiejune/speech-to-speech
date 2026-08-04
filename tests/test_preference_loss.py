from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from anydataset.types import Modality
from anytrain.framework.rl import gather_token_logps, sequence_logps
from anytrain.module.idspace import Layout

from speech_to_speech.rl import PreferenceBatch
from speech_to_speech.datamodule.types import ModelBatch, PredictionModality
from speech_to_speech.loss.preference import DPOObjective
from speech_to_speech.task import Task


class _DPOModel:
    def __init__(self) -> None:
        self.layout = Layout(text=(0, 5), audio=(5, 7))
        self.hidden_calls = 0
        self.hidden_batch_sizes: list[int] = []
        self.full_vocab_calls = 0
        self.logit_modalities: list[Modality] = []
        self.audio_input_positions: torch.Tensor | None = None
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
        del attention_mask
        self.hidden_calls += 1
        self.hidden_batch_sizes.append(input_ids.size(0))
        self.audio_input_positions = audio_input_positions
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
            raise AssertionError("preference loss must not build global vocabulary logits")
        self.logit_modalities.append(modality)
        start, end = self.layout.blocks[modality.value]
        logits = hidden_states.new_zeros(hidden_states.size(0), end - start)
        if modality is Modality.TEXT:
            values = hidden_states[:, 0]
            logits[values.eq(0), 1] = 1.0
            logits[values.eq(1), 2] = 3.0
            logits[values.eq(1), 3] = -1.0
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


def _batch(
    token_ids: list[int],
    *,
    supervised: bool = True,
    audio_input_positions: torch.Tensor | None = None,
) -> ModelBatch:
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
        tasks=[Task.T2TT],
        predictions=[PredictionModality.TEXT],
        pad_token_id=0,
        audio_input_positions=audio_input_positions,
    )


def _logps(batch: ModelBatch, model: _DPOModel) -> torch.Tensor:
    mask = batch.token_labels[:, 1:].ne(-100)
    hidden = model.token_hidden_states(batch.input_ids)
    logits = model.token_logits(hidden[:, :-1][mask], Modality.TEXT)
    selected = gather_token_logps(logits, batch.input_ids[:, 1:][mask])
    token_logps = selected.new_zeros(mask.shape)
    token_logps[mask] = selected
    return sequence_logps(token_logps, mask)


class PreferenceLossTest(unittest.TestCase):
    def test_dpo_objective_scores_teacher_forced_preferences(self):
        model = _DPOModel()
        chosen = _batch([0, 1, 2])
        rejected = _batch([0, 1, 3])
        objective = DPOObjective(beta=0.5, reference_free=True)

        outputs = objective(PreferenceBatch(chosen=chosen, rejected=rejected), model)

        expected_model = _DPOModel()
        margin = _logps(chosen, expected_model) - _logps(rejected, expected_model)
        expected = -F.logsigmoid(0.5 * margin).mean()
        torch.testing.assert_close(outputs["loss"], expected)
        dpo = outputs["dpo"]
        self.assertIn("preferences", dpo.details or {})
        self.assertEqual(float((dpo.details or {})["accuracy"]), 1.0)
        self.assertEqual(model.hidden_calls, 1)
        self.assertEqual(model.hidden_batch_sizes, [2])
        self.assertEqual(model.predictions, [PredictionModality.TEXT])
        self.assertEqual(model.full_vocab_calls, 0)
        self.assertEqual(model.logit_modalities, [Modality.TEXT])

    def test_dpo_concatenates_different_audio_position_widths(self):
        chosen = _batch(
            [0, 1, 2],
            audio_input_positions=torch.tensor([[0, 1]]),
        )
        rejected = _batch(
            [0, 1, 3],
            audio_input_positions=torch.tensor([[1]]),
        )
        model = _DPOModel()

        DPOObjective(reference_free=True)(
            PreferenceBatch(chosen=chosen, rejected=rejected),
            model,
        )

        self.assertIsNotNone(model.audio_input_positions)
        torch.testing.assert_close(
            model.audio_input_positions,
            torch.tensor([[0, 1], [1, -1]]),
        )

    def test_dpo_combines_validated_input_modality_hints(self):
        chosen = _batch([0, 1, 2])
        rejected = _batch([0, 1, 3])
        chosen.set_input_hints(
            frozenset({Modality.TEXT}),
            audio_input_positions_validated=True,
        )
        rejected.set_input_hints(
            frozenset({Modality.TEXT}),
            audio_input_positions_validated=False,
        )
        model = _DPOModel()

        DPOObjective(reference_free=True)(
            PreferenceBatch(chosen=chosen, rejected=rejected),
            model,
        )

        self.assertEqual(
            model.input_hints,
            [(frozenset({Modality.TEXT}), False, True)],
        )

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
