from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omegaconf import OmegaConf

from scripts._overfit_composition import decoder, flow, rvq, token
from speech_to_speech.pl_module import Config


class OverfitCompositionTest(unittest.TestCase):
    def test_decoder_forwards_every_option(self):
        options = decoder(
            OmegaConf.create(
                {"hidden_dim": 12, "layers": 3, "heads": 2, "ffn_ratio": 5}
            )
        )

        self.assertEqual(
            options,
            {"hidden_dim": 12, "layers": 3, "heads": 2, "ffn_ratio": 5},
        )

    @patch("scripts._overfit_composition.SpeechToSpeechModule")
    @patch("scripts._overfit_composition.TokenObjective")
    @patch("scripts._overfit_composition.TokenModel")
    def test_token_closes_model_and_objective(self, model, objective, module):
        runtime = SimpleNamespace(layout=Mock())

        built_module, built_model = token(runtime, Config())

        self.assertIs(built_model, model.return_value)
        objective.assert_called_once_with(runtime.layout)
        module.assert_called_once_with(
            unittest.mock.ANY,
            model=model.return_value,
            objective=objective.return_value,
        )
        self.assertIs(built_module, module.return_value)

    @patch("scripts._overfit_composition.SpeechToSpeechModule")
    @patch("scripts._overfit_composition.FlowObjective")
    @patch("scripts._overfit_composition.SpeechToSpeechFlowModel")
    @patch("scripts._overfit_composition.WavLMTeacher")
    def test_flow_closes_repa_model_and_objective(
        self,
        teacher,
        model,
        objective,
        module,
    ):
        teacher.return_value.feature_dim = 7
        runtime = SimpleNamespace(
            codec=Mock(),
            layout=Mock(),
            flow_matching=Mock(),
            backbone=SimpleNamespace(
                get_input_embeddings=lambda: SimpleNamespace(
                    weight=SimpleNamespace(device="cpu")
                )
            ),
        )
        acoustic = OmegaConf.create(
            {
                "decoder": {
                    "hidden_dim": None,
                    "layers": 2,
                    "heads": 1,
                    "ffn_ratio": 3,
                },
                "repa": {
                    "weight": 0.2,
                    "teacher_checkpoint": "teacher",
                    "teacher_layer": 4,
                    "student_layer": 1,
                },
            }
        )

        _, built_model, weight = flow(runtime, Config(), acoustic)

        self.assertIs(built_model, model.return_value)
        self.assertEqual(weight, 0.2)
        self.assertEqual(
            model.call_args.kwargs["repa"],
            {"feature_dim": 7, "student_layer": 1},
        )
        self.assertEqual(
            objective.call_args.kwargs["repa"],
            {"weight": 0.2, "teacher": teacher.return_value},
        )

    @patch("scripts._overfit_composition.SpeechToSpeechModule")
    @patch("scripts._overfit_composition.RVQObjective")
    @patch("scripts._overfit_composition.SpeechToSpeechRVQModel")
    def test_rvq_model_receives_only_decoder_options(self, model, objective, module):
        runtime = SimpleNamespace(layout=Mock())
        acoustic = OmegaConf.create(
            {
                "decoder": {
                    "hidden_dim": None,
                    "layers": 2,
                    "heads": 1,
                    "ffn_ratio": 3,
                }
            }
        )

        rvq(runtime, Config(), acoustic)

        self.assertEqual(set(model.call_args.kwargs), {"runtime", "decoder"})


if __name__ == "__main__":
    unittest.main()
