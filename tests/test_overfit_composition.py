from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts._config import FlowConfig, RepaConfig, RVQConfig
from scripts._overfit_composition import flow, rvq, token
from speech_to_speech.model import Config as ModelConfig, DecoderConfig
from speech_to_speech.pl_module import Config as ModuleConfig


class OverfitCompositionTest(unittest.TestCase):
    @patch("scripts._overfit_composition.SpeechToSpeechModule")
    @patch("scripts._overfit_composition.TokenObjective")
    @patch("scripts._overfit_composition.TokenModel")
    def test_token_closes_model_and_objective(self, model, objective, module):
        runtime = SimpleNamespace(layout=Mock())
        model_config = ModelConfig()

        built_module, built_model = token(runtime, ModuleConfig(), model_config)

        self.assertIs(built_model, model.return_value)
        model.assert_called_once_with(model_config, runtime=runtime)
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
        acoustic = FlowConfig(
            type="flow",
            name="flow-2l",
            decoder=DecoderConfig(hidden_dim=None, layers=2, heads=1, ffn_ratio=3),
            repa=RepaConfig(
                weight=0.2,
                teacher_checkpoint="teacher",
                teacher_layer=4,
                student_layer=1,
            ),
        )
        model_config = ModelConfig()

        _, built_model, weight = flow(
            runtime,
            ModuleConfig(),
            model_config,
            acoustic,
        )

        self.assertIs(built_model, model.return_value)
        self.assertEqual(weight, 0.2)
        self.assertIs(model.call_args.args[0], model_config)
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
        acoustic = RVQConfig(
            type="rvq",
            name="rvq-2l",
            decoder=DecoderConfig(hidden_dim=None, layers=2, heads=1, ffn_ratio=3),
        )
        model_config = ModelConfig()

        rvq(runtime, ModuleConfig(), model_config, acoustic)

        self.assertIs(model.call_args.args[0], model_config)
        self.assertEqual(set(model.call_args.kwargs), {"runtime", "decoder"})


if __name__ == "__main__":
    unittest.main()
