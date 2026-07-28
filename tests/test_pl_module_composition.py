from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import torch
from semantic_acoustic_codec.config import Route

from speech_to_speech.model import Config as ModelConfig
from speech_to_speech.model.acoustic import DecoderConfig
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.pl_module.composition import flow, rvq, token


class PlModuleCompositionTest(unittest.TestCase):
    @patch("speech_to_speech.pl_module.composition.SpeechToSpeechModule")
    @patch("speech_to_speech.pl_module.composition.TokenObjective")
    @patch("speech_to_speech.pl_module.composition.TokenModel")
    def test_token_closes_model_and_objective(self, model, objective, module):
        runtime = SimpleNamespace(layout=Mock(), audio_tokenizer=None)
        model_config = ModelConfig()

        built_module, built_model = token(runtime, ModuleConfig(), model_config)

        self.assertIs(built_model, model.return_value)
        model.assert_called_once_with(model_config, runtime=runtime)
        objective.assert_called_once_with(runtime.layout, runtime.audio_tokenizer)
        module.assert_called_once_with(
            ANY,
            model=model.return_value,
            objective=objective.return_value,
        )
        self.assertIs(built_module, module.return_value)

    @patch("speech_to_speech.pl_module.composition.SpeechToSpeechModule")
    @patch("speech_to_speech.pl_module.composition.FlowObjective")
    @patch("speech_to_speech.pl_module.composition.FlowModel")
    @patch("speech_to_speech.pl_module.composition.WavLMTeacher")
    def test_flow_closes_repa_model_and_objective(
        self,
        teacher,
        model,
        objective,
        module,
    ):
        teacher.return_value.feature_dim = 7
        runtime = SimpleNamespace(
            codec=SimpleNamespace(
                sample_rate=16_000,
                frame_rate=50.0,
                codebook_sizes=(8, 3),
                encode=Mock(),
                decode=Mock(),
            ),
            semantic_codec=Mock(),
            layout=Mock(),
            audio_tokenizer=None,
            flow_matching=Mock(),
            backbone=SimpleNamespace(
                get_input_embeddings=lambda: SimpleNamespace(
                    weight=SimpleNamespace(device="cpu")
                )
            ),
        )
        acoustic = SimpleNamespace(
            init_artifact=None,
            decoder=DecoderConfig(hidden_dim=None, layers=2, heads=1, ffn_ratio=3),
            repa=SimpleNamespace(
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

    @patch("speech_to_speech.pl_module.composition.SpeechToSpeechModule")
    @patch("speech_to_speech.pl_module.composition.RVQObjective")
    @patch("speech_to_speech.pl_module.composition.RVQModel")
    def test_rvq_model_receives_only_decoder_options(self, model, objective, module):
        runtime = SimpleNamespace(layout=Mock(), audio_tokenizer=None)
        acoustic = SimpleNamespace(
            init_artifact=None,
            decoder=DecoderConfig(hidden_dim=None, layers=2, heads=1, ffn_ratio=3),
        )
        model_config = ModelConfig()

        rvq(runtime, ModuleConfig(), model_config, acoustic)

        self.assertIs(model.call_args.args[0], model_config)
        self.assertEqual(
            set(model.call_args.kwargs),
            {"runtime", "decoder", "initialization"},
        )

    @patch("speech_to_speech.pl_module.composition.SpeechToSpeechModule")
    @patch("speech_to_speech.pl_module.composition.RVQObjective")
    @patch("speech_to_speech.pl_module.composition.RVQModel")
    @patch("speech_to_speech.pl_module.composition.load_acoustic_initialization")
    def test_rvq_loads_joint_initialization_in_composition(
        self,
        load_initialization,
        model,
        objective,
        module,
    ):
        runtime = SimpleNamespace(
            codec=Mock(),
            layout=Mock(),
            audio_tokenizer=None,
            backbone=SimpleNamespace(
                get_input_embeddings=lambda: SimpleNamespace(
                    weight=SimpleNamespace(device=torch.device("cpu"))
                )
            ),
        )
        acoustic = SimpleNamespace(
            init_artifact="/tmp/acoustic-generator",
            decoder=DecoderConfig(hidden_dim=None, layers=2, heads=1, ffn_ratio=3),
        )

        rvq(runtime, ModuleConfig(), ModelConfig(), acoustic)

        load_initialization.assert_called_once_with(
            "/tmp/acoustic-generator",
            codec=runtime.codec,
            route=Route.RVQ,
            device=torch.device("cpu"),
        )
        self.assertIs(
            model.call_args.kwargs["initialization"],
            load_initialization.return_value,
        )


if __name__ == "__main__":
    unittest.main()
