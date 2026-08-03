from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

import torch
from semantic_acoustic_codec.config import Route

from speech_to_speech.model import Config as ModelConfig
from speech_to_speech.model.acoustic import AcousticType, DecoderConfig
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.pl_module.composition import build, flow, rvq, token


class PlModuleCompositionTest(unittest.TestCase):
    @patch("speech_to_speech.pl_module.composition.rvq")
    @patch("speech_to_speech.pl_module.composition.flow")
    @patch("speech_to_speech.pl_module.composition.token")
    def test_build_dispatches_acoustic_composition(self, token, flow, rvq):
        runtime = Mock(acoustic_side_channel=True)
        module_config = ModuleConfig()
        model_config = ModelConfig()
        token.return_value = (Mock(), Mock())
        flow.return_value = (Mock(), Mock())
        rvq.return_value = (Mock(), Mock())

        cases = (
            (AcousticType.NONE, token, token.return_value),
            (AcousticType.FLOW, flow, flow.return_value),
            (AcousticType.RVQ, rvq, rvq.return_value),
        )
        for acoustic_type, factory, expected in cases:
            with self.subTest(acoustic_type=acoustic_type):
                acoustic = SimpleNamespace(type=acoustic_type.value)

                result = build(runtime, module_config, model_config, acoustic)

                self.assertEqual(result, (acoustic_type, *expected))
                factory.assert_called_once()
                token.reset_mock()
                flow.reset_mock()
                rvq.reset_mock()

    @patch("speech_to_speech.pl_module.composition.flow")
    @patch("speech_to_speech.pl_module.composition.token")
    def test_build_owns_side_channel_constraint(self, token, flow):
        runtime = Mock(acoustic_side_channel=False)
        token.return_value = (Mock(), Mock())

        result = build(
            runtime,
            ModuleConfig(),
            ModelConfig(),
            SimpleNamespace(type=AcousticType.NONE.value),
        )

        self.assertEqual(result, (AcousticType.NONE, *token.return_value))

        with self.assertRaisesRegex(ValueError, "model/acoustic=none"):
            build(
                runtime,
                ModuleConfig(),
                ModelConfig(),
                SimpleNamespace(type=AcousticType.FLOW.value),
            )

        flow.assert_not_called()

    @patch("speech_to_speech.pl_module.composition.SpeechToSpeechModule")
    @patch("speech_to_speech.pl_module.composition.TokenObjective")
    @patch("speech_to_speech.pl_module.composition.Model")
    def test_token_closes_model_and_objective(self, model, objective, module):
        runtime = SimpleNamespace(layout=Mock())
        model_config = ModelConfig()

        built_module, built_model = token(runtime, ModuleConfig(), model_config)

        self.assertIs(built_model, model.return_value)
        model.assert_called_once_with(model_config, runtime=runtime)
        objective.assert_called_once_with(runtime.layout)
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

        _, built_model = flow(
            runtime,
            ModuleConfig(),
            model_config,
            acoustic,
        )

        self.assertIs(built_model, model.return_value)
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
