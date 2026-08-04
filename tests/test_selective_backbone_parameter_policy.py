from __future__ import annotations

import unittest
from types import SimpleNamespace

from anytrain.lightning import apply_parameter_trainability
from peft import LoraConfig, inject_adapter_in_model
from torch import nn

from speech_to_speech.model.ctc import CTCConfig, CTCRouteConfig
from speech_to_speech.training.parameter_policy import (
    PARAMETER_POLICY_SPECS,
    ParameterGroup,
    ParameterPolicyName,
    ParameterPolicyTrainability,
    parameter_group,
)


class _ProjectionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, name, nn.Linear(2, 2, bias=False))
        self.mlp = nn.Module()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self.mlp, name, nn.Linear(2, 2, bias=False))


class _BackboneConfig(SimpleNamespace):
    def get(self, name: str, default: object = None) -> object:
        return getattr(self, name, default)


class _SelectiveBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _BackboneConfig(num_hidden_layers=3)
        self.embed_tokens = nn.Embedding(4, 2)
        self.layers = nn.ModuleList(_ProjectionBlock() for _ in range(3))
        self.mimo_layers = nn.ModuleList(_ProjectionBlock() for _ in range(3))
        self.norm = nn.LayerNorm(2)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(3))

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


class _SelectiveModel(nn.Module):
    def __init__(self, ctc: CTCConfig | None = None) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            ctc=ctc
            or CTCConfig(
                source=CTCRouteConfig(weight=1.0),
                target=CTCRouteConfig(weight=1.0),
            )
        )
        self.backbone = _SelectiveBackbone()
        self.ctc_decoders = nn.ModuleDict(
            {
                "source": nn.Linear(2, 2, bias=False),
                "target": nn.Linear(2, 2, bias=False),
            }
        )


class SelectiveBackboneParameterPolicyTest(unittest.TestCase):
    def test_model_parameters_use_v3_ownership_paths(self):
        self.assertIs(
            parameter_group("backbone.embed_tokens.weight"),
            ParameterGroup.BACKBONE,
        )
        self.assertIs(
            parameter_group("tokens.audio_embedding.weight"),
            ParameterGroup.SEMANTIC_AUDIO_EMBEDDING,
        )
        self.assertIs(
            parameter_group("tokens.audio_projection.module.weight"),
            ParameterGroup.SEMANTIC_AUDIO_ADAPTER,
        )
        self.assertIs(
            parameter_group("tokens.audio_head.projection.weight"),
            ParameterGroup.AUDIO_OUTPUT,
        )
        self.assertIs(
            parameter_group("source_audio_encoder.projection.weight"),
            ParameterGroup.AUDIO_INPUT_ADAPTER,
        )
        for route in ("source", "target"):
            with self.subTest(route=route):
                self.assertIs(
                    parameter_group(f"ctc_decoders.{route}.projection.weight"),
                    ParameterGroup.ALIGNMENT_DECODER,
                )

        for legacy in (
            "token_embedding.audio_embedding.weight",
            "token_embedding.audio_projection.module.weight",
            "token_embedding.embeddings.text.weight",
            "token_embedding.embeddings.audio.weight",
            "token_embedding.adapters.audio.weight",
            "audio_input_adapter.projection.weight",
            "audio_output_adapter.projection.weight",
        ):
            with self.subTest(legacy=legacy):
                with self.assertRaisesRegex(ValueError, "legacy model ownership"):
                    parameter_group(legacy)

        for nonstandard in (
            "body.layers.0.weight",
            "text_embedding.weight",
            "text_head.weight",
            "audio_embedding.weight",
            "audio_feature_projection.weight",
            "audio_head.weight",
            "tokens.text_embedding.weight",
            "ctc_decoders.other.projection.weight",
        ):
            with self.subTest(nonstandard=nonstandard):
                with self.assertRaisesRegex(ValueError, "does not belong"):
                    parameter_group(nonstandard)

    def test_top_third_accepts_direct_body_paths_without_broadening_matching(self):
        model = _SelectiveModel()

        apply_parameter_trainability(
            model,
            ParameterPolicyTrainability(
                PARAMETER_POLICY_SPECS[
                    ParameterPolicyName.SPEECH_INTERFACE_TOP_THIRD
                ]
            ),
        )
        parameters = dict(model.named_parameters())

        self.assertFalse(parameters["backbone.embed_tokens.weight"].requires_grad)
        self.assertFalse(parameters["backbone.layers.0.self_attn.q_proj.weight"].requires_grad)
        self.assertFalse(parameters["backbone.layers.1.self_attn.q_proj.weight"].requires_grad)
        self.assertTrue(parameters["backbone.layers.2.self_attn.q_proj.weight"].requires_grad)
        self.assertFalse(
            parameters["backbone.mimo_layers.0.self_attn.q_proj.weight"].requires_grad
        )
        self.assertFalse(
            parameters["backbone.mimo_layers.1.self_attn.q_proj.weight"].requires_grad
        )
        self.assertTrue(
            parameters["backbone.mimo_layers.2.self_attn.q_proj.weight"].requires_grad
        )
        self.assertTrue(parameters["backbone.norm.weight"].requires_grad)
        self.assertFalse(parameters["backbone.encoder.layers.2.weight"].requires_grad)
        self.assertTrue(parameters["ctc_decoders.source.weight"].requires_grad)
        self.assertTrue(parameters["ctc_decoders.target.weight"].requires_grad)

    def test_alignment_decoders_follow_semantic_not_acoustic_policy(self):
        for policy, expected in (
            (ParameterPolicyName.FULL, True),
            (ParameterPolicyName.LORA, True),
            (ParameterPolicyName.SPEECH_INTERFACE, True),
            (ParameterPolicyName.SEMANTIC_ONLY, True),
            (ParameterPolicyName.ACOUSTIC_ONLY, False),
            (ParameterPolicyName.SPEECH_INTERFACE_TOP_THIRD, True),
        ):
            with self.subTest(policy=policy):
                model = _SelectiveModel()
                apply_parameter_trainability(
                    model,
                    ParameterPolicyTrainability(PARAMETER_POLICY_SPECS[policy]),
                )

                self.assertIs(
                    model.ctc_decoders["source"].weight.requires_grad,
                    expected,
                )
                self.assertIs(
                    model.ctc_decoders["target"].weight.requires_grad,
                    expected,
                )

    def test_inactive_alignment_route_is_structurally_frozen(self):
        model = _SelectiveModel(
            CTCConfig(
                source=CTCRouteConfig(weight=1.0),
                target=CTCRouteConfig(weight=0.0),
            )
        )

        apply_parameter_trainability(
            model,
            ParameterPolicyTrainability(
                PARAMETER_POLICY_SPECS[ParameterPolicyName.FULL]
            ),
        )

        self.assertTrue(model.ctc_decoders["source"].weight.requires_grad)
        self.assertFalse(model.ctc_decoders["target"].weight.requires_grad)

    def test_full_keeps_text_embedding_structurally_frozen(self) -> None:
        model = _SelectiveModel()
        self.assertTrue(model.backbone.embed_tokens.weight.requires_grad)

        apply_parameter_trainability(
            model,
            ParameterPolicyTrainability(
                PARAMETER_POLICY_SPECS[ParameterPolicyName.FULL]
            ),
        )

        self.assertFalse(model.backbone.embed_tokens.weight.requires_grad)
        self.assertTrue(model.backbone.layers[0].self_attn.q_proj.weight.requires_grad)

    def test_lora_targets_main_and_mimo_layers(self):
        model = _SelectiveModel()
        inject_adapter_in_model(
            LoraConfig(
                r=1,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
            model.backbone,
            adapter_name="speech",
        )

        apply_parameter_trainability(
            model,
            ParameterPolicyTrainability(
                PARAMETER_POLICY_SPECS[ParameterPolicyName.LORA]
            ),
        )
        parameters = dict(model.named_parameters())

        for branch in ("layers", "mimo_layers"):
            for projection_group, projection_names in (
                ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                ("mlp", ("gate_proj", "up_proj", "down_proj")),
            ):
                for projection in projection_names:
                    prefix = f"backbone.{branch}.0.{projection_group}.{projection}"
                    self.assertFalse(
                        parameters[f"{prefix}.base_layer.weight"].requires_grad
                    )
                    self.assertTrue(
                        parameters[f"{prefix}.lora_A.speech.weight"].requires_grad
                    )
                    self.assertTrue(
                        parameters[f"{prefix}.lora_B.speech.weight"].requires_grad
                    )

        adapter_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if parameter_group(name) is ParameterGroup.BACKBONE_ADAPTER
        )
        self.assertEqual(adapter_parameters, 168)


if __name__ == "__main__":
    unittest.main()
