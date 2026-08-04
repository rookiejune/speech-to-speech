from __future__ import annotations

import unittest
from types import SimpleNamespace

from peft import LoraConfig, inject_adapter_in_model
from torch import nn

from speech_to_speech.parameter_policy import (
    PARAMETER_POLICY_SPECS,
    ParameterGroup,
    ParameterPolicyName,
    apply_parameter_policy,
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
        self.layers = nn.ModuleList(_ProjectionBlock() for _ in range(3))
        self.mimo_layers = nn.ModuleList(_ProjectionBlock() for _ in range(3))
        self.norm = nn.LayerNorm(2)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(3))


class _SelectiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _SelectiveBackbone()


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
        ):
            with self.subTest(nonstandard=nonstandard):
                with self.assertRaisesRegex(ValueError, "does not belong"):
                    parameter_group(nonstandard)

    def test_top_third_accepts_direct_body_paths_without_broadening_matching(self):
        model = _SelectiveModel()

        apply_parameter_policy(
            model,
            PARAMETER_POLICY_SPECS[
                ParameterPolicyName.SPEECH_INTERFACE_TOP_THIRD
            ],
        )
        parameters = dict(model.named_parameters())

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

        counts = apply_parameter_policy(
            model,
            PARAMETER_POLICY_SPECS[ParameterPolicyName.LORA],
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

        self.assertEqual(counts[ParameterGroup.BACKBONE_ADAPTER], 168)


if __name__ == "__main__":
    unittest.main()
