from __future__ import annotations

import unittest

from peft import LoraConfig, inject_adapter_in_model
from torch import nn

from speech_to_speech.parameter_policy import (
    PARAMETER_POLICY_SPECS,
    ParameterGroup,
    ParameterPolicyName,
    apply_parameter_policy,
    parameter_group,
)


class _Tokens(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.audio_embedding = nn.Embedding(2, 2)
        self.audio_projection = nn.Linear(2, 2)
        self.audio_head = nn.Linear(2, 2)


class _LoraStageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        layer = nn.Module()
        layer.projection = nn.Linear(2, 2, bias=False)
        self.backbone.layers = nn.ModuleList([layer])
        inject_adapter_in_model(
            LoraConfig(r=1, target_modules=["projection"]),
            self.backbone,
            adapter_name="speech",
        )
        self.tokens = _Tokens()
        self.source_audio_encoder = nn.Linear(2, 2)
        self.acoustic_decoder = nn.Linear(2, 2)


class LoraTest(unittest.TestCase):
    def test_lora_policy_trains_only_backbone_adapters_and_speech_modules(self):
        model = _LoraStageModel()

        counts = apply_parameter_policy(
            model,
            PARAMETER_POLICY_SPECS[ParameterPolicyName.LORA],
        )
        parameters = dict(model.named_parameters())

        base = "backbone.layers.0.projection.base_layer.weight"
        adapter_a = "backbone.layers.0.projection.lora_A.speech.weight"
        adapter_b = "backbone.layers.0.projection.lora_B.speech.weight"
        self.assertIs(parameter_group(base), ParameterGroup.BACKBONE)
        self.assertIs(parameter_group(adapter_a), ParameterGroup.BACKBONE_ADAPTER)
        self.assertIs(parameter_group(adapter_b), ParameterGroup.BACKBONE_ADAPTER)
        self.assertFalse(parameters[base].requires_grad)
        self.assertTrue(parameters[adapter_a].requires_grad)
        self.assertTrue(parameters[adapter_b].requires_grad)
        self.assertTrue(model.tokens.audio_embedding.weight.requires_grad)
        self.assertTrue(model.tokens.audio_head.weight.requires_grad)
        self.assertTrue(model.source_audio_encoder.weight.requires_grad)
        self.assertTrue(model.acoustic_decoder.weight.requires_grad)
        self.assertEqual(counts[ParameterGroup.BACKBONE_ADAPTER], 4)


if __name__ == "__main__":
    unittest.main()
