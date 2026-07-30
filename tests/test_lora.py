from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from speech_to_speech.model.lora import LoraConfig, inject
from speech_to_speech.stage import (
    PARAMETER_POLICY_SPECS,
    ParameterGroup,
    ParameterPolicyName,
    apply_parameter_policy,
    parameter_group,
)


class _AdapterProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_layer = nn.Linear(2, 2, bias=False)
        self.lora_A = nn.ModuleDict(
            {"speech": nn.Linear(2, 1, bias=False)}
        )
        self.lora_B = nn.ModuleDict(
            {"speech": nn.Linear(1, 2, bias=False)}
        )


class _LoraStageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.model = nn.Module()
        layer = nn.Module()
        layer.projection = _AdapterProjection()
        self.backbone.model.layers = nn.ModuleList([layer])
        self.semantic_audio_embedding = nn.Embedding(2, 2)
        self.semantic_audio_adapter = nn.Linear(2, 2)
        self.audio_output_adapter = nn.Linear(2, 2)
        self.acoustic_decoder = nn.Linear(2, 2)


class LoraTest(unittest.TestCase):
    def test_disabled_injection_does_not_import_peft(self):
        backbone = nn.Linear(2, 2)

        with patch(
            "speech_to_speech.model.lora._peft",
            side_effect=AssertionError("PEFT must remain optional"),
        ):
            adapted = inject(backbone, LoraConfig())

        self.assertIs(adapted, backbone)

    def test_injection_uses_supported_peft_api_and_mixed_precision_cast(self):
        backbone = nn.Linear(2, 2).to(dtype=torch.bfloat16)
        peft_config = object()
        peft = SimpleNamespace(
            LoraConfig=Mock(return_value=peft_config),
            inject_adapter_in_model=Mock(return_value=backbone),
            cast_mixed_precision_params=Mock(),
        )
        config = LoraConfig(
            enabled=True,
            rank=8,
            alpha=16,
            dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            use_rslora=True,
        )

        with patch("speech_to_speech.model.lora._peft", return_value=peft):
            adapted = inject(backbone, config)

        self.assertIs(adapted, backbone)
        peft.LoraConfig.assert_called_once_with(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            use_rslora=True,
        )
        peft.inject_adapter_in_model.assert_called_once_with(
            peft_config,
            backbone,
            adapter_name="speech",
        )
        peft.cast_mixed_precision_params.assert_called_once_with(
            backbone,
            torch.bfloat16,
        )

    def test_injection_requires_in_place_peft_result(self):
        backbone = nn.Linear(2, 2)
        peft = SimpleNamespace(
            LoraConfig=Mock(return_value=object()),
            inject_adapter_in_model=Mock(return_value=nn.Linear(2, 2)),
            cast_mixed_precision_params=Mock(),
        )

        with (
            patch("speech_to_speech.model.lora._peft", return_value=peft),
            self.assertRaisesRegex(RuntimeError, "preserve the backbone object"),
        ):
            inject(backbone, LoraConfig(enabled=True))

    def test_missing_peft_dependency_is_explicit(self):
        error = ModuleNotFoundError("No module named 'peft'", name="peft")

        with (
            patch(
                "speech_to_speech.model.lora.import_module",
                side_effect=error,
            ),
            self.assertRaisesRegex(RuntimeError, "install peft and accelerate"),
        ):
            inject(nn.Linear(2, 2), LoraConfig(enabled=True))

    def test_incompatible_peft_api_is_explicit(self):
        with (
            patch(
                "speech_to_speech.model.lora.import_module",
                return_value=SimpleNamespace(LoraConfig=Mock()),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "inject_adapter_in_model, cast_mixed_precision_params",
            ),
        ):
            inject(nn.Linear(2, 2), LoraConfig(enabled=True))

    def test_lora_policy_trains_only_backbone_adapters_and_speech_modules(self):
        model = _LoraStageModel()

        counts = apply_parameter_policy(
            model,
            PARAMETER_POLICY_SPECS[ParameterPolicyName.LORA],
        )
        parameters = dict(model.named_parameters())

        base = "backbone.model.layers.0.projection.base_layer.weight"
        adapter_a = "backbone.model.layers.0.projection.lora_A.speech.weight"
        adapter_b = "backbone.model.layers.0.projection.lora_B.speech.weight"
        self.assertIs(parameter_group(base), ParameterGroup.BACKBONE)
        self.assertIs(parameter_group(adapter_a), ParameterGroup.BACKBONE_ADAPTER)
        self.assertIs(parameter_group(adapter_b), ParameterGroup.BACKBONE_ADAPTER)
        self.assertFalse(parameters[base].requires_grad)
        self.assertTrue(parameters[adapter_a].requires_grad)
        self.assertTrue(parameters[adapter_b].requires_grad)
        self.assertTrue(model.semantic_audio_embedding.weight.requires_grad)
        self.assertTrue(model.acoustic_decoder.weight.requires_grad)
        self.assertEqual(counts[ParameterGroup.BACKBONE_ADAPTER], 4)


if __name__ == "__main__":
    unittest.main()
