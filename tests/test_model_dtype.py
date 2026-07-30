from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast

import torch
from anytrain.module.idspace import Layout
from torch import Tensor, nn

from speech_to_speech.model import Config
from speech_to_speech.model.acoustic import DecoderConfig, FlowModel, RVQModel
from speech_to_speech.runtime.audio_tokenizer import NativeAudioTokenizer
from speech_to_speech.stage import (
    PARAMETER_POLICY_SPECS,
    ParameterGroup,
    ParameterPolicyName,
    apply_parameter_policy,
    parameter_group,
)


class ModelDtypeTest(unittest.TestCase):
    def test_speech_interface_uses_fp32_storage_with_bf16_backbone(self):
        model = _rvq_model()
        apply_parameter_policy(
            model,
            PARAMETER_POLICY_SPECS[ParameterPolicyName.SPEECH_INTERFACE],
        )

        backbone = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter_group(name) is ParameterGroup.BACKBONE
        ]
        speech = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter_group(name) is not ParameterGroup.BACKBONE
        ]

        self.assertTrue(backbone)
        self.assertTrue(speech)
        self.assertTrue(all(value.dtype is torch.bfloat16 for value in backbone))
        self.assertTrue(all(not value.requires_grad for value in backbone))
        self.assertTrue(all(value.dtype is torch.float32 for value in speech))
        self.assertTrue(any(value.requires_grad for value in speech))

    def test_rvq_boundaries_and_optimizer_state_remain_fp32(self):
        model = _rvq_model()
        apply_parameter_policy(
            model,
            PARAMETER_POLICY_SPECS[ParameterPolicyName.SPEECH_INTERFACE],
        )
        hidden = torch.randn(1, 2, 4, dtype=torch.bfloat16)
        positions = torch.tensor([[1, 2]])
        targets = torch.zeros(1, 2, 1, dtype=torch.long)

        semantic = model.semantic_audio_logits(hidden)
        acoustic = model.acoustic_logits(hidden, positions, targets)

        self.assertEqual(semantic.dtype, torch.float32)
        self.assertTrue(torch.isfinite(semantic).all())
        self.assertTrue(all(value.dtype is torch.float32 for value in acoustic))
        self.assertTrue(all(torch.isfinite(value).all() for value in acoustic))

        optimizer = torch.optim.AdamW(
            [value for value in model.parameters() if value.requires_grad],
            lr=2e-5,
        )
        loss = semantic.square().mean() + sum(
            value.square().mean() for value in acoustic
        )
        loss.backward()
        optimizer.step()

        states = [
            value
            for state in optimizer.state.values()
            for key, value in state.items()
            if key in {"exp_avg", "exp_avg_sq"}
        ]
        self.assertTrue(states)
        self.assertTrue(all(value.dtype is torch.float32 for value in states))

    def test_combined_vocabulary_logits_promote_fp32_audio_head(self):
        model = _rvq_model()
        hidden = torch.randn(1, 2, 4, dtype=torch.bfloat16)

        dense = model.token_logits(hidden)
        selected, _ = model.selected_logits(hidden, torch.tensor([0, 4, 8]))
        audio, _ = model.selected_logits(hidden, torch.tensor([4, 8]))

        self.assertEqual(dense.dtype, torch.float32)
        self.assertEqual(selected.dtype, torch.float32)
        self.assertEqual(audio.dtype, torch.float32)
        self.assertTrue(torch.isfinite(dense).all())
        self.assertTrue(torch.isfinite(selected).all())
        self.assertTrue(torch.isfinite(audio).all())

    def test_flow_inputs_follow_fp32_decoder_storage(self):
        model = _flow_model()
        hidden = torch.randn(1, 2, 4, dtype=torch.bfloat16)
        positions = torch.tensor([[1, 2]])
        codes = torch.zeros(1, 2, 1, dtype=torch.long)

        condition = model.target_frame_condition(hidden, positions)
        target = model.acoustic_target_latent(codes)

        self.assertTrue(
            all(
                value.dtype is torch.float32
                for value in model.acoustic_decoder.parameters()
            )
        )
        self.assertEqual(condition.dtype, torch.float32)
        self.assertEqual(target.dtype, torch.float32)

    def test_audio_embeddings_merge_into_bf16_backbone_input(self):
        model = _flow_model()
        token_labels = torch.tensor([[0, 4]])
        positions = torch.tensor([[0, 1]])

        condition = model.target_frame_label_condition(token_labels, positions)

        self.assertEqual(condition.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(condition).all())


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.input_embeddings = nn.Embedding(4, 4)
        self.output_embeddings = nn.Linear(4, 4, bias=False)
        self.to(dtype=torch.bfloat16)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.input_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.output_embeddings


class _Codec:
    sample_rate = 24_000
    frame_rate = 50.0
    semantic_feature_dim = 4
    codebook_sizes = (3, 3)
    semantic_codebook = torch.randn(3, 4, dtype=torch.bfloat16)
    acoustic_feature_dim = 4
    acoustic_codebook_sizes = (3,)

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        del sample_rate
        return audio.new_zeros(1, 2, dtype=torch.long)

    def decode(self, codes: Tensor) -> Tensor:
        return codes.new_zeros(1, dtype=torch.float32)

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        return torch.nn.functional.one_hot(
            acoustic_codes[..., 0],
            num_classes=self.acoustic_feature_dim,
        ).to(dtype=torch.bfloat16)

    def decode_features(
        self,
        semantic_codes: Tensor,
        acoustic_features: Tensor,
    ) -> Tensor:
        del semantic_codes
        return acoustic_features.new_zeros(1)


class _FlowRuntime:
    def sample(self, *args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(final=torch.zeros(1))


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(
        backbone=_Backbone(),
        codec=_Codec(),
        audio_tokenizer=NativeAudioTokenizer(vocab_size=3),
        layout=Layout(text=(0, 4), audio=(4, 10)),
        flow_matching=_FlowRuntime(),
    )


def _rvq_model() -> RVQModel:
    return RVQModel(
        Config(),
        runtime=cast(Any, _runtime()),
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
    )


def _flow_model() -> FlowModel:
    return FlowModel(
        Config(),
        runtime=cast(Any, _runtime()),
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
    )


if __name__ == "__main__":
    unittest.main()
