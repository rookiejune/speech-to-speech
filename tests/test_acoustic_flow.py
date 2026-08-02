from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from anytrain.loss import MaskedCosineAlignmentLoss
from torch import Tensor, nn

from semantic_acoustic_codec.loss.repa import WavLMTeacher
from speech_to_speech.model.acoustic import AcousticFlow


class AcousticFlowTest(unittest.TestCase):
    def test_sample_zeros_masked_frames(self):
        flow = AcousticFlow(2, 2, _FlowRuntime(), hidden_dim=2, layers=1, heads=1)
        mask = torch.tensor([[True, False]])

        output = flow.sample(
            torch.zeros(1, 2, 2),
            mask=mask,
            generator=torch.Generator().manual_seed(1),
        )

        self.assertTrue(torch.equal(output[~mask], torch.zeros(1, 2)))

    def test_sample_validates_mask(self):
        flow = AcousticFlow(2, 2, _FlowRuntime(), hidden_dim=2, layers=1, heads=1)
        condition = torch.zeros(1, 2, 2)

        with self.assertRaisesRegex(ValueError, "align"):
            flow.sample(condition, mask=torch.ones(1, 1, dtype=torch.bool))
        with self.assertRaisesRegex(TypeError, "boolean"):
            flow.sample(condition, mask=torch.ones(1, 2))

    def test_wraps_sac_feature_generator(self):
        from semantic_acoustic_codec.model import FMFeatureGenerator
        from semantic_acoustic_codec.model.dit import DiTDecoder

        flow = AcousticFlow(2, 2, _FlowRuntime(), hidden_dim=2, layers=1, heads=1)
        self.assertIsInstance(flow.generator, FMFeatureGenerator)
        self.assertIsInstance(flow.decoder, DiTDecoder)
        self.assertIs(flow.decoder, flow.generator.core)


class AcousticRepaLossTest(unittest.TestCase):
    def test_repa_detaches_teacher(self):
        representation = torch.randn(2, 3, 5, requires_grad=True)
        condition = torch.randn(2, 3, 5, requires_grad=True)
        item = MaskedCosineAlignmentLoss()(
            representation,
            condition,
            torch.tensor([[True, True, True], [True, False, False]]),
        )

        item.loss.mean().backward()

        self.assertIsNotNone(representation.grad)
        self.assertIsNone(condition.grad)

    def test_repa_ignores_nonfinite_padding_in_forward_and_backward(self):
        representation = torch.tensor(
            [[[1.0, 0.0], [float("nan"), float("inf")]]],
            requires_grad=True,
        )
        target = torch.tensor([[[1.0, 0.0], [float("nan"), float("inf")]]])

        item = MaskedCosineAlignmentLoss()(
            representation,
            target,
            torch.tensor([[True, False]]),
        )
        item.loss.mean().backward()

        self.assertTrue(torch.isfinite(item.loss).all())
        self.assertIsNotNone(representation.grad)
        gradient = representation.grad
        if gradient is None:
            self.fail("REPA representation gradient is unavailable")
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.equal(gradient[:, 1], torch.zeros_like(gradient[:, 1])))

    def test_wavlm_teacher_uses_layer_nine_and_preserves_prefix_padding_mask(self):
        wavlm = _WavLM()
        with patch(
            "transformers.WavLMModel.from_pretrained",
            return_value=wavlm,
        ):
            teacher = WavLMTeacher(_Codec(), layer=9)
        mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

        features = teacher(
            torch.zeros(2, 4, 1, dtype=torch.long),
            torch.zeros(2, 4, 1, dtype=torch.long),
            mask,
        )
        teacher.train()

        self.assertEqual(features.shape, (2, 4, 3))
        torch.testing.assert_close(features[mask], torch.full((5, 3), 9.0))
        self.assertTrue(torch.equal(features[~mask], torch.zeros(3, 3)))
        self.assertFalse(wavlm.training)

        with self.assertRaisesRegex(TypeError, "boolean"):
            teacher(
                torch.zeros(2, 4, 1, dtype=torch.long),
                torch.zeros(2, 4, 1, dtype=torch.long),
                mask.to(dtype=torch.long),
            )


class _Codec:
    sample_rate = 16_000

    def decode(self, codes: Tensor) -> Tensor:
        length = codes.size(1) * 8
        return torch.arange(length, device=codes.device, dtype=torch.float32)[None]


class _FlowRuntime:
    def sample(self, model: nn.Module, x_0: Tensor, **kwargs):
        del model, kwargs
        return SimpleNamespace(final=x_0)


class _WavLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            hidden_size=3,
            num_hidden_layers=12,
            conv_kernel=[2],
            conv_stride=[2],
        )

    def forward(
        self,
        inputs: Tensor,
        *,
        attention_mask: Tensor,
        output_hidden_states: bool,
    ):
        del attention_mask, output_hidden_states
        length = (inputs.size(1) - 2) // 2 + 1
        hidden_states = tuple(
            inputs.new_full((inputs.size(0), length, 3), float(layer))
            for layer in range(13)
        )
        return SimpleNamespace(hidden_states=hidden_states)


if __name__ == "__main__":
    unittest.main()
