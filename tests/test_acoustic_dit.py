from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import Tensor, nn

from speech_to_speech.loss import RepaLoss, WavLMTeacher
from speech_to_speech.model.acoustic import AcousticDiT


class AcousticDiTTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = AcousticDiT(
            condition_dim=6,
            latent_dim=4,
            hidden_dim=8,
            layers=2,
            heads=2,
            ffn_ratio=2,
        )

    def test_shapes_and_backward(self):
        x_t = torch.randn(2, 5, 4, requires_grad=True)
        condition = torch.randn(2, 5, 6, requires_grad=True)
        mask = torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        )

        velocity, representation = self.model.forward_with_features(
            x_t,
            torch.tensor([0.2, 0.8]),
            condition=condition,
            mask=mask,
        )
        (velocity.square().mean() + representation.square().mean()).backward()

        self.assertEqual(velocity.shape, x_t.shape)
        self.assertEqual(representation.shape, condition.shape)
        self.assertTrue(torch.equal(velocity[1, 3:], torch.zeros(2, 4)))
        self.assertIsNotNone(x_t.grad)
        self.assertIsNotNone(condition.grad)

    def test_padding_does_not_change_valid_frames(self):
        x_t = torch.randn(1, 4, 4)
        condition = torch.randn(1, 4, 6)
        mask = torch.tensor([[True, True, False, False]])
        changed_x = x_t.clone()
        changed_condition = condition.clone()
        changed_x[:, 2:] = 1000
        changed_condition[:, 2:] = -1000

        output = self.model(x_t, torch.tensor([0.5]), condition=condition, mask=mask)
        changed = self.model(
            changed_x,
            torch.tensor([0.5]),
            condition=changed_condition,
            mask=mask,
        )

        torch.testing.assert_close(output[:, :2], changed[:, :2])

    def test_film_condition_and_self_attention_affect_output(self):
        for block in self.model.blocks:
            torch.nn.init.normal_(block.film.weight, std=0.1)
        x_t = torch.randn(1, 3, 4)
        condition = torch.randn(1, 3, 6)
        time = torch.tensor([0.5])
        baseline = self.model(x_t, time, condition=condition)

        changed_condition = condition.clone()
        changed_condition[:, 1] += 1
        film_changed = self.model(x_t, time, condition=changed_condition)
        changed_x = x_t.clone()
        changed_x[:, 1] += 1
        attention_changed = self.model(changed_x, time, condition=condition)

        self.assertFalse(torch.equal(baseline[:, 1], film_changed[:, 1]))
        self.assertFalse(torch.equal(baseline[:, 0], attention_changed[:, 0]))

    def test_repa_detaches_teacher(self):
        representation = torch.randn(2, 3, 5, requires_grad=True)
        condition = torch.randn(2, 3, 5, requires_grad=True)
        item = RepaLoss()(
            representation,
            condition,
            torch.tensor([[True, True, True], [True, False, False]]),
        )

        item.loss.mean().backward()

        self.assertIsNotNone(representation.grad)
        self.assertIsNone(condition.grad)

    def test_eight_layer_dit_exposes_block_four_for_repa(self):
        model = AcousticDiT(
            condition_dim=6,
            latent_dim=4,
            hidden_dim=8,
            layers=8,
            heads=2,
            ffn_ratio=2,
            repa_dim=5,
            repa_layer=4,
        )
        captured: list[Tensor] = []
        handle = model.blocks[3].register_forward_hook(
            lambda module, inputs, output: captured.append(output)
        )
        try:
            _, representation = model.forward_with_features(
                torch.randn(1, 3, 4),
                torch.tensor([0.5]),
                condition=torch.randn(1, 3, 6),
            )
        finally:
            handle.remove()

        torch.testing.assert_close(representation, model.repa(captured[0]))

    def test_wavlm_teacher_uses_layer_nine_and_aligns_frames(self):
        wavlm = _WavLM()
        with patch(
            "speech_to_speech.loss.repa.WavLMModel.from_pretrained",
            return_value=wavlm,
        ):
            teacher = WavLMTeacher(_Codec(), layer=9)
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])

        features = teacher(
            torch.zeros(2, 4, 1, dtype=torch.long),
            torch.zeros(2, 4, 1, dtype=torch.long),
            mask,
        )
        teacher.train()

        self.assertEqual(features.shape, (2, 4, 3))
        torch.testing.assert_close(features[0], torch.full((4, 3), 9.0))
        torch.testing.assert_close(features[1, :2], torch.full((2, 3), 9.0))
        self.assertTrue(torch.equal(features[1, 2:], torch.zeros(2, 3)))
        self.assertFalse(wavlm.training)


class _Codec:
    sample_rate = 16_000

    def decode(self, codes: Tensor) -> Tensor:
        length = codes.size(1) * 8
        return torch.arange(length, device=codes.device, dtype=torch.float32)[None]


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
