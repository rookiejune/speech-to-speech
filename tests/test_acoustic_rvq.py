from __future__ import annotations

import unittest

import torch

from speech_to_speech.loss import CausalAcousticLoss
from speech_to_speech.model.acoustic import AcousticRVQDecoder


class AcousticRVQTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = AcousticRVQDecoder(
            condition_dim=6,
            codebooks=3,
            codebook_size=(5, 6, 7),
            hidden_dim=8,
            layers=2,
            heads=2,
            ffn_ratio=2,
        ).eval()

    def test_shapes_and_backward(self):
        condition = torch.randn(2, 4, 6, requires_grad=True)
        labels = torch.stack(
            (
                torch.randint(0, 5, (2, 4)),
                torch.randint(0, 6, (2, 4)),
                torch.randint(0, 7, (2, 4)),
            ),
            dim=-1,
        )

        logits = self.model(condition, labels)
        sum(value.square().mean() for value in logits).backward()

        self.assertEqual([value.shape for value in logits], [(2, 4, 5), (2, 4, 6), (2, 4, 7)])
        self.assertIsNotNone(condition.grad)

    def test_future_codebook_does_not_change_previous_logits(self):
        condition = torch.randn(1, 2, 6)
        labels = torch.tensor([[[1, 2, 3], [2, 3, 4]]])
        changed = labels.clone()
        changed[..., 1] = (changed[..., 1] + 1) % 6

        baseline = self.model(condition, labels)
        future_changed = self.model(condition, changed)

        torch.testing.assert_close(baseline[0], future_changed[0])
        torch.testing.assert_close(baseline[1], future_changed[1])
        self.assertFalse(torch.equal(baseline[2], future_changed[2]))

    def test_default_depth_is_eight_qwen_layers(self):
        model = AcousticRVQDecoder(
            condition_dim=8,
            codebooks=2,
            codebook_size=4,
            heads=2,
        )

        self.assertEqual(len(model.decoder.layers), 8)

    def test_causal_loss_ignores_padding_frames(self):
        labels = torch.tensor([[[1, 2], [2, 1], [-1, -1]]])
        mask = torch.tensor([[True, True, False]])
        logits = (
            torch.randn(1, 3, 4),
            torch.randn(1, 3, 3),
        )
        changed = tuple(value.clone() for value in logits)
        for value in changed:
            value[:, -1] = 1000

        baseline = CausalAcousticLoss()(logits, labels, mask)
        padded_changed = CausalAcousticLoss()(changed, labels, mask)

        torch.testing.assert_close(baseline.loss, padded_changed.loss)


if __name__ == "__main__":
    unittest.main()
