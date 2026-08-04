from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from anydataset.types import Modality
from anytrain.module.idspace import Layout

from speech_to_speech.loss.token import TokenLoss
from speech_to_speech.model.embedding.fsq import FsqNeighbors


class FsqNeighborLossTest(unittest.TestCase):
    def test_audio_loss_mixes_hard_and_normalized_neighbor_nll(self) -> None:
        layout = Layout(text=(0, 3), audio=(3, 10))
        loss = TokenLoss(layout, audio_neighbor_smoothing=0.2)
        hidden = torch.zeros(1, 2, 4)
        labels = torch.tensor([[-100, 3]])
        logits = torch.tensor([[3.0, 1.0, -1.0, 0.5, 0.0, -0.5, -1.5]])

        output = loss(
            hidden,
            labels,
            Modality.AUDIO,
            lambda states, modality: logits.expand(states.size(0), -1),
            audio_neighbors=_neighbors,
        )

        log_probabilities = F.log_softmax(logits, dim=-1)
        hard = -log_probabilities[0, 0]
        smooth = -0.5 * (log_probabilities[0, 1] + log_probabilities[0, 2])
        expected = 0.8 * hard + 0.2 * smooth
        torch.testing.assert_close(output.loss, expected.reshape(1))

    def test_special_audio_rows_keep_hard_targets(self) -> None:
        layout = Layout(text=(0, 3), audio=(3, 10))
        loss = TokenLoss(layout, audio_neighbor_smoothing=0.2)
        hidden = torch.zeros(1, 2, 4)
        labels = torch.tensor([[-100, 9]])
        logits = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]])

        output = loss(
            hidden,
            labels,
            Modality.AUDIO,
            lambda states, modality: logits.expand(states.size(0), -1),
            audio_neighbors=_neighbors,
        )

        expected = F.cross_entropy(logits, torch.tensor([6]), reduction="none")
        torch.testing.assert_close(output.loss, expected)

    def test_enabled_smoothing_requires_fsq_neighbors(self) -> None:
        layout = Layout(text=(0, 3), audio=(3, 10))
        loss = TokenLoss(layout, audio_neighbor_smoothing=0.1)
        with self.assertRaisesRegex(ValueError, "factorized FSQ"):
            loss(
                torch.zeros(1, 2, 4),
                torch.tensor([[-100, 3]]),
                Modality.AUDIO,
                lambda states, modality: states.new_zeros(states.size(0), 7),
                audio_neighbors=lambda target: None,
            )

    def test_invalid_smoothing_is_rejected(self) -> None:
        layout = Layout(text=(0, 3), audio=(3, 10))
        for value in (-0.1, 1.0, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"in \[0, 1\)"):
                    TokenLoss(layout, audio_neighbor_smoothing=value)


def _neighbors(target: torch.Tensor) -> FsqNeighbors:
    token_ids = target.new_zeros((target.numel(), 2))
    valid = torch.zeros_like(token_ids, dtype=torch.bool)
    code = target.lt(6)
    token_ids[code, 0] = 1
    token_ids[code, 1] = 2
    valid[code] = True
    weights = valid.to(dtype=torch.float32) / valid.sum(dim=-1, keepdim=True).clamp_min(1)
    return FsqNeighbors(token_ids=token_ids, weights=weights, valid=valid)


if __name__ == "__main__":
    unittest.main()
