from __future__ import annotations

import unittest

import torch
from torch import nn

from speech_to_speech.loss.ctc import CTCAlignmentLoss, CTCConfig


class _Readout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor([[-1.0], [0.0], [1.0], [2.0]]),
            requires_grad=False,
        )
        self.inputs: list[torch.Tensor] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.inputs.append(hidden_states.detach().clone())
        return torch.nn.functional.linear(hidden_states, self.weight)


class CTCAlignmentLossTest(unittest.TestCase):
    def test_source_reads_current_slots_and_target_reads_causal_predecessors(self):
        hidden = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
        hidden.requires_grad_()
        readout = _Readout()
        loss = CTCAlignmentLoss(
            blank_token_id=0,
            config=CTCConfig(source_weight=1.0, target_weight=1.0),
        )

        item = loss(
            hidden,
            source={
                "token_positions": torch.tensor([[1, 3]]),
                "text_token_ids": torch.tensor([[1]]),
            },
            target={
                "token_positions": torch.tensor([[1, 3]]),
                "text_token_ids": torch.tensor([[2]]),
            },
            text_readout=readout,
        )

        self.assertEqual(len(readout.inputs), 2)
        self.assertTrue(
            torch.equal(readout.inputs[0], torch.tensor([[[1.0], [3.0]]]))
        )
        self.assertTrue(
            torch.equal(readout.inputs[1], torch.tensor([[[0.0], [2.0]]]))
        )
        self.assertTrue(torch.isfinite(item.loss).all())
        self.assertIsNotNone(item.details)
        assert item.details is not None
        self.assertEqual(item.details["source_tokens"].tolist(), [1.0])
        self.assertEqual(item.details["target_tokens"].tolist(), [1.0])
        self.assertEqual(item.details["sequences"].tolist(), [1.0])

        item.loss.sum().backward()

        self.assertIsNotNone(hidden.grad)
        assert hidden.grad is not None
        self.assertTrue(hidden.grad[0, :4].ne(0).all())
        self.assertEqual(float(hidden.grad[0, 4].item()), 0.0)
        self.assertIsNone(readout.weight.grad)

    def test_disabled_route_does_not_require_a_target(self):
        hidden = torch.randn(2, 3, 2, requires_grad=True)
        readout = nn.Linear(2, 4, bias=False)
        loss = CTCAlignmentLoss(
            blank_token_id=0,
            config=CTCConfig(source_weight=1.0, target_weight=0.0),
        )

        item = loss(
            hidden,
            source={
                "token_positions": torch.tensor([[0, 1], [-1, -1]]),
                "text_token_ids": torch.tensor([[1], [-1]]),
            },
            target=None,
            text_readout=readout,
        )

        self.assertIsNotNone(item.details)
        assert item.details is not None
        self.assertEqual(item.details["sequences"].tolist(), [1.0, 0.0])
        self.assertEqual(item.details["target_tokens"].tolist(), [0.0, 0.0])
        self.assertEqual(float(item.loss[1].item()), 0.0)

    def test_blank_token_is_rejected_as_a_transcript_label(self):
        loss = CTCAlignmentLoss(
            blank_token_id=0,
            config=CTCConfig(source_weight=1.0),
        )

        with self.assertRaisesRegex(ValueError, "must not contain the blank"):
            loss(
                torch.randn(1, 2, 3),
                source={
                    "token_positions": torch.tensor([[0, 1]]),
                    "text_token_ids": torch.tensor([[0]]),
                },
                target=None,
                text_readout=nn.Linear(3, 4, bias=False),
            )


if __name__ == "__main__":
    unittest.main()
