from __future__ import annotations

import unittest

import torch
from torch import Tensor, nn

from speech_to_speech.loss import CTCConfig as ExportedCTCConfig
from speech_to_speech.loss.ctc import CTCAlignmentLoss, CTCConfig as ModuleCTCConfig
from speech_to_speech.model.ctc import CTCConfig, CTCRoute, CTCRouteConfig


class _Decode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor([[-1.0], [0.0], [1.0], [2.0]]),
            requires_grad=False,
        )
        self.calls: list[tuple[CTCRoute, Tensor, Tensor]] = []

    def forward(
        self,
        route: CTCRoute,
        hidden_states: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.calls.append(
            (route, hidden_states.detach().clone(), mask.detach().clone())
        )
        return torch.nn.functional.linear(hidden_states, self.weight), mask


class CTCAlignmentLossTest(unittest.TestCase):
    def test_config_remains_available_from_loss_exports(self):
        self.assertIs(ExportedCTCConfig, CTCConfig)
        self.assertIs(ModuleCTCConfig, CTCConfig)

    def test_routes_use_independent_hidden_states_and_causal_positions(self):
        reference = torch.zeros((1, 5, 1), dtype=torch.float32)
        source_hidden = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
        target_hidden = source_hidden.detach().clone() + 10
        source_hidden.requires_grad_()
        target_hidden.requires_grad_()
        decode = _Decode()
        loss = CTCAlignmentLoss(
            blank_token_id=0,
            config=CTCConfig(
                source=CTCRouteConfig(weight=1.0),
                target=CTCRouteConfig(weight=1.0),
            ),
        )

        item = loss(
            reference,
            source_hidden_states=source_hidden,
            target_hidden_states=target_hidden,
            source={
                "token_positions": torch.tensor([[1, 3]]),
                "text_token_ids": torch.tensor([[1]]),
            },
            target={
                "token_positions": torch.tensor([[1, 3]]),
                "text_token_ids": torch.tensor([[2]]),
            },
            decode=decode,
        )

        self.assertEqual(len(decode.calls), 2)
        source_route, source_states, source_mask = decode.calls[0]
        target_route, target_states, target_mask = decode.calls[1]
        self.assertIs(source_route, CTCRoute.SOURCE)
        self.assertIs(target_route, CTCRoute.TARGET)
        torch.testing.assert_close(source_states, torch.tensor([[[1.0], [3.0]]]))
        torch.testing.assert_close(target_states, torch.tensor([[[10.0], [12.0]]]))
        torch.testing.assert_close(source_mask, torch.ones((1, 2), dtype=torch.bool))
        torch.testing.assert_close(target_mask, torch.ones((1, 2), dtype=torch.bool))
        self.assertTrue(torch.isfinite(item.loss).all())
        self.assertIsNotNone(item.details)
        assert item.details is not None
        self.assertEqual(item.details["source_tokens"].tolist(), [1.0])
        self.assertEqual(item.details["target_tokens"].tolist(), [1.0])
        self.assertEqual(item.details["sequences"].tolist(), [1.0])

        item.loss.sum().backward()

        self.assertIsNotNone(source_hidden.grad)
        self.assertIsNotNone(target_hidden.grad)
        assert source_hidden.grad is not None
        assert target_hidden.grad is not None
        self.assertTrue(source_hidden.grad[0, [1, 3]].ne(0).all())
        self.assertTrue(target_hidden.grad[0, [0, 2]].ne(0).all())
        self.assertEqual(float(source_hidden.grad[0, [0, 2, 4]].abs().sum()), 0.0)
        self.assertEqual(float(target_hidden.grad[0, [1, 3, 4]].abs().sum()), 0.0)
        self.assertIsNone(decode.weight.grad)

    def test_disabled_route_does_not_decode_even_when_target_is_present(self):
        reference = torch.randn(2, 3, 2)
        source_hidden = torch.randn(2, 3, 2, requires_grad=True)
        decode_calls: list[CTCRoute] = []

        def decode(
            route: CTCRoute,
            hidden_states: Tensor,
            mask: Tensor,
        ) -> tuple[Tensor, Tensor]:
            decode_calls.append(route)
            return hidden_states.new_zeros((*hidden_states.shape[:2], 4)), mask

        loss = CTCAlignmentLoss(
            blank_token_id=0,
            config=CTCConfig(
                source=CTCRouteConfig(weight=1.0),
                target=CTCRouteConfig(weight=0.0),
            ),
        )

        item = loss(
            reference,
            source_hidden_states=source_hidden,
            target_hidden_states=None,
            source={
                "token_positions": torch.tensor([[0, 1], [-1, -1]]),
                "text_token_ids": torch.tensor([[1], [-1]]),
            },
            target={
                "token_positions": torch.tensor([[1, 2], [1, 2]]),
                "text_token_ids": torch.tensor([[2], [2]]),
            },
            decode=decode,
        )

        self.assertEqual(decode_calls, [CTCRoute.SOURCE])
        self.assertIsNotNone(item.details)
        assert item.details is not None
        self.assertEqual(item.details["sequences"].tolist(), [1.0, 0.0])
        self.assertEqual(item.details["target_tokens"].tolist(), [0.0, 0.0])
        self.assertEqual(float(item.loss[1].item()), 0.0)

    def test_decoder_mask_defines_pooled_step_count(self):
        reference = torch.zeros((1, 5, 2))
        hidden = torch.randn(1, 5, 2)

        def pooled_decode(
            route: CTCRoute,
            hidden_states: Tensor,
            mask: Tensor,
        ) -> tuple[Tensor, Tensor]:
            self.assertIs(route, CTCRoute.SOURCE)
            self.assertEqual(hidden_states.shape[:2], (1, 5))
            self.assertTrue(mask.all())
            return hidden_states.new_zeros((1, 3, 4)), torch.ones(
                (1, 3), dtype=torch.bool
            )

        item = CTCAlignmentLoss(
            0,
            CTCConfig(source=CTCRouteConfig(weight=1.0)),
        )(
            reference,
            source_hidden_states=hidden,
            target_hidden_states=None,
            source={
                "token_positions": torch.tensor([[0, 1, 2, 3, 4]]),
                "text_token_ids": torch.tensor([[1, 2]]),
            },
            target=None,
            decode=pooled_decode,
        )

        self.assertIsNotNone(item.details)
        assert item.details is not None
        self.assertEqual(item.details["source_steps"].tolist(), [3.0])
        self.assertEqual(item.details["source_tokens"].tolist(), [2.0])

    def test_pooling_that_is_too_short_for_repeated_labels_is_rejected(self):
        def too_short_decode(
            route: CTCRoute,
            hidden_states: Tensor,
            mask: Tensor,
        ) -> tuple[Tensor, Tensor]:
            del route, mask
            return hidden_states.new_zeros((1, 2, 4)), torch.ones(
                (1, 2), dtype=torch.bool
            )

        with self.assertRaisesRegex(ValueError, "pooling leaves too few steps"):
            CTCAlignmentLoss(
                0,
                CTCConfig(source=CTCRouteConfig(weight=1.0)),
            )(
                torch.zeros((1, 4, 2)),
                source_hidden_states=torch.randn(1, 4, 2),
                target_hidden_states=None,
                source={
                    "token_positions": torch.tensor([[0, 1, 2, 3]]),
                    "text_token_ids": torch.tensor([[1, 1]]),
                },
                target=None,
                decode=too_short_decode,
            )

    def test_blank_token_is_rejected_as_a_transcript_label(self):
        loss = CTCAlignmentLoss(
            blank_token_id=0,
            config=CTCConfig(source=CTCRouteConfig(weight=1.0)),
        )

        with self.assertRaisesRegex(ValueError, "must not contain the blank"):
            loss(
                torch.randn(1, 2, 3),
                source_hidden_states=torch.randn(1, 2, 3),
                target_hidden_states=None,
                source={
                    "token_positions": torch.tensor([[0, 1]]),
                    "text_token_ids": torch.tensor([[0]]),
                },
                target=None,
                decode=lambda route, states, mask: (
                    states.new_zeros((1, 2, 4)),
                    mask,
                ),
            )


if __name__ == "__main__":
    unittest.main()
