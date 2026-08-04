from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from speech_to_speech.model.ctc import (
    CTCDecoderConfig,
    CTCDecoderRoutes,
    CTCDecoderRoutesConfig,
    CTCDecoderType,
    CTCRoute,
)


class CTCDecoderTest(unittest.TestCase):
    def test_source_and_target_have_independent_realized_configs(self) -> None:
        routes = CTCDecoderRoutes(
            CTCDecoderRoutesConfig(
                source=CTCDecoderConfig(
                    type=CTCDecoderType.LINEAR,
                    backbone_readout="hidden_states[2]",
                    pool_factor=2,
                ),
                target=CTCDecoderConfig(
                    type=CTCDecoderType.TRANSFORMER,
                    pool_factor=4,
                    layers=1,
                    heads=2,
                    ffn_ratio=2.0,
                ),
            ),
            hidden_size=4,
        )

        self.assertIs(routes.source.config.type, CTCDecoderType.LINEAR)
        self.assertFalse(routes.source.causal)
        self.assertEqual(routes.source.config.backbone_readout, "hidden_states[2]")
        self.assertEqual(routes.source.config.pool_factor, 2)
        self.assertIs(routes.target.config.type, CTCDecoderType.TRANSFORMER)
        self.assertTrue(routes.target.causal)
        self.assertEqual(routes.target.config.pool_factor, 4)
        self.assertTrue(
            routes.requires_hidden_states(frozenset({CTCRoute.SOURCE}))
        )
        self.assertFalse(
            routes.requires_hidden_states(frozenset({CTCRoute.TARGET}))
        )

    def test_identity_decoder_applies_masked_mean_pooling(self) -> None:
        routes = CTCDecoderRoutes(
            CTCDecoderRoutesConfig(
                source=CTCDecoderConfig(pool_factor=2),
            ),
            hidden_size=1,
        )
        values = torch.tensor([[[1.0], [3.0], [5.0], [100.0], [9.0]]])
        mask = torch.tensor([[True, True, True, False, True]])

        pooled, pooled_mask = routes(CTCRoute.SOURCE, values, mask)

        torch.testing.assert_close(
            pooled,
            torch.tensor([[[2.0], [5.0], [9.0]]]),
        )
        self.assertTrue(
            torch.equal(pooled_mask, torch.tensor([[True, True, True]]))
        )

    def test_target_transformer_is_causal_but_source_is_bidirectional(self) -> None:
        torch.manual_seed(7)
        decoder = CTCDecoderConfig(
            type=CTCDecoderType.TRANSFORMER,
            layers=1,
            heads=2,
            ffn_ratio=2.0,
            dropout=0.0,
        )
        routes = CTCDecoderRoutes(
            CTCDecoderRoutesConfig(
                source=decoder,
                target=decoder,
            ),
            hidden_size=4,
        ).eval()
        first = torch.randn(1, 4, 4)
        changed = first.clone()
        changed[:, -1] += 20
        mask = torch.ones(1, 4, dtype=torch.bool)

        source_first, _ = routes(CTCRoute.SOURCE, first, mask)
        source_changed, _ = routes(CTCRoute.SOURCE, changed, mask)
        target_first, _ = routes(CTCRoute.TARGET, first, mask)
        target_changed, _ = routes(CTCRoute.TARGET, changed, mask)

        self.assertGreater(
            float(
                (source_first[:, :-1] - source_changed[:, :-1])
                .abs()
                .max()
                .detach()
            ),
            1e-5,
        )
        torch.testing.assert_close(
            target_first[:, :-1],
            target_changed[:, :-1],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_trainable_decoder_receives_gradient_before_frozen_text_head(self) -> None:
        routes = CTCDecoderRoutes(
            CTCDecoderRoutesConfig(
                source=CTCDecoderConfig(type=CTCDecoderType.LINEAR),
            ),
            hidden_size=4,
        )
        text_head = nn.Embedding(6, 4)
        text_head.requires_grad_(False)
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        mask = torch.ones(2, 3, dtype=torch.bool)

        decoded, _ = routes(CTCRoute.SOURCE, hidden, mask)
        F.linear(decoded, text_head.weight).square().mean().backward()

        projection = routes.source.decoder
        self.assertIsInstance(projection, nn.Linear)
        assert isinstance(projection, nn.Linear)
        self.assertIsNotNone(projection.weight.grad)
        self.assertIsNotNone(hidden.grad)
        self.assertIsNone(text_head.weight.grad)

    def test_decoder_config_rejects_invalid_pooling_and_transformer_shape(self) -> None:
        with self.assertRaises(ValueError):
            CTCDecoderConfig(pool_factor=0)
        with self.assertRaises(TypeError):
            CTCDecoderConfig(pool_factor=True)
        with self.assertRaises(ValueError):
            CTCDecoderRoutes(
                CTCDecoderRoutesConfig(
                    source=CTCDecoderConfig(
                        type=CTCDecoderType.TRANSFORMER,
                        heads=3,
                    )
                ),
                hidden_size=4,
            )


if __name__ == "__main__":
    unittest.main()
