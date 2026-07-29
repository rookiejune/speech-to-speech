from __future__ import annotations

import unittest

import torch

from speech_to_speech.model.audio_input import (
    AudioInputAdapterConfig,
    AudioInputAdapterType,
    AudioInputTower,
    audio_input_options,
    create_audio_input_adapter,
)


class AudioInputTowerTest(unittest.TestCase):
    def test_mlp_preserves_shape_masks_padding_and_uses_fp32(self) -> None:
        tower = create_audio_input_adapter(
            AudioInputAdapterConfig(type=AudioInputAdapterType.MLP),
            in_features=3,
            out_features=6,
        )
        features = torch.randn(2, 4, 3, dtype=torch.bfloat16)
        mask = torch.tensor([[True, True, False, False], [True, False, True, False]])

        output = tower(features, mask)

        self.assertEqual(output.shape, (2, 4, 6))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(
            all(parameter.dtype is torch.float32 for parameter in tower.parameters())
        )
        self.assertTrue(torch.equal(output[~mask], torch.zeros_like(output[~mask])))
        self.assertTrue(torch.isfinite(output).all())

    def test_transformer_does_not_use_padding_as_context(self) -> None:
        torch.manual_seed(0)
        tower = create_audio_input_adapter(
            AudioInputAdapterConfig(
                type=AudioInputAdapterType.TRANSFORMER,
                layers=2,
                heads=2,
                ffn_ratio=2,
            ),
            in_features=3,
            out_features=8,
        )
        tower.eval()
        valid = torch.randn(1, 2, 3)
        padded = torch.cat([valid, torch.full((1, 2, 3), float("nan"))], dim=1)

        valid_output = tower(valid)
        padded_output = tower(
            padded,
            torch.tensor([[True, True, False, False]]),
        )

        self.assertTrue(torch.allclose(padded_output[:, :2], valid_output, atol=1e-6))
        self.assertTrue(torch.equal(padded_output[:, 2:], torch.zeros(1, 2, 8)))

    def test_transformer_all_padding_is_finite_and_zero(self) -> None:
        tower = create_audio_input_adapter(
            {
                "type": "transformer",
                "layers": 1,
                "heads": 2,
                "ffn_ratio": 2,
            },
            in_features=4,
            out_features=8,
        )
        output = tower(
            torch.randn(2, 3, 4),
            torch.zeros(2, 3, dtype=torch.bool),
        )

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_config_mapping_is_normalized(self) -> None:
        config = audio_input_options(
            {
                "type": "mlp",
                "layers": 3,
                "heads": 4,
                "ffn_ratio": 3,
                "dropout": 0.1,
            }
        )

        self.assertEqual(config.type, AudioInputAdapterType.MLP)
        self.assertEqual(config.layers, 3)
        self.assertEqual(config.heads, 4)
        self.assertEqual(config.ffn_ratio, 3)
        self.assertEqual(config.dropout, 0.1)

    def test_invalid_shapes_and_transformer_width_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            AudioInputTower(
                AudioInputAdapterConfig(type=AudioInputAdapterType.MLP),
                3,
                4,
            )(torch.randn(2, 3))
        with self.assertRaisesRegex(ValueError, "divisible"):
            create_audio_input_adapter(
                AudioInputAdapterConfig(type=AudioInputAdapterType.TRANSFORMER),
                in_features=3,
                out_features=7,
            )


if __name__ == "__main__":
    unittest.main()
