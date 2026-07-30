from __future__ import annotations

import unittest

import torch

from speech_to_speech.model.audio_output import (
    AudioOutputAdapter,
    AudioOutputAdapterConfig,
    AudioOutputAdapterType,
    audio_output_options,
    create_audio_output_adapter,
)


class AudioOutputAdapterTest(unittest.TestCase):
    def test_mlp_is_pointwise_and_uses_fp32(self) -> None:
        adapter = create_audio_output_adapter(
            AudioOutputAdapterConfig(type=AudioOutputAdapterType.MLP),
            in_features=4,
            out_features=6,
        )
        hidden = torch.randn(2, 3, 4, dtype=torch.bfloat16)

        output = adapter(hidden)

        self.assertEqual(output.shape, (2, 3, 6))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(
            all(parameter.dtype is torch.float32 for parameter in adapter.parameters())
        )
        self.assertTrue(torch.isfinite(output).all())

    def test_none_requires_matching_dimensions(self) -> None:
        adapter = create_audio_output_adapter(
            AudioOutputAdapterConfig(type=AudioOutputAdapterType.NONE),
            in_features=4,
            out_features=4,
        )
        hidden = torch.randn(2, 3, 4, dtype=torch.bfloat16)

        torch.testing.assert_close(adapter(hidden), hidden.to(dtype=torch.float32))

        with self.assertRaisesRegex(ValueError, "matching feature dimensions"):
            AudioOutputAdapter(
                AudioOutputAdapterConfig(type=AudioOutputAdapterType.NONE),
                in_features=4,
                out_features=6,
            )

    def test_config_mapping_is_normalized(self) -> None:
        config = audio_output_options({"type": "mlp"})

        self.assertEqual(config.type, AudioOutputAdapterType.MLP)

    def test_default_config_selects_linear(self) -> None:
        self.assertIs(
            AudioOutputAdapterConfig().type,
            AudioOutputAdapterType.LINEAR,
        )

    def test_invalid_hidden_shape_is_rejected(self) -> None:
        adapter = create_audio_output_adapter(
            AudioOutputAdapterConfig(type=AudioOutputAdapterType.LINEAR),
            in_features=4,
            out_features=6,
        )

        with self.assertRaisesRegex(ValueError, "at least two dimensions"):
            adapter(torch.randn(4))
        with self.assertRaisesRegex(ValueError, "does not match"):
            adapter(torch.randn(2, 3, 5))


if __name__ == "__main__":
    unittest.main()
