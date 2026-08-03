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

        output, past = adapter(hidden)

        self.assertIsNone(past)
        self.assertEqual(output.shape, (2, 3, 6))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(
            all(parameter.dtype is torch.float32 for parameter in adapter.parameters())
        )
        self.assertTrue(torch.isfinite(output).all())

    def test_pointwise_selection_projects_only_supervised_rows(self) -> None:
        adapter = create_audio_output_adapter(
            AudioOutputAdapterConfig(type=AudioOutputAdapterType.MLP),
            in_features=4,
            out_features=6,
        )
        hidden = torch.randn(2, 4, 4, dtype=torch.bfloat16)
        selection = torch.tensor(
            [[False, True, False, True], [False, False, True, False]]
        )
        full, _ = adapter(hidden)
        input_shapes: list[torch.Size] = []
        handle = adapter.adapter.register_forward_pre_hook(
            lambda _module, args: input_shapes.append(args[0].shape)
        )

        try:
            selected, past = adapter(hidden, selection_mask=selection)
        finally:
            handle.remove()

        self.assertIsNone(past)
        self.assertEqual(input_shapes, [torch.Size((3, 4))])
        torch.testing.assert_close(selected, full[selection])

    def test_none_requires_matching_dimensions(self) -> None:
        adapter = create_audio_output_adapter(
            AudioOutputAdapterConfig(type=AudioOutputAdapterType.NONE),
            in_features=4,
            out_features=4,
        )
        hidden = torch.randn(2, 3, 4, dtype=torch.bfloat16)

        output, past = adapter(hidden)
        self.assertIsNone(past)
        torch.testing.assert_close(output, hidden.to(dtype=torch.float32))

        with self.assertRaisesRegex(ValueError, "matching feature dimensions"):
            AudioOutputAdapter(
                AudioOutputAdapterConfig(type=AudioOutputAdapterType.NONE),
                in_features=4,
                out_features=6,
            )

    def test_config_mapping_is_normalized(self) -> None:
        config = audio_output_options({"type": "mlp"})

        self.assertEqual(config.type, AudioOutputAdapterType.MLP)

    def test_default_config_selects_tied_head(self) -> None:
        self.assertIs(
            AudioOutputAdapterConfig().type,
            AudioOutputAdapterType.NONE,
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

    def test_transformer_cache_matches_full_recompute(self) -> None:
        adapter = create_audio_output_adapter(
            AudioOutputAdapterConfig(
                type=AudioOutputAdapterType.TRANSFORMER,
                layers=2,
                heads=2,
                ffn_ratio=2.0,
            ),
            in_features=8,
            out_features=8,
        )
        adapter.eval()
        hidden = torch.randn(2, 4, 8)
        mask = torch.tensor(
            [[True, True, True, False], [True, True, False, False]],
            dtype=torch.bool,
        )
        full, _ = adapter(hidden, attention_mask=mask, use_cache=False)

        past = None
        steps = []
        for index in range(hidden.size(1)):
            step, past = adapter(
                hidden[:, index : index + 1],
                attention_mask=mask[:, index : index + 1],
                past_key_values=past,
                use_cache=True,
            )
            steps.append(step)
        cached = torch.cat(steps, dim=1)
        torch.testing.assert_close(cached, full, atol=1e-5, rtol=1e-5)

    def test_transformer_selection_preserves_full_sequence_context(self) -> None:
        adapter = create_audio_output_adapter(
            AudioOutputAdapterConfig(
                type=AudioOutputAdapterType.TRANSFORMER,
                layers=1,
                heads=2,
                ffn_ratio=2.0,
            ),
            in_features=8,
            out_features=8,
        ).eval()
        hidden = torch.randn(2, 4, 8)
        attention = torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        )
        selection = torch.tensor(
            [[False, True, True, False], [True, False, False, False]]
        )

        full, _ = adapter(hidden, attention_mask=attention)
        selected, _ = adapter(
            hidden,
            attention_mask=attention,
            selection_mask=selection,
        )

        torch.testing.assert_close(selected, full[selection])

    def test_transformer_batch_select_past(self) -> None:
        adapter = create_audio_output_adapter(
            AudioOutputAdapterConfig(
                type=AudioOutputAdapterType.TRANSFORMER,
                layers=1,
                heads=2,
                ffn_ratio=2.0,
            ),
            in_features=8,
            out_features=8,
        )
        hidden = torch.randn(3, 2, 8)
        _, past = adapter(hidden, use_cache=True)
        selected = adapter.batch_select_past(past, torch.tensor([0, 2]))
        assert selected is not None
        self.assertEqual(selected[0][0].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
