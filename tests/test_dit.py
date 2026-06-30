from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from speech_to_speech.config import DiTAttentionMode
from speech_to_speech.model.DiT import model as dit_model
from speech_to_speech.model.DiT.model import DiT
from speech_to_speech.model.qwen3 import Qwen3Config


class DiTTest(unittest.TestCase):
    def test_forward_uses_transformers_mask_contract(self) -> None:
        config = Qwen3Config()
        config.hidden_size = 8
        config.num_hidden_layers = 1
        config.num_attention_heads = 2
        config.num_key_value_heads = 2
        config.intermediate_size = 16
        model = DiT(config)

        output = model(
            x_t=torch.zeros((1, 3, 8)),
            last_hidden_state=torch.zeros((1, 3, 8)),
            timesteps=torch.tensor([0.5]),
            acoustic_condition=torch.zeros((1, 8)),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
        )

        self.assertEqual(tuple(output.last_hidden_state.shape), (1, 3, 8))

    def test_forward_supports_input_embeds_mask_signature(self) -> None:
        calls: list[torch.Tensor] = []
        x_t = torch.zeros((1, 3, 8))
        model = DiT(_minimal_config(num_hidden_layers=0))

        with patch.object(dit_model, "create_causal_mask", _input_embeds_mask(calls)):
            output = model(
                x_t=x_t,
                last_hidden_state=torch.zeros_like(x_t),
                timesteps=torch.tensor([0.5]),
                acoustic_condition=torch.zeros((1, 8)),
            )

        self.assertIs(calls[0], x_t)
        self.assertTrue(torch.equal(calls[1], torch.arange(3)))
        self.assertIs(output.last_hidden_state, x_t)

    def test_forward_supports_inputs_embeds_mask_signature_without_cache_position(self) -> None:
        calls: list[torch.Tensor] = []
        x_t = torch.zeros((1, 3, 8))
        model = DiT(_minimal_config(num_hidden_layers=0))

        with patch.object(dit_model, "create_causal_mask", _inputs_embeds_mask(calls)):
            output = model(
                x_t=x_t,
                last_hidden_state=torch.zeros_like(x_t),
                timesteps=torch.tensor([0.5]),
                acoustic_condition=torch.zeros((1, 8)),
            )

        self.assertIs(calls[0], x_t)
        self.assertIs(output.last_hidden_state, x_t)

    def test_bidirectional_attention_uses_padding_only_mask(self) -> None:
        config = _minimal_config()
        config.attention_mode = DiTAttentionMode.BIDIRECTIONAL
        model = DiT(config)
        masks: list[torch.Tensor] = []

        def record_attention_mask(
            module: torch.nn.Module,
            args: tuple[torch.Tensor, ...],
            kwargs: dict[str, torch.Tensor],
            output: object,
        ) -> None:
            del module, args, output
            masks.append(kwargs["attention_mask"])

        handle = model.layers[0].self_attn.register_forward_hook(
            record_attention_mask,
            with_kwargs=True,
        )
        try:
            model(
                x_t=torch.zeros((1, 3, 8)),
                last_hidden_state=torch.zeros((1, 3, 8)),
                timesteps=torch.tensor([0.5]),
                acoustic_condition=torch.zeros((1, 8)),
                attention_mask=torch.tensor([[1, 1, 0]], dtype=torch.long),
            )
        finally:
            handle.remove()

        mask = masks[0]
        self.assertEqual(tuple(mask.shape), (1, 1, 3, 3))
        self.assertTrue(torch.equal(mask[0, 0, 0, :2], torch.zeros(2)))
        self.assertEqual(float(mask[0, 0, 0, 2]), torch.finfo(mask.dtype).min)
        self.assertTrue(torch.equal(mask[0, 0, 2, :2], torch.zeros(2)))

    def test_bidirectional_attention_does_not_call_causal_mask_builder(self) -> None:
        config = _minimal_config(num_hidden_layers=0)
        config.attention_mode = DiTAttentionMode.BIDIRECTIONAL
        model = DiT(config)

        with patch.object(dit_model, "create_causal_mask") as create_causal_mask:
            output = model(
                x_t=torch.zeros((1, 3, 8)),
                last_hidden_state=torch.zeros((1, 3, 8)),
                timesteps=torch.tensor([0.5]),
                acoustic_condition=torch.zeros((1, 8)),
            )

        create_causal_mask.assert_not_called()
        self.assertEqual(tuple(output.last_hidden_state.shape), (1, 3, 8))

    def test_condition_tensors_apply_configured_branch_norm(self) -> None:
        config = _minimal_config(num_hidden_layers=0)
        config.norm_time = False
        config.norm_hidden = True
        config.norm_acoustic = True
        model = DiT(config)

        tensors = model.condition_tensors(
            last_hidden_state=torch.tensor([[[1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0]]]),
            timesteps=torch.tensor([0.5]),
            acoustic_condition=torch.tensor([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]]),
        )

        self.assertTrue(
            torch.allclose(tensors.hidden.mean(dim=-1), torch.zeros((1, 1)), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(tensors.acoustic.mean(dim=-1), torch.zeros((1, 1)), atol=1e-6)
        )
        self.assertGreater(float(tensors.time.abs().sum()), 0.0)


def _minimal_config(*, num_hidden_layers: int = 1) -> Qwen3Config:
    config = Qwen3Config()
    config.hidden_size = 8
    config.num_hidden_layers = num_hidden_layers
    config.num_attention_heads = 2
    config.num_key_value_heads = 2
    config.intermediate_size = 16
    config.layer_types = ["full_attention"] * num_hidden_layers
    return config


def _input_embeds_mask(calls: list[torch.Tensor]):
    def mask(
        *,
        config: Qwen3Config,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
        past_key_values: object | None,
        position_ids: torch.Tensor | None = None,
    ) -> None:
        del config, attention_mask, past_key_values, position_ids
        calls.extend([input_embeds, cache_position])

    return mask


def _inputs_embeds_mask(calls: list[torch.Tensor]):
    def mask(
        *,
        config: Qwen3Config,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: object | None,
        position_ids: torch.Tensor | None = None,
    ) -> None:
        del config, attention_mask, past_key_values, position_ids
        calls.append(inputs_embeds)

    return mask


if __name__ == "__main__":
    unittest.main()
