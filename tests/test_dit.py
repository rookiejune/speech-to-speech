from __future__ import annotations

import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
