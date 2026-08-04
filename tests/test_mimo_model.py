from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from speech_to_speech.mimo import MimoBatch
from speech_to_speech.loss.mimo import MimoObjective
from speech_to_speech.model.mimo import MimoModel, MimoModelConfig
from speech_to_speech.runtime.backbone.contract import BackboneReadout


class _Body(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, *, inputs_embeds: torch.Tensor, **_kwargs: object):
        self.calls += 1
        return SimpleNamespace(
            last_hidden_state=(inputs_embeds + 1, inputs_embeds + 2),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


def _batch(*, with_features: bool = False) -> MimoBatch:
    return MimoBatch(
        text_input_ids=torch.tensor([[1, 2, 3]]),
        audio_input_ids=torch.tensor([[1, 2, 3]]),
        text_labels=torch.tensor([[-100, 2, 3]]),
        audio_labels=torch.tensor([[-100, 2, 3]]),
        text_pad_token_id=0,
        audio_pad_token_id=0,
        attention_mask=torch.ones((1, 3), dtype=torch.bool),
        audio_features=(torch.randn(1, 3, 2) if with_features else None),
        audio_feature_mask=(
            torch.tensor([[True, False, False]]) if with_features else None
        ),
    )


class MimoModelTest(unittest.TestCase):
    def test_one_body_forward_produces_both_routes_and_objective(self) -> None:
        body = _Body()
        model = MimoModel(
            body,
            text_embedding=nn.Embedding(8, 4),
            audio_embedding=nn.Embedding(9, 4),
            text_readout=BackboneReadout("last_hidden_state[0]"),
            audio_readout=BackboneReadout("last_hidden_state[1]"),
            config=MimoModelConfig(audio_feature_scale=1.0),
        )
        batch = _batch()

        logits = model(batch)
        self.assertEqual(body.calls, 1)
        item = MimoObjective().from_batch(batch, model)

        self.assertEqual(logits.text.shape, (1, 3, 8))
        self.assertEqual(logits.audio.shape, (1, 3, 9))
        self.assertEqual(body.calls, 2)
        self.assertTrue(torch.isfinite(item.loss).all())
        item.loss.mean().backward()
        self.assertIsNotNone(model.text_embedding.weight.grad)
        self.assertIsNotNone(model.audio_embedding.weight.grad)

    def test_continuous_features_require_source_mask(self) -> None:
        model = MimoModel(
            _Body(),
            text_embedding=nn.Embedding(8, 4),
            audio_embedding=nn.Embedding(9, 4),
            text_readout=BackboneReadout("last_hidden_state[0]"),
            audio_readout=BackboneReadout("last_hidden_state[1]"),
            audio_feature_projection=nn.Linear(2, 4, bias=False),
        )
        batch = _batch(with_features=True)

        hidden = model.dual_hidden_states(batch)

        self.assertEqual(hidden.text.shape, (1, 3, 4))
        self.assertEqual(hidden.audio.shape, (1, 3, 4))


if __name__ == "__main__":
    unittest.main()
