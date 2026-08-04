from __future__ import annotations

import unittest
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import torch
from torch import nn

from speech_to_speech.datamodule.mimo import MimoBatch
from speech_to_speech.model.mimo_factory import (
    MimoFactoryConfig,
    build_mimo_model,
    derive_mimo_vocab,
)
from speech_to_speech.model.mimo import TiedEmbeddingHead
from speech_to_speech.runtime.backbone.adapter import BackboneBodyAdapter
from speech_to_speech.runtime.types import BackboneOutput, BackboneReadout


class _StrictBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(12, 4)
        self.calls: list[bool] = []

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        return_dict: bool,
        **_: object,
    ) -> object:
        self.calls.append(return_dict)
        if not return_dict:
            raise AssertionError("Kimi body must be called with return_dict=True")
        hidden = inputs_embeds + 1
        return SimpleNamespace(
            last_hidden_state=(hidden, hidden + 1),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


class _Adapter:
    def __init__(self, body: _StrictBody) -> None:
        self._body = body
        self.body = BackboneBodyAdapter(
            cast(Callable[..., BackboneOutput], self._call),
            readout=BackboneReadout("last_hidden_state[0]"),
            supports_cache_position=False,
        )
        self.hidden_size = 4

    def input_embeddings(self) -> nn.Embedding:
        return self._body.embedding

    def _call(self, **kwargs: object) -> object:
        # This mirrors Hugging Face Kimi's adapter wrapper.  The factory must
        # retain this callable instead of invoking the raw body directly.
        kwargs["return_dict"] = True
        return self._body(**kwargs)


class _Root(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = _StrictBody()


class _Runtime:
    def __init__(
        self,
        *,
        text_vocab_size: int = 12,
        audio_tokenizer: object | None = None,
    ) -> None:
        self.backbone = _Root()
        self.backbone_adapter = _Adapter(self.backbone.base_model)
        self.backbone_body = "base_model"
        self.layout = SimpleNamespace(blocks={"text": (0, text_vocab_size)})
        self.audio_tokenizer = (
            SimpleNamespace(vocab_size=7)
            if audio_tokenizer is None
            else audio_tokenizer
        )
        self.backbone_readouts = {
            "text": "last_hidden_state[0]",
            "audio": "last_hidden_state[1]",
        }
        self.backbone_readout = "last_hidden_state[0]"
        self.backbone_supports_cache_position = False
        self.codec = object()
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2


def _batch() -> MimoBatch:
    return MimoBatch(
        text_input_ids=torch.tensor([[1, 2, 3]]),
        audio_input_ids=torch.tensor([[1, 2, 3]]),
        text_labels=torch.tensor([[-100, 2, 3]]),
        audio_labels=torch.tensor([[-100, 2, 3]]),
        text_pad_token_id=0,
        audio_pad_token_id=0,
    )


class MimoFactoryTest(unittest.TestCase):
    def test_factory_keeps_runtime_body_wrapper_and_two_readouts(self) -> None:
        runtime = _Runtime()
        model = build_mimo_model(runtime, MimoFactoryConfig(audio_embedding_dim=4))

        logits = model(_batch())

        self.assertEqual(logits.text.shape, (1, 3, 12))
        self.assertEqual(logits.audio.shape, (1, 3, 10))
        self.assertEqual(runtime.backbone.base_model.calls, [True])
        self.assertIn(
            "backbone.base_model.embedding.weight",
            dict(model.named_parameters()),
        )
        self.assertIn("audio_embedding.weight", dict(model.named_parameters()))

    def test_vocab_derivation_stays_local_to_each_route(self) -> None:
        vocab = derive_mimo_vocab(_Runtime())

        self.assertEqual(vocab.text_size, 12)
        self.assertEqual(vocab.audio_size, 10)
        self.assertEqual(vocab.audio_bos, 7)
        self.assertEqual(vocab.audio_eos, 8)
        self.assertEqual(vocab.audio_blank, 9)
        special = vocab.special_tokens(audio_delay_tokens=2)
        self.assertEqual(special.audio_delay_tokens, 2)

    def test_bicodec_like_tokenizer_uses_semantic_payload_vocab(self) -> None:
        runtime = _Runtime(
            audio_tokenizer=SimpleNamespace(
                semantic_vocab_size=4,
                # The complete structured serializer has extra acoustic and
                # marker rows; MIMO's semantic route must not allocate them.
                vocab_size=99,
            )
        )

        vocab = derive_mimo_vocab(runtime)
        self.assertEqual(vocab.audio_size, 7)
        self.assertEqual(vocab.audio_bos, 4)
        self.assertEqual(vocab.audio_eos, 5)
        self.assertEqual(vocab.audio_blank, 6)

        model = build_mimo_model(runtime, MimoFactoryConfig(audio_embedding_dim=4))
        self.assertEqual(model(_batch()).audio.shape[-1], 7)

    def test_explicit_audio_vocab_must_cover_payload_and_specials(self) -> None:
        runtime = _Runtime(
            audio_tokenizer=SimpleNamespace(
                semantic_vocab_size=4,
                vocab_size=99,
            )
        )

        with self.assertRaisesRegex(ValueError, "cover the semantic audio payload"):
            derive_mimo_vocab(
                runtime,
                MimoFactoryConfig(audio_vocab_size=6),
            )
        with self.assertRaisesRegex(ValueError, "cover the semantic audio payload"):
            build_mimo_model(
                runtime,
                MimoFactoryConfig(audio_embedding_dim=4, audio_vocab_size=6),
            )

    def test_factory_projects_configured_continuous_audio_features(self) -> None:
        runtime = _Runtime()
        model = build_mimo_model(
            runtime,
            MimoFactoryConfig(audio_embedding_dim=4, audio_feature_dim=2),
        )
        batch = _batch()
        batch.audio_features = torch.randn(1, 3, 2)
        batch.audio_feature_mask = torch.tensor([[True, False, False]])

        logits = model(batch)

        self.assertEqual(logits.audio.shape, (1, 3, 10))
        self.assertIsInstance(model.audio_feature_projection, nn.Linear)

    def test_local_text_head_ties_prefix_without_duplicate_embedding_owner(self) -> None:
        runtime = _Runtime(text_vocab_size=9)
        model = build_mimo_model(runtime, MimoFactoryConfig(audio_embedding_dim=4))

        logits = model(_batch())
        names = list(dict(model.named_parameters()))
        state_names = list(model.state_dict())

        self.assertEqual(logits.text.shape[-1], 9)
        self.assertIsInstance(model.text_head, TiedEmbeddingHead)
        self.assertEqual(
            names.count("backbone.base_model.embedding.weight"),
            1,
        )
        self.assertNotIn("text_head.embedding.weight", names)
        self.assertEqual(
            state_names.count("backbone.base_model.embedding.weight"),
            1,
        )


if __name__ == "__main__":
    unittest.main()
