from __future__ import annotations

import unittest
from typing import Any, cast

import torch
from anytrain.module.idspace import Layout

from speech_to_speech.model import AdapterType
from speech_to_speech.model.audio_output import (
    AudioOutputAdapterConfig,
    AudioOutputAdapterType,
)
from speech_to_speech.model.base import Config, Model
from speech_to_speech.model.embedding.fsq import FsqAffineEmbedding
from speech_to_speech.model.token import TokenInterface
from speech_to_speech.model.toy import ToyConfig, create_toy_backbone
from speech_to_speech.runtime import AudioSequenceLayout
from speech_to_speech.runtime.audio_tokenizer import (
    FlattenedAudioTokenizer,
    NativeAudioTokenizer,
)


class _Codec:
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_codebook = torch.randn(2, 4)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes[..., 0].float()


class _FsqCodec:
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_feature_dim = 1
    fsq_levels = ((2, 2),)
    codebook_sizes = (4,)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes[..., 0].float()


class _Runtime:
    def __init__(
        self,
        *,
        codec: object | None = None,
        audio_tokenizer: object | None = None,
        audio_vocab_size: int = 5,
    ) -> None:
        self.layout = Layout(text=(0, 8), audio=(8, 8 + audio_vocab_size))
        self.audio_tokenizer = audio_tokenizer or NativeAudioTokenizer(vocab_size=2)
        self.codec = codec or _Codec()
        self.audio_sequence_layout = AudioSequenceLayout.SEMANTIC
        self.eos_token_id = 3
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.boa_token_id = 8 + audio_vocab_size - 3
        self.eoa_token_id = 8 + audio_vocab_size - 2
        self.mask_token_id = 8 + audio_vocab_size - 1
        self.structured_full_sequence = False

    @property
    def codec_audio_range(self) -> tuple[int, int]:
        return 8, self.boa_token_id

    @property
    def acoustic_side_channel(self) -> bool:
        return False


class TokenInterfaceStructureTest(unittest.TestCase):
    def test_interface_has_no_unused_local_checkpoint_version(self) -> None:
        self.assertNotIn("_version", TokenInterface.__dict__)

    def test_interface_routes_and_ties_logits(self) -> None:
        model = Model(
            Config(
                semantic_audio_adapter=AdapterType.LINEAR,
                audio_output_adapter=AudioOutputAdapterConfig(
                    type=AudioOutputAdapterType.NONE,
                ),
                toy=ToyConfig(
                    hidden_size=8,
                    intermediate_size=16,
                    layers=1,
                    heads=2,
                    max_position_embeddings=32,
                ),
            ),
            runtime=cast(Any, _Runtime()),
        ).eval()
        ids = torch.tensor([[1, 8, 2, 9]])
        embeds = model._input_embedding(ids)
        self.assertEqual(tuple(embeds.shape), (1, 4, 8))
        hidden = torch.randn(1, 4, 8)
        text = model.text_logits(hidden)
        projected, audio_past = model.project_audio_hidden(hidden)
        audio = model.semantic_audio_logits(projected)
        text_embedding = model.text_embedding
        self.assertIsInstance(text_embedding, torch.nn.Embedding)
        self.assertIs(
            model.backbone.get_input_embeddings(),
            text_embedding,
        )
        audio_table = model.tokens.audio_embedding
        audio_projection = model.tokens.audio_projection
        audio_adapter = audio_projection.module
        self.assertIsInstance(audio_table, torch.nn.Embedding)
        self.assertIsInstance(audio_adapter, torch.nn.Linear)
        self.assertFalse(hasattr(model.tokens, "text_embedding"))
        self.assertFalse(hasattr(model.tokens, "embeddings"))
        self.assertFalse(hasattr(model.tokens, "adapters"))
        text_start, text_end = model.layout.blocks["text"]
        text_vocab_size = text_end - text_start
        self.assertEqual(text.shape[-1], text_vocab_size)
        self.assertEqual(audio.shape[-1], audio_table.num_embeddings)
        self.assertIsNone(audio_past)
        torch.testing.assert_close(projected, hidden.to(dtype=torch.float32))
        with torch.no_grad():
            tied_text = torch.nn.functional.linear(
                hidden,
                text_embedding.weight[:text_vocab_size],
            )
            tied_audio_weight = audio_adapter(audio_table.weight.to(dtype=torch.float32))
            tied_audio = torch.nn.functional.linear(
                hidden.to(dtype=tied_audio_weight.dtype),
                tied_audio_weight,
            )
        torch.testing.assert_close(text, tied_text)
        torch.testing.assert_close(audio, tied_audio)

        state = model.state_dict(keep_vars=True)
        state_keys = set(state)
        text_state_paths = [
            name for name, value in state.items() if value is text_embedding.weight
        ]
        self.assertEqual(text_state_paths, ["backbone.embed_tokens.weight"])
        self.assertIn("backbone.embed_tokens.weight", state_keys)
        self.assertIn("tokens.audio_embedding.weight", state_keys)
        self.assertTrue(
            any(key.startswith("tokens.audio_projection.") for key in state_keys)
        )
        self.assertFalse(any(key.startswith("token_embedding.") for key in state_keys))
        self.assertFalse(
            any(key.startswith("tokens.text_embedding.") for key in state_keys)
        )
        self.assertFalse(any(module is text_embedding for module in model.tokens.modules()))

    def test_toy_backbone_has_no_lm_head_state_alias(self) -> None:
        backbone = create_toy_backbone(
            ToyConfig(
                hidden_size=8,
                intermediate_size=16,
                layers=1,
                heads=2,
                max_position_embeddings=32,
            ),
            text_vocab_size=8,
        )

        self.assertFalse(hasattr(backbone, "lm_head"))
        state = cast(torch.nn.Module, cast(object, backbone)).state_dict()
        self.assertEqual(
            [name for name in state if "embed_tokens" in name],
            ["embed_tokens.weight"],
        )
        self.assertFalse(any("lm_head" in name for name in state))

    def test_current_model_state_strict_roundtrips(self) -> None:
        source = _checkpoint_model()
        target = _checkpoint_model()

        incompatible = target.load_state_dict(source.state_dict(), strict=True)

        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        expected = source.state_dict()
        actual = target.state_dict()
        self.assertEqual(actual.keys(), expected.keys())
        for key, value in expected.items():
            torch.testing.assert_close(actual[key], value, msg=key)

    def test_strict_load_rejects_legacy_model_ownership(self) -> None:
        model = _checkpoint_model()
        checkpoint = dict(model.state_dict())
        checkpoint["token_embedding.embeddings.text.weight"] = checkpoint.pop(
            "backbone.embed_tokens.weight"
        )
        checkpoint["token_embedding.embeddings.audio.weight"] = checkpoint.pop(
            "tokens.audio_embedding.weight"
        )
        checkpoint["backbone.model.layers.0.self_attn.q_proj.weight"] = checkpoint.pop(
            "backbone.layers.0.self_attn.q_proj.weight"
        )

        with self.assertRaises(RuntimeError) as raised:
            model.load_state_dict(checkpoint, strict=True)

        message = str(raised.exception)
        self.assertIn('Missing key(s) in state_dict: "backbone.embed_tokens.weight"', message)
        self.assertIn('"backbone.layers.0.self_attn.q_proj.weight"', message)
        self.assertIn('"tokens.audio_embedding.weight"', message)
        self.assertIn(
            'Unexpected key(s) in state_dict: "token_embedding.embeddings.text.weight"',
            message,
        )
        self.assertIn('"token_embedding.embeddings.audio.weight"', message)
        self.assertIn('"backbone.model.layers.0.self_attn.q_proj.weight"', message)

    def test_fsq_audio_embedding_builds_model_and_ties_logits(self) -> None:
        tokenizer = FlattenedAudioTokenizer(
            codebook_sizes=_FsqCodec.codebook_sizes,
            codec_name="stable_codec",
        )
        model = Model(
            Config(
                semantic_audio_adapter=AdapterType.LINEAR,
                audio_output_adapter=AudioOutputAdapterConfig(
                    type=AudioOutputAdapterType.NONE,
                ),
                toy=ToyConfig(
                    hidden_size=8,
                    intermediate_size=16,
                    layers=1,
                    heads=2,
                    max_position_embeddings=32,
                ),
            ),
            runtime=cast(
                Any,
                _Runtime(
                    codec=_FsqCodec(),
                    audio_tokenizer=tokenizer,
                    audio_vocab_size=tokenizer.vocab_size + 3,
                ),
            ),
        ).eval()

        audio_table = model.tokens.audio_embedding
        self.assertIsInstance(audio_table, FsqAffineEmbedding)
        self.assertEqual(audio_table.num_embeddings, tokenizer.vocab_size + 3)

        hidden = torch.randn(1, 3, 8)
        logits = model.semantic_audio_logits(hidden)
        audio_adapter = model.tokens.audio_projection.module

        self.assertEqual(logits.shape[-1], audio_table.num_embeddings)
        tied_audio_weight = audio_adapter(audio_table.weight.to(dtype=torch.float32))
        torch.testing.assert_close(
            logits,
            torch.nn.functional.linear(hidden.to(dtype=tied_audio_weight.dtype), tied_audio_weight),
        )


def _checkpoint_model() -> Model:
    return Model(
        Config(
            semantic_audio_adapter=AdapterType.LINEAR,
            audio_output_adapter=AudioOutputAdapterConfig(
                type=AudioOutputAdapterType.NONE,
            ),
            toy=ToyConfig(
                hidden_size=8,
                intermediate_size=16,
                layers=1,
                heads=2,
                max_position_embeddings=32,
            ),
        ),
        runtime=cast(Any, _Runtime()),
    )


if __name__ == "__main__":
    unittest.main()
