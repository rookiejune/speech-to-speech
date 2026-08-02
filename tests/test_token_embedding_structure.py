from __future__ import annotations

import unittest

import torch
from anytrain.module.idspace import Layout

from speech_to_speech.model import AdapterType
from speech_to_speech.model.audio_output import (
    AudioOutputAdapterConfig,
    AudioOutputAdapterType,
)
from speech_to_speech.model.base import Config, Model
from speech_to_speech.model.embedding.fsq import FsqAffineEmbedding
from speech_to_speech.model.toy import ToyConfig
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


class TokenEmbeddingStructureTest(unittest.TestCase):
    def test_idspace_routes_and_ties_heads(self) -> None:
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
            runtime=_Runtime(),
        ).eval()
        ids = torch.tensor([[1, 8, 2, 9]])
        embeds = model._input_embedding(ids)
        self.assertEqual(tuple(embeds.shape), (1, 4, 8))
        hidden = torch.randn(1, 4, 8)
        text = model.text_logits(hidden)
        projected, audio_past = model.project_audio_hidden(hidden)
        audio = model.semantic_audio_logits(projected)
        self.assertIsInstance(
            model.token_embedding.embeddings["text"],
            torch.nn.Embedding,
        )
        self.assertIs(
            model.backbone.get_input_embeddings().weight,
            model.token_embedding.embeddings["text"].weight,
        )
        audio_table = model.token_embedding.embeddings["audio"]
        audio_adapter = getattr(
            model.token_embedding.adapters["audio"],
            "module",
            model.token_embedding.adapters["audio"],
        )
        self.assertIsInstance(audio_table, torch.nn.Embedding)
        self.assertIsInstance(audio_adapter, torch.nn.Linear)
        self.assertEqual(
            text.shape[-1],
            model.token_embedding.embeddings["text"].num_embeddings,
        )
        self.assertEqual(audio.shape[-1], audio_table.num_embeddings)
        self.assertIsNone(audio_past)
        torch.testing.assert_close(projected, hidden.to(dtype=torch.float32))
        with torch.no_grad():
            tied_text = torch.nn.functional.linear(
                hidden,
                model.token_embedding.embeddings["text"].weight,
            )
            tied_audio_weight = audio_adapter(audio_table.weight.to(dtype=torch.float32))
            tied_audio = torch.nn.functional.linear(
                hidden.to(dtype=tied_audio_weight.dtype),
                tied_audio_weight,
            )
        torch.testing.assert_close(text, tied_text)
        torch.testing.assert_close(audio, tied_audio)

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
            runtime=_Runtime(
                codec=_FsqCodec(),
                audio_tokenizer=tokenizer,
                audio_vocab_size=tokenizer.vocab_size + 3,
            ),
        ).eval()

        audio_table = model.token_embedding.embeddings["audio"]
        self.assertIsInstance(audio_table, FsqAffineEmbedding)
        self.assertEqual(audio_table.num_embeddings, tokenizer.vocab_size + 3)

        hidden = torch.randn(1, 3, 8)
        logits = model.semantic_audio_logits(hidden)
        audio_adapter = getattr(
            model.token_embedding.adapters["audio"],
            "module",
            model.token_embedding.adapters["audio"],
        )

        self.assertEqual(logits.shape[-1], audio_table.num_embeddings)
        tied_audio_weight = audio_adapter(audio_table.weight.to(dtype=torch.float32))
        torch.testing.assert_close(
            logits,
            torch.nn.functional.linear(hidden.to(dtype=tied_audio_weight.dtype), tied_audio_weight),
        )


if __name__ == "__main__":
    unittest.main()
