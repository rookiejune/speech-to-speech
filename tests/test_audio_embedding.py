from __future__ import annotations

import unittest
from typing import cast

import torch

from speech_to_speech.model.embedding.audio import (
    base_weight,
    embedding,
)


class AudioEmbeddingTest(unittest.TestCase):
    def test_base_weight_chunks_large_vocabularies(self):
        codebook = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        tokenizer = _Tokenizer(
            [[(token_id % 3,)] for token_id in range(2_049)]
        )

        weight = base_weight(_Codec(codebook), tokenizer)

        self.assertEqual(tokenizer.decode_batch_sizes, [2_048, 1])
        self.assertEqual(tokenizer.span_batch_sizes, [2_048, 1])
        torch.testing.assert_close(weight[0], codebook[0])
        torch.testing.assert_close(weight[-1], codebook[2])

    def test_base_weight_batches_variable_span_tokens(self):
        codebook = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        tokenizer = _Tokenizer(
            [
                [(0,), (1,)],
                [(2,)],
                [(1,), (2,), (0,)],
            ]
        )

        weight = base_weight(_Codec(codebook), tokenizer)

        expected = torch.stack(
            [
                _reference_merge(codebook[torch.tensor([frame[0] for frame in token])])
                for token in tokenizer.tokens
            ]
        )
        torch.testing.assert_close(weight, expected)

    def test_random_embedding_initialization_uses_tokenizer_vocab(self):
        tokenizer = _Tokenizer([])
        tokenizer.embedding_initialization = "random"
        tokenizer.vocab_size_override = 7
        reference = torch.empty(1, dtype=torch.float64)

        audio = embedding(
            _RandomCodec(4),
            tokenizer,
            reference=reference,
        )

        self.assertEqual(audio.weight.shape, (10, 4))
        self.assertEqual(audio.weight.dtype, reference.dtype)
        self.assertEqual(audio.weight.device, reference.device)
        self.assertEqual(tokenizer.decode_batch_sizes, [])
        self.assertEqual(tokenizer.span_batch_sizes, [])

    def test_codec_initialization_requires_a_semantic_codebook(self):
        tokenizer = _Tokenizer([[(0,)]])

        with self.assertRaisesRegex(TypeError, "semantic codebook"):
            embedding(
                _RandomCodec(4),
                tokenizer,
                reference=torch.empty(1),
            )

    def test_random_initialization_requires_valid_feature_metadata(self):
        tokenizer = _Tokenizer([])
        tokenizer.embedding_initialization = "random"
        tokenizer.vocab_size_override = 2

        for codec, error, message in (
            (object(), TypeError, "feature dimension"),
            (_RandomCodec(0), ValueError, "positive"),
            (_RandomCodec(cast(int, True)), TypeError, "integer"),
            (_Codec(torch.zeros(2, 3, 0)), ValueError, "positive"),
        ):
            with self.subTest(codec=codec), self.assertRaisesRegex(error, message):
                embedding(codec, tokenizer, reference=torch.empty(1))


def _reference_merge(embeddings: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(embeddings.size(0), dtype=torch.float32)
    dimensions = torch.arange(0, embeddings.size(-1), 2, dtype=torch.float32)
    angles = positions[:, None] * 10_000 ** (-dimensions / embeddings.size(-1))
    even = embeddings[:, 0::2]
    odd = embeddings[:, 1::2]
    rotated = torch.stack(
        (
            even * angles.cos() - odd * angles.sin(),
            even * angles.sin() + odd * angles.cos(),
        ),
        dim=-1,
    )
    return rotated.flatten(-2).mean(0)


class _Codec:
    def __init__(self, semantic_codebook: torch.Tensor) -> None:
        self.semantic_codebook = semantic_codebook


class _RandomCodec:
    def __init__(self, semantic_feature_dim: int) -> None:
        self.semantic_feature_dim = semantic_feature_dim


class _Tokenizer:
    def __init__(self, tokens: list[list[tuple[int, ...]]]) -> None:
        self.tokens = tokens
        self.embedding_initialization = "codec"
        self.vocab_size_override: int | None = None
        self.decode_batch_sizes: list[int] = []
        self.span_batch_sizes: list[int] = []

    @property
    def vocab_size(self) -> int:
        if self.vocab_size_override is not None:
            return self.vocab_size_override
        return len(self.tokens)

    def decode(self, token_ids):
        self.decode_batch_sizes.append(len(token_ids))
        return [frame for token_id in token_ids for frame in self.tokens[token_id]]

    def frame_spans(self, token_ids):
        self.span_batch_sizes.append(len(token_ids))
        return [len(self.tokens[token_id]) for token_id in token_ids]


if __name__ == "__main__":
    unittest.main()
