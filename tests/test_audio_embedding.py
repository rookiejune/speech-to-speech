from __future__ import annotations

import unittest

import torch

from speech_to_speech.model.embedding.audio import (
    base_weight,
    embedding,
    merge_by_positions,
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

    def test_merge_by_positions_matches_grouped_rope_mean(self):
        features = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
        features.requires_grad_()
        positions = torch.tensor([[2, 1, 2, -1, 1], [0, 3, 0, 3, -1]])

        output, occupied = merge_by_positions(features, positions, sequence_length=4)

        expected = torch.zeros_like(output)
        for row in range(features.size(0)):
            for position in positions[row][positions[row] >= 0].unique():
                selected = positions[row] == position
                expected[row, position] = _reference_merge(features[row][selected])
        self.assertTrue(torch.allclose(output, expected))
        self.assertTrue(
            torch.equal(
                occupied,
                torch.tensor([[False, True, True, False], [True, False, False, True]]),
            )
        )

        output.sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_random_embedding_initialization_uses_tokenizer_vocab(self):
        codebook = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        tokenizer = _Tokenizer([])
        tokenizer.embedding_initialization = "random"
        tokenizer.vocab_size_override = 7

        audio = embedding(_Codec(codebook), tokenizer)

        self.assertEqual(audio.weight.shape, (9, 4))
        self.assertEqual(tokenizer.decode_batch_sizes, [])
        self.assertEqual(tokenizer.span_batch_sizes, [])


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
