from __future__ import annotations

import unittest

import torch

from speech_to_speech.model.embedding.audio import merge_by_positions


class AudioEmbeddingTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
