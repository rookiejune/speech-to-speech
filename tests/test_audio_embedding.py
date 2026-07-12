from __future__ import annotations

import unittest

import torch

from speech_to_speech.model.embedding.audio import _merge, merge_by_positions


class AudioEmbeddingTest(unittest.TestCase):
    def test_merge_by_positions_matches_grouped_rope_mean(self):
        features = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
        features.requires_grad_()
        positions = torch.tensor([[2, 1, 2, -1, 1], [0, 3, 0, 3, -1]])

        output = merge_by_positions(features, positions, sequence_length=4)

        expected = torch.zeros_like(output)
        for row in range(features.size(0)):
            for position in (positions[row][positions[row] >= 0].unique()):
                selected = positions[row] == position
                expected[row, position] = _merge(features[row][selected])
        self.assertTrue(torch.allclose(output, expected))

        output.sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())


if __name__ == "__main__":
    unittest.main()
