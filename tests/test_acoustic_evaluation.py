from __future__ import annotations

import unittest

import torch

from scripts._acoustic_evaluation import mono, stft_distance


class AcousticEvaluationTest(unittest.TestCase):
    def test_identical_waveforms_have_zero_stft_distance(self):
        waveform = torch.randn(4096)

        metrics = stft_distance(waveform, waveform)

        self.assertEqual(float(metrics["stft_spectral_convergence"]), 0.0)
        self.assertEqual(float(metrics["stft_log_magnitude"]), 0.0)

    def test_mono_collapses_batch_and_channel_axes(self):
        waveform = torch.tensor([[[1.0, 3.0], [3.0, 5.0]]])

        result = mono(waveform)

        self.assertTrue(torch.equal(result, torch.tensor([2.0, 4.0])))


if __name__ == "__main__":
    unittest.main()
