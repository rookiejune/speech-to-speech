from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from anydataset.types import AudioView

from speech_to_speech.runtime import AudioRepresentation, Config
from speech_to_speech.runtime.codec import load_codec


class RuntimeCodecTest(unittest.TestCase):
    def test_load_codec_longcat_uses_anytrain_backend(self) -> None:
        backend = SimpleNamespace(name="longcat")

        with patch(
            "speech_to_speech.runtime.codec.load_semantic_acoustic",
            return_value=backend,
        ) as load_codec_backend:
            codec = load_codec("longcat", device="cuda")

        self.assertIs(codec, backend)
        load_codec_backend.assert_called_once_with("longcat", device="cuda")

    def test_load_codec_stable_uses_frame_backend(self) -> None:
        backend = _StableSource()

        with patch(
            "speech_to_speech.runtime.codec.load_frame",
            return_value=backend,
        ) as load_codec_backend:
            codec = load_codec("stable_codec", device="cuda")

        self.assertEqual(codec.name, "stable_codec")
        self.assertEqual(codec.sample_rate, 16_000)
        self.assertEqual(codec.frame_rate, 25.0)
        self.assertEqual(codec.codebook_sizes, (46_656,))
        codes = codec.encode(torch.zeros(1, 1, 8), 16_000)
        torch.testing.assert_close(codes, backend.codes)
        waveform = codec.decode(codes)
        torch.testing.assert_close(waveform, backend.waveform)
        load_codec_backend.assert_called_once_with("stable_codec", device="cuda")

    def test_stable_runtime_uses_stable_audio_view(self) -> None:
        config = Config(
            codec="stable_codec",
            audio_representation=AudioRepresentation.FULL_CODEC_SEQUENCE,
        )

        self.assertIs(config.audio_view, AudioView.STABLE)


class _StableSource:
    sample_rate = 16_000
    frame_rate = 25.0
    codebook_sizes = (46_656,)

    def __init__(self) -> None:
        self.codes = torch.tensor([[[1], [2]]], dtype=torch.long)
        self.waveform = torch.zeros(1, 1, 8)

    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        del audio, sample_rate
        return self.codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        del codes
        return self.waveform


if __name__ == "__main__":
    unittest.main()
