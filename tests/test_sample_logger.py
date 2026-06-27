from __future__ import annotations

import unittest

import torch

from speech_to_speech.pl_module.sample_logger import _decode_side
from speech_to_speech.types import LongCatSide


class FakeBPE:
    def __init__(self, expanded: list[int]) -> None:
        self.expanded = expanded

    def encode_units(self, units: list[int]) -> list[int]:
        return list(units)

    def expand_ids(self, ids: list[int]) -> list[int]:
        return list(self.expanded)


class FakeCodec:
    def __init__(self) -> None:
        self.semantic_shape: tuple[int, ...] | None = None
        self.acoustic_shape: tuple[int, ...] | None = None

    def decode(self, semantic_ids: torch.Tensor, acoustic_ids: torch.Tensor) -> torch.Tensor:
        self.semantic_shape = tuple(semantic_ids.shape)
        self.acoustic_shape = tuple(acoustic_ids.shape)
        return torch.ones(1, 8)


class TaskSampleLoggerTest(unittest.TestCase):
    def test_decode_side_rejects_bpe_acoustic_length_mismatch(self) -> None:
        side = LongCatSide(
            semantic_ids=torch.tensor([1, 2]),
            acoustic_ids=torch.zeros((4, 2), dtype=torch.long),
        )

        with self.assertRaisesRegex(
            ValueError,
            "BPE-expanded semantic length must match LongCat acoustic length",
        ):
            _decode_side(FakeCodec(), FakeBPE([1, 2, 3]), side, max_audio_samples=None)

    def test_decode_side_passes_expanded_semantic_and_acoustic_to_codec(self) -> None:
        codec = FakeCodec()
        side = LongCatSide(
            semantic_ids=torch.tensor([1, 2]),
            acoustic_ids=torch.zeros((4, 3), dtype=torch.long),
        )

        audio = _decode_side(codec, FakeBPE([1, 2, 2]), side, max_audio_samples=4)

        self.assertEqual(tuple(audio.shape), (1, 4))
        self.assertEqual(codec.semantic_shape, (1, 3))
        self.assertEqual(codec.acoustic_shape, (1, 4, 3))


if __name__ == "__main__":
    unittest.main()
