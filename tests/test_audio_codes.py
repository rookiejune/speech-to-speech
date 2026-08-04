from __future__ import annotations

import unittest

import torch
from anytrain.codec import SemanticAcousticCodes, SemanticGlobalCodes

from speech_to_speech.audio import AudioCodes


class AudioCodesTest(unittest.TestCase):
    def test_partial_global_stream_is_valid(self) -> None:
        codes = AudioCodes(global_codes=torch.tensor([[1], [2]]))

        self.assertIsNone(codes.semantic_codes)
        self.assertIsNone(codes.acoustic_codes)

    def test_semantic_global_export_requires_semantic_stream(self) -> None:
        codes = AudioCodes(global_codes=torch.tensor([[1], [2]]))

        with self.assertRaisesRegex(ValueError, "requires semantic_codes"):
            codes.to_semantic_global()

    def test_semantic_global_codes_roundtrip(self) -> None:
        structured = SemanticGlobalCodes(
            semantic=torch.tensor([[1], [2]]),
            global_codes=torch.tensor([[3, 4], [5, 6]]),
        )

        codes = AudioCodes.from_semantic_global(structured)

        torch.testing.assert_close(codes.semantic_codes, structured.semantic)
        torch.testing.assert_close(codes.global_codes, structured.global_codes)
        self.assertIsNone(codes.acoustic_codes)
        restored = codes.to_semantic_global()
        torch.testing.assert_close(restored.semantic, structured.semantic)
        torch.testing.assert_close(restored.global_codes, structured.global_codes)

    def test_semantic_acoustic_codes_roundtrip(self) -> None:
        structured = SemanticAcousticCodes(
            semantic=torch.tensor([[1], [2]]),
            acoustic=torch.tensor([[3, 4], [5, 6]]),
        )

        codes = AudioCodes.from_semantic_acoustic(structured)

        self.assertIsNone(codes.global_codes)
        torch.testing.assert_close(codes.acoustic_codes, structured.acoustic)
        restored = codes.to_semantic_acoustic()
        torch.testing.assert_close(restored.semantic, structured.semantic)
        torch.testing.assert_close(restored.acoustic, structured.acoustic)

    def test_semantic_acoustic_export_rejects_global_codes(self) -> None:
        codes = AudioCodes(
            semantic_codes=torch.tensor([[1]]),
            global_codes=torch.tensor([[2]]),
        )

        with self.assertRaisesRegex(ValueError, "requires.*acoustic_codes"):
            codes.to_semantic_acoustic()


if __name__ == "__main__":
    unittest.main()
