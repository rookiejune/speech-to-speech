from __future__ import annotations

import unittest

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes

from speech_to_speech.codes import AudioCodes


class AudioCodesTest(unittest.TestCase):
    def test_partial_global_stream_is_valid(self) -> None:
        codes = AudioCodes(global_codes=torch.tensor([[1], [2]]))

        self.assertIsNone(codes.semantic_codes)
        self.assertIsNone(codes.acoustic_codes)

    def test_anycodec_export_requires_semantic_stream(self) -> None:
        codes = AudioCodes(global_codes=torch.tensor([[1], [2]]))

        with self.assertRaisesRegex(ValueError, "requires semantic_codes"):
            codes.to_anycodec(AcousticLayout.FIXED_LENGTH)

    def test_fixed_anycodec_codes_become_global(self) -> None:
        anycodec_codes = SemanticAcousticCodes(
            semantic=torch.tensor([[1], [2]]),
            acoustic=torch.tensor([[3, 4], [5, 6]]),
        )

        codes = AudioCodes.from_anycodec(
            anycodec_codes,
            AcousticLayout.FIXED_LENGTH,
        )

        torch.testing.assert_close(codes.semantic_codes, anycodec_codes.semantic)
        torch.testing.assert_close(codes.global_codes, anycodec_codes.acoustic)
        self.assertIsNone(codes.acoustic_codes)
        restored = codes.to_anycodec(AcousticLayout.FIXED_LENGTH)
        torch.testing.assert_close(restored.semantic, anycodec_codes.semantic)
        torch.testing.assert_close(restored.acoustic, anycodec_codes.acoustic)

    def test_frame_aligned_anycodec_codes_remain_acoustic(self) -> None:
        anycodec_codes = SemanticAcousticCodes(
            semantic=torch.tensor([[1], [2]]),
            acoustic=torch.tensor([[3, 4], [5, 6]]),
        )

        codes = AudioCodes.from_anycodec(
            anycodec_codes,
            AcousticLayout.FRAME_ALIGNED,
        )

        self.assertIsNone(codes.global_codes)
        torch.testing.assert_close(codes.acoustic_codes, anycodec_codes.acoustic)
        restored = codes.to_anycodec(AcousticLayout.FRAME_ALIGNED)
        torch.testing.assert_close(restored.semantic, anycodec_codes.semantic)
        torch.testing.assert_close(restored.acoustic, anycodec_codes.acoustic)

    def test_anycodec_export_rejects_the_wrong_non_semantic_layout(self) -> None:
        codes = AudioCodes(
            semantic_codes=torch.tensor([[1]]),
            global_codes=torch.tensor([[2]]),
        )

        with self.assertRaisesRegex(ValueError, "acoustic_codes only"):
            codes.to_anycodec(AcousticLayout.FRAME_ALIGNED)


if __name__ == "__main__":
    unittest.main()
