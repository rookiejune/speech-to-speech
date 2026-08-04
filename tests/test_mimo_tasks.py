from __future__ import annotations

import unittest

import torch

from speech_to_speech.mimo import (
    KIMI_PRETRAIN_TASK_WEIGHTS,
    MimoSegment,
    MimoSpecialTokens,
    MimoTask,
    build_mimo_sample,
)


def _segment(index: int = 0) -> MimoSegment:
    return MimoSegment(
        text_input_ids=torch.tensor([3 + index, 4 + index]),
        audio_input_ids=torch.tensor([12 + index, 13 + index, 14 + index]),
        audio_features=torch.arange(6, dtype=torch.float32).view(3, 2) + index,
        recording_id="recording",
        segment_index=index,
    )


class MimoTaskCompositionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.special = MimoSpecialTokens(
            text_bos=1,
            text_eos=2,
            text_blank=0,
            audio_bos=10,
            audio_eos=11,
            audio_blank=9,
            audio_delay_tokens=2,
        )

    def test_declares_all_seven_kimi_task_weights(self) -> None:
        self.assertEqual(set(KIMI_PRETRAIN_TASK_WEIGHTS), set(MimoTask))
        self.assertEqual(KIMI_PRETRAIN_TASK_WEIGHTS[MimoTask.TEXT_ONLY], 7.0)
        self.assertEqual(
            KIMI_PRETRAIN_TASK_WEIGHTS[
                MimoTask.AUDIO_TO_NEXT_SEMANTIC_AND_TEXT
            ],
            2.0,
        )

    def test_text_only_supervises_only_text_after_bos(self) -> None:
        sample = build_mimo_sample(
            MimoTask.TEXT_ONLY,
            [_segment()],
            self.special,
        )

        self.assertTrue(
            torch.equal(sample.text_input_ids, torch.tensor([1, 3, 4, 2]))
        )
        self.assertTrue(sample.audio_input_ids.eq(self.special.audio_blank).all())
        self.assertTrue(
            torch.equal(
                sample.text_loss_mask,
                torch.tensor([False, True, True, True]),
            )
        )
        self.assertFalse(bool(sample.audio_loss_mask.any()))
        self.assertIsNone(sample.audio_features)

    def test_audio_only_applies_configured_delay_without_feature_leakage(self) -> None:
        sample = build_mimo_sample(
            MimoTask.AUDIO_ONLY,
            [_segment()],
            self.special,
        )

        self.assertTrue(
            torch.equal(
                sample.audio_input_ids,
                torch.tensor([9, 9, 10, 12, 13, 14, 11]),
            )
        )
        self.assertTrue(
            torch.equal(
                sample.audio_loss_mask,
                torch.tensor([False, False, False, True, True, True, True]),
            )
        )
        self.assertIsNone(sample.audio_features)
        self.assertIsNone(sample.audio_feature_mask)

    def test_audio_to_text_injects_features_only_on_observed_audio(self) -> None:
        sample = build_mimo_sample(
            MimoTask.AUDIO_TO_TEXT,
            [_segment()],
            self.special,
        )

        self.assertEqual(sample.text_input_ids.numel(), 9)
        self.assertIsNotNone(sample.audio_feature_mask)
        assert sample.audio_feature_mask is not None
        self.assertTrue(
            torch.equal(
                sample.audio_feature_mask,
                torch.tensor(
                    [False, True, True, True, False, False, False, False, False]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                sample.text_loss_mask,
                torch.tensor(
                    [False, False, False, False, False, False, True, True, True]
                ),
            )
        )
        self.assertFalse(bool(sample.audio_loss_mask.any()))

    def test_next_parallel_uses_context_then_delayed_dual_target(self) -> None:
        sample = build_mimo_sample(
            MimoTask.AUDIO_TO_NEXT_SEMANTIC_AND_TEXT,
            [_segment(0), _segment(1)],
            self.special,
        )

        context_length = 5
        self.assertEqual(sample.text_input_ids.numel(), context_length + 7)
        self.assertEqual(sample.recording_id, "recording")
        self.assertTrue(
            torch.equal(
                sample.text_loss_mask[context_length:],
                torch.tensor([False, True, True, True, False, False, False]),
            )
        )
        self.assertTrue(
            torch.equal(
                sample.audio_loss_mask[context_length:],
                torch.tensor([False, False, False, True, True, True, True]),
            )
        )
        assert sample.audio_feature_mask is not None
        self.assertFalse(bool(sample.audio_feature_mask[context_length:].any()))

    def test_contextual_tasks_require_ordered_adjacent_segments(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly two"):
            build_mimo_sample(
                MimoTask.AUDIO_TO_NEXT_TEXT,
                [_segment()],
                self.special,
            )
        with self.assertRaisesRegex(ValueError, "ordered and unique"):
            build_mimo_sample(
                MimoTask.AUDIO_TO_NEXT_TEXT,
                [_segment(1), _segment(0)],
                self.special,
            )
        with self.assertRaisesRegex(ValueError, "consecutive"):
            build_mimo_sample(
                MimoTask.AUDIO_TO_NEXT_TEXT,
                [_segment(0), _segment(2)],
                self.special,
            )
        missing = _segment(0)
        missing = MimoSegment(
            missing.text_input_ids,
            missing.audio_input_ids,
            missing.audio_features,
        )
        with self.assertRaisesRegex(ValueError, "recording_id"):
            build_mimo_sample(
                MimoTask.AUDIO_TO_NEXT_TEXT,
                [missing, _segment(1)],
                self.special,
            )
        other_recording = _segment(1)
        other_recording = MimoSegment(
            other_recording.text_input_ids,
            other_recording.audio_input_ids,
            other_recording.audio_features,
            recording_id="other-recording",
            segment_index=1,
        )
        with self.assertRaisesRegex(ValueError, "recording_id"):
            build_mimo_sample(
                MimoTask.AUDIO_TO_NEXT_TEXT,
                [_segment(0), other_recording],
                self.special,
            )
        missing_index = _segment(1)
        missing_index = MimoSegment(
            missing_index.text_input_ids,
            missing_index.audio_input_ids,
            missing_index.audio_features,
            recording_id="recording",
        )
        with self.assertRaisesRegex(ValueError, "segment_index"):
            build_mimo_sample(
                MimoTask.AUDIO_TO_NEXT_TEXT,
                [_segment(0), missing_index],
                self.special,
            )
        with self.assertRaisesRegex(ValueError, "exactly two"):
            build_mimo_sample(
                MimoTask.AUDIO_TO_NEXT_TEXT,
                [_segment(0), _segment(1), _segment(2)],
                self.special,
            )


if __name__ == "__main__":
    unittest.main()
