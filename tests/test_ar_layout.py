from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout

from speech_to_speech.datamodule.build.ar import (
    build_ar_sample,
    build_pretraining_ar_sample,
)
from speech_to_speech.datamodule.types import Language, Speech, Text
from speech_to_speech.loss.token import TokenLoss
from speech_to_speech.prediction import PredictionModality
from speech_to_speech.runtime import AudioSequenceLayout
from speech_to_speech.task import Task


class _Tokenizer:
    bos_token_id = 6

    def apply_chat_template(self, *_args, **_kwargs) -> str:
        return "marker"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [1] if text == "marker" else [2, 3]


def _speech(*, frames: int, text_ids: list[int], acoustic: bool) -> Speech:
    codes = torch.arange(frames).view(frames, 1)
    return Speech(
        semantic_codes=codes,
        acoustic_codes=codes + 10 if acoustic else None,
        text_token_ids=torch.tensor(text_ids, dtype=torch.long),
        audio_token_ids=torch.arange(frames, dtype=torch.long),
        audio_token_spans=torch.ones(frames, dtype=torch.long),
        language=Language.EN,
        duration_seconds=float(frames) * 0.25,
    )


class AutoregressiveLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = SimpleNamespace(
            text_tokenizer=_Tokenizer(),
            layout=Layout(text=(0, 8), audio=(8, 20)),
            boa_token_id=18,
            eoa_token_id=19,
            eos_token_id=7,
            pad_token_id=0,
            semantic_codec_artifact=None,
            audio_sequence_layout=AudioSequenceLayout.SEMANTIC,
            audio_tokenizer=None,
        )

    def test_text_ar_supervises_content_after_marker(self):
        sample = build_ar_sample(
            Text(text_token_ids=torch.tensor([2, 3]), language=Language.EN),
            Task.TEXT_AR,
            self.runtime,
            prompt="marker",
        )
        self.assertEqual(int(sample.generation_prompt_length), 1)
        self.assertTrue(torch.equal(sample.token_labels[:1], torch.tensor([-100])))
        self.assertTrue(torch.equal(sample.token_labels[1:], sample.input_ids[1:]))
        self.assertIsNone(sample.acoustic_target)

    def test_audio_ar_uses_boa_as_unsupervised_generation_prompt(self):
        speech = _speech(frames=3, text_ids=[2, 3], acoustic=True)
        sample = build_ar_sample(
            speech,
            Task.AUDIO_AR,
            self.runtime,
            prompt="marker",
        )

        prompt_length = int(sample.generation_prompt_length)
        audio_start, _ = self.runtime.layout.blocks[Modality.AUDIO.value]
        expected_audio = speech.audio_token_ids + audio_start
        expected_response = torch.cat(
            [expected_audio, torch.tensor([self.runtime.eoa_token_id])]
        )

        self.assertEqual(prompt_length, 2)
        self.assertEqual(
            int(sample.request["prompt_ids"][-1]),
            self.runtime.boa_token_id,
        )
        self.assertTrue(sample.token_labels[:prompt_length].eq(-100).all())
        self.assertTrue(torch.equal(sample.labels.response_ids, expected_response))
        self.assertTrue(
            torch.equal(sample.token_labels[prompt_length:], expected_response)
        )
        self.assertIsNotNone(sample.acoustic_target)
        assert sample.acoustic_target is not None
        expected_positions = torch.arange(
            prompt_length,
            prompt_length + speech.audio_token_ids.numel(),
            dtype=torch.long,
        )
        self.assertTrue(
            torch.equal(
                sample.acoustic_target["token_positions"],
                expected_positions,
            )
        )
        self.assertTrue(
            torch.equal(
                sample.input_ids.index_select(0, expected_positions),
                expected_audio,
            )
        )
        self.assertIsNotNone(sample.target_ctc)
        assert sample.target_ctc is not None
        self.assertTrue(
            torch.equal(sample.target_ctc["token_positions"], expected_positions)
        )
        self.assertTrue(
            torch.equal(sample.target_ctc["text_token_ids"], speech.text_token_ids)
        )

    def test_pretraining_text_ar_uses_bos_prompt(self):
        sample = build_pretraining_ar_sample(
            Text(text_token_ids=torch.tensor([2, 3]), language=Language.EN),
            Task.TEXT_AR,
            self.runtime,
        )

        text_start, _ = self.runtime.layout.blocks[Modality.TEXT.value]
        expected_response = torch.tensor(
            [text_start + 2, text_start + 3, self.runtime.eos_token_id]
        )
        self.assertTrue(torch.equal(sample.request["prompt_ids"], torch.tensor([6])))
        self.assertTrue(torch.equal(sample.labels.response_ids, expected_response))
        self.assertEqual(int(sample.generation_prompt_length), 1)
        self.assertTrue(
            torch.equal(
                sample.token_labels,
                torch.cat([torch.tensor([-100]), expected_response]),
            )
        )
        self.assertIsNone(sample.acoustic_target)

    def test_pretraining_audio_ar_uses_boa_prompt(self):
        speech = _speech(frames=3, text_ids=[2, 3], acoustic=True)
        sample = build_pretraining_ar_sample(
            speech,
            Task.AUDIO_AR,
            self.runtime,
        )

        audio_start, _ = self.runtime.layout.blocks[Modality.AUDIO.value]
        expected_audio = speech.audio_token_ids + audio_start
        expected_response = torch.cat(
            [expected_audio, torch.tensor([self.runtime.eoa_token_id])]
        )
        self.assertTrue(
            torch.equal(
                sample.request["prompt_ids"],
                torch.tensor([self.runtime.boa_token_id]),
            )
        )
        self.assertTrue(torch.equal(sample.labels.response_ids, expected_response))
        self.assertTrue(
            torch.equal(
                sample.token_labels,
                torch.cat([torch.tensor([-100]), expected_response]),
            )
        )
        self.assertIsNotNone(sample.acoustic_target)
        assert sample.acoustic_target is not None
        self.assertTrue(
            torch.equal(
                sample.acoustic_target["token_positions"],
                torch.arange(1, 4, dtype=torch.long),
            )
        )
        self.assertIsNotNone(sample.target_ctc)
        assert sample.target_ctc is not None
        self.assertTrue(
            torch.equal(
                sample.target_ctc["token_positions"],
                torch.arange(1, 4, dtype=torch.long),
            )
        )

    def test_parallel_ar_supervises_text_then_audio(self):
        sample = build_ar_sample(
            _speech(frames=4, text_ids=[2, 3], acoustic=True),
            Task.PARALLEL_AR,
            self.runtime,
            prompt="marker",
        )
        labels = sample.token_labels
        text_start, text_end = self.runtime.layout.blocks[Modality.TEXT.value]
        audio_start, audio_end = self.runtime.layout.blocks[Modality.AUDIO.value]
        supervised = labels[labels.ne(-100)]
        self.assertTrue(((supervised >= text_start) & (supervised < text_end)).any())
        self.assertTrue(((supervised >= audio_start) & (supervised < audio_end)).any())
        self.assertIsNotNone(sample.acoustic_target)
        self.assertIsNone(sample.target_ctc)

    def test_interleaved_ar_splits_by_frame_chunks(self):
        sample = build_ar_sample(
            _speech(frames=4, text_ids=[2, 3, 4, 5], acoustic=True),
            Task.INTERLEAVED_AR,
            self.runtime,
            prompt="marker",
            interleave_audio_frames=2,
        )
        self.assertEqual(int(sample.input_ids[-1]), self.runtime.eos_token_id)
        self.assertEqual(int(sample.input_ids.eq(self.runtime.boa_token_id).sum()), 2)
        self.assertEqual(int(sample.input_ids.eq(self.runtime.eoa_token_id).sum()), 2)
        self.assertIsNone(sample.target_ctc)

    def test_token_loss_accepts_parallel_prediction(self):
        sample = build_ar_sample(
            _speech(frames=2, text_ids=[2, 3], acoustic=False),
            Task.PARALLEL_AR,
            self.runtime,
            prompt="marker",
        )
        hidden = torch.randn(1, sample.input_ids.numel(), 4)
        labels = sample.token_labels.unsqueeze(0)

        def logits(states: torch.Tensor, modality: Modality) -> torch.Tensor:
            start, end = self.runtime.layout.blocks[modality.value]
            return states.new_zeros(states.size(0), end - start)

        item = TokenLoss(self.runtime.layout)(
            hidden,
            labels,
            PredictionModality.PARALLEL,
            logits,
        )
        self.assertTrue(torch.isfinite(item.loss).all())
        self.assertGreater(float(item.details["text_tokens"].sum()), 0)
        self.assertGreater(float(item.details["audio_tokens"].sum()), 0)

    def test_token_loss_accepts_interleaved_prediction(self):
        sample = build_ar_sample(
            _speech(frames=4, text_ids=[2, 3, 4, 5], acoustic=False),
            Task.INTERLEAVED_AR,
            self.runtime,
            prompt="marker",
            interleave_audio_frames=2,
        )
        hidden = torch.randn(1, sample.input_ids.numel(), 4)
        labels = sample.token_labels.unsqueeze(0)
        modalities: list[Modality] = []

        def logits(states: torch.Tensor, modality: Modality) -> torch.Tensor:
            modalities.append(modality)
            start, end = self.runtime.layout.blocks[modality.value]
            return states.new_zeros(states.size(0), end - start)

        item = TokenLoss(self.runtime.layout)(
            hidden,
            labels,
            PredictionModality.INTERLEAVED,
            logits,
        )
        self.assertTrue(torch.isfinite(item.loss).all())
        self.assertEqual(set(modalities), {Modality.TEXT, Modality.AUDIO})
        self.assertGreater(float(item.details["text_tokens"].sum()), 0)
        self.assertGreater(float(item.details["audio_tokens"].sum()), 0)

    def test_masked_ar_inserts_mask_token_in_source(self):
        from speech_to_speech.datamodule.build.masked import build_masked_sample

        self.runtime.mask_token_id = 20
        self.runtime.layout = Layout(text=(0, 8), audio=(8, 21))
        generator = torch.Generator().manual_seed(0)
        sample = build_masked_sample(
            _speech(frames=4, text_ids=[2, 3, 4, 5], acoustic=False),
            Task.MASKED_AR,
            self.runtime,
            prompt="marker",
            mask_text_ratio=1.0,
            mask_audio_ratio=1.0,
            generator=generator,
        )
        prompt_len = int(sample.generation_prompt_length)
        self.assertGreater(prompt_len, 1)
        self.assertTrue(sample.input_ids[:prompt_len].eq(20).any())
        self.assertEqual(sample.prediction, PredictionModality.PARALLEL)
        self.assertTrue(sample.token_labels[:prompt_len].eq(-100).all())

    def test_translation_parallel_override(self):
        from speech_to_speech.datamodule.build.sample import build_speech_sample

        sample = build_speech_sample(
            _speech(frames=2, text_ids=[2, 3], acoustic=False),
            _speech(frames=2, text_ids=[2, 3], acoustic=False),
            Task.T2ST,
            self.runtime,
            prompt="marker $$$PLACEHOLDER$$$ end",
            prediction=PredictionModality.PARALLEL,
        )
        self.assertEqual(sample.prediction, PredictionModality.PARALLEL)
        text_start, text_end = self.runtime.layout.blocks[Modality.TEXT.value]
        supervised = sample.token_labels[sample.token_labels.ne(-100)]
        self.assertTrue(((supervised >= text_start) & (supervised < text_end)).any())


if __name__ == "__main__":
    unittest.main()
