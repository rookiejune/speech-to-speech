from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _contracts_helpers import *
from speech_to_speech.loader_plan import ARFraming


class SpeechSampleContractTest(unittest.TestCase):
    def test_raw_text_is_encoded_at_the_datamodule_boundary(self):
        tokenizer = _Tokenizer(10)
        runtime = SimpleNamespace(
            audio_view=AudioView.LONGCAT,
            codec_frame_rate=50.0,
            audio_sequence_layout=AudioSequenceLayout.SEMANTIC,
            semantic_codec_artifact=None,
            acoustic_layout=AcousticLayout.FRAME_ALIGNED,
            acoustic_unit_length=None,
            text_tokenizer=tokenizer,
            audio_tokenizer=NativeAudioTokenizer(vocab_size=8),
        )
        raw = _raw_sample()

        pair = parse_sample(raw, runtime)

        self.assertTrue(torch.equal(pair.source.text_token_ids, torch.tensor([1, 2])))
        self.assertIs(pair.source.language, Language.ZH)
        self.assertIs(pair.target.language, Language.EN)
        self.assertEqual(pair.source.acoustic_codes.shape, (2, 1))
        self.assertTrue(
            torch.equal(pair.source.acoustic_codes, torch.tensor([[2], [3]]))
        )
        self.assertEqual(tokenizer.encoded, ("target text", False))

    def test_parse_sample_infers_duration_from_codec_frames_when_metadata_missing(self):
        runtime = _data_runtime()

        pair = parse_sample(_raw_sample_without_duration(), runtime)

        self.assertEqual(pair.source.duration_seconds, 0.04)
        self.assertEqual(pair.target.duration_seconds, 0.04)

    def test_build_sample_uses_inferred_audio_seconds_for_audio_tasks(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        pair = parse_sample(_raw_sample_without_duration(), runtime)

        tts = build_sample(pair, Task.TTS, runtime)
        s2st = build_sample(pair, Task.S2ST, runtime)

        self.assertEqual(tts.audio_seconds, 0.04)
        self.assertEqual(s2st.audio_seconds, 0.08)

    def test_audio_spans_emit_ctc_only_when_their_transcript_is_not_visible(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        pair = parse_sample(_raw_sample(), runtime)

        expected = {
            Task.ASR: (True, False),
            Task.S2TT: (True, False),
            Task.T2ST: (False, True),
            Task.S2ST: (True, True),
            Task.TTS: (False, False),
            Task.MT: (False, False),
        }
        for task, (source_expected, target_expected) in expected.items():
            with self.subTest(task=task):
                sample = build_sample(pair, task, runtime)
                self.assertEqual(sample.source_ctc is not None, source_expected)
                self.assertEqual(sample.target_ctc is not None, target_expected)
                if sample.source_ctc is not None:
                    self.assertIsNotNone(sample.audio_input_positions)
                    source_speech = pair.source if task.uses_source_role else pair.target
                    self.assertTrue(
                        torch.equal(
                            sample.source_ctc["token_positions"],
                            sample.audio_input_positions,
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            sample.source_ctc["text_token_ids"],
                            source_speech.text_token_ids,
                        )
                    )
                if sample.target_ctc is not None:
                    positions = sample.target_ctc["token_positions"]
                    audio_start, audio_end = runtime.layout.blocks[Modality.AUDIO.value]
                    audio_ids = sample.input_ids.index_select(0, positions)
                    self.assertTrue(audio_ids.ge(audio_start).all())
                    self.assertTrue(audio_ids.lt(audio_end).all())
                    self.assertTrue(
                        torch.equal(
                            sample.target_ctc["text_token_ids"],
                            pair.target.text_token_ids,
                        )
                    )

    def test_single_collator_builds_tts_from_default_utterance(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        utterance = parse_single_sample(_raw_single_sample(), runtime)
        sample = build_single_sample(utterance, Task.TTS, runtime)

        batch = SingleCollator(runtime, {Task.TTS: 1.0})([_raw_single_sample()])

        self.assertEqual(sample.task, Task.TTS)
        self.assertEqual(batch.tasks, [Task.TTS])
        self.assertIsNotNone(batch.acoustic_target)
        supervised = batch.token_labels[batch.token_labels.ne(-100)]
        self.assertTrue(torch.equal(supervised, torch.tensor([10, 11, 19])))
        self.assertAlmostEqual(float(batch.audio_seconds[0].item()), 0.04)

    def test_single_collator_builds_asr_from_the_same_utterance_shape(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)

        batch = SingleCollator(runtime, {Task.ASR: 1.0})([_raw_single_sample()])

        self.assertEqual(batch.tasks, [Task.ASR])
        self.assertIsNone(batch.acoustic_target)
        supervised = batch.token_labels[batch.token_labels.ne(-100)]
        self.assertTrue(torch.equal(supervised, torch.tensor([1, 2, 1])))

    def test_single_text_task_does_not_require_or_encode_audio(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        runtime.codec = _EncodingCodec()

        batch = SingleCollator(
            runtime,
            {Task.TEXT_AR: 1.0},
            encode_missing_codes=True,
        )([_raw_single_waveform_sample()])

        self.assertIsInstance(batch, ModelBatch)
        self.assertEqual(batch.tasks, [Task.TEXT_AR])
        self.assertEqual(runtime.codec.calls, [])

    def test_single_audio_ar_pretraining_collator_uses_boa_prompt(self):
        runtime = _data_runtime()
        tokenizer = _ChatTokenizer(10)
        tokenizer.apply_chat_template = Mock(
            side_effect=AssertionError("pretraining must not render chat prompts")
        )
        runtime.text_tokenizer = tokenizer

        batch = SingleCollator(
            runtime,
            {Task.AUDIO_AR: 1.0},
            ar_framing=ARFraming.PRETRAINING,
        )([_raw_single_sample()])

        self.assertIsInstance(batch, ModelBatch)
        self.assertEqual(int(batch.generation_prompt_lengths[0]), 1)
        self.assertEqual(int(batch.input_ids[0, 0]), runtime.boa_token_id)
        self.assertEqual(int(batch.token_labels[0, 0]), -100)
        self.assertTrue(
            torch.equal(
                batch.token_labels[0, 1:],
                torch.tensor([10, 11, runtime.eoa_token_id]),
            )
        )
        tokenizer.apply_chat_template.assert_not_called()

    def test_pair_audio_ar_pretraining_collator_routes_without_chat_prompt(self):
        runtime = _data_runtime()
        tokenizer = _ChatTokenizer(10)
        tokenizer.apply_chat_template = Mock(
            side_effect=AssertionError("pretraining must not render chat prompts")
        )
        runtime.text_tokenizer = tokenizer

        batch = Collator(
            runtime,
            {Task.AUDIO_AR: 1.0},
            ar_framing=ARFraming.PRETRAINING,
        )([_raw_sample()])

        self.assertIsInstance(batch, ModelBatch)
        self.assertEqual(int(batch.generation_prompt_lengths[0]), 1)
        self.assertEqual(int(batch.input_ids[0, 0]), runtime.boa_token_id)
        supervised = batch.token_labels[0][batch.token_labels[0].ne(-100)]
        audio_start, _ = runtime.layout.blocks[Modality.AUDIO.value]
        self.assertTrue(supervised[:-1].ge(audio_start).all())
        self.assertEqual(int(supervised[-1]), runtime.eoa_token_id)
        tokenizer.apply_chat_template.assert_not_called()

    def test_single_collator_emits_raw_batch_only_for_explicit_waveform_fallback(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        raw = _raw_single_waveform_sample()

        with self.assertRaisesRegex(ValueError, "missing .* codec"):
            SingleCollator(runtime, {Task.TTS: 1.0})([raw])

        batch = SingleCollator(
            runtime,
            {Task.TTS: 1.0},
            encode_missing_codes=True,
        )([raw])

        self.assertIsInstance(batch, RawSpeechBatch)
        self.assertEqual(batch.tasks, [Task.TTS])
        target = batch.samples[0].target
        self.assertIsInstance(target, RawSpeech)
        self.assertEqual(target.sample_rate, 4)
        self.assertEqual(target.duration_seconds, 1.0)

    def test_on_device_codec_materializer_converts_raw_single_batch(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        runtime.codec = _EncodingCodec()
        raw = SingleCollator(
            runtime,
            {Task.TTS: 1.0},
            encode_missing_codes=True,
        )([_raw_single_waveform_sample()])

        with torch.autocast("cpu", dtype=torch.bfloat16):
            batch = OnDeviceCodecMaterializer(runtime)(
                raw,
                device=torch.device("cpu"),
            )

        self.assertIsInstance(batch, ModelBatch)
        self.assertEqual(batch.tasks, [Task.TTS])
        self.assertIsNotNone(batch.acoustic_target)
        self.assertEqual(runtime.codec.calls, [((1, 1, 4), 4)])
        self.assertEqual(runtime.codec.input_dtypes, [torch.float32])
        self.assertEqual(runtime.codec.autocast_enabled, [False])

    def test_audio_ar_pretraining_framing_survives_codec_materialization(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        runtime.codec = _EncodingCodec()
        raw = SingleCollator(
            runtime,
            {Task.AUDIO_AR: 1.0},
            encode_missing_codes=True,
            ar_framing=ARFraming.PRETRAINING,
        )([_raw_single_waveform_sample()])

        batch = OnDeviceCodecMaterializer(runtime)(
            raw,
            device=torch.device("cpu"),
        )

        self.assertIsInstance(raw, RawSpeechBatch)
        self.assertIs(raw.ar_framing, ARFraming.PRETRAINING)
        self.assertEqual(int(batch.generation_prompt_lengths[0]), 1)
        self.assertEqual(int(batch.input_ids[0, 0]), runtime.boa_token_id)
        self.assertEqual(int(batch.token_labels[0, 0]), -100)

    def test_bicodec_online_tokenize_stays_fp32_outside_autocast(self):
        runtime = _bicodec_data_runtime()
        raw = SingleCollator(
            runtime,
            {Task.TTS: 1.0},
            encode_missing_codes=True,
        )([_raw_single_waveform_sample()])

        with torch.autocast("cpu", dtype=torch.bfloat16):
            batch = OnDeviceCodecMaterializer(runtime)(
                raw,
                device=torch.device("cpu"),
            )

        self.assertIsInstance(batch, ModelBatch)
        self.assertEqual(batch.tasks, [Task.TTS])
        self.assertEqual(runtime.codec.calls, [((1, 1, 4), 4)])
        self.assertEqual(runtime.codec.input_dtypes, [torch.float32])
        self.assertEqual(runtime.codec.autocast_enabled, [False])

    def test_longcat_route_uses_frame_encode_when_codec_has_both_capabilities(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        runtime.codec = _EncodingCodec()
        raw = SingleCollator(
            runtime,
            {Task.TTS: 1.0},
            encode_missing_codes=True,
        )([_raw_single_waveform_sample()])

        with patch(
            "speech_to_speech.callback.codec.supports_structured",
            return_value=True,
        ), patch(
            "speech_to_speech.callback.codec.structured_codec",
        ) as structured:
            batch = OnDeviceCodecMaterializer(runtime)(raw, device=torch.device("cpu"))

        self.assertIsInstance(batch, ModelBatch)
        structured.assert_not_called()
        self.assertEqual(runtime.codec.calls, [((1, 1, 4), 4)])

    def test_pair_waveform_fallback_encodes_both_s2st_roles(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        runtime.codec = _EncodingCodec()
        sample = _raw_pair_waveform_sample()

        with self.assertRaisesRegex(ValueError, "missing .* codec"):
            Collator(runtime, {Task.S2ST: 1.0})([sample])

        raw = Collator(
            runtime,
            {Task.S2ST: 1.0},
            encode_missing_codes=True,
        )([sample])
        batch = OnDeviceCodecMaterializer(runtime)(raw, device=torch.device("cpu"))

        self.assertIsInstance(raw, RawSpeechBatch)
        self.assertIsInstance(batch, ModelBatch)
        self.assertEqual(batch.tasks, [Task.S2ST])
        self.assertEqual(
            runtime.codec.calls,
            [((1, 1, 4), 4), ((1, 1, 6), 4)],
        )

    def test_pair_waveform_fallback_encodes_only_task_audio_roles(self):
        for task, expected_shape in (
            (Task.S2TT, (1, 1, 4)),
            (Task.TTS, (1, 1, 6)),
        ):
            with self.subTest(task=task):
                runtime = _data_runtime()
                runtime.text_tokenizer = _ChatTokenizer(10)
                runtime.codec = _EncodingCodec()
                raw = Collator(
                    runtime,
                    {task: 1.0},
                    encode_missing_codes=True,
                )([_raw_pair_waveform_sample()])

                batch = OnDeviceCodecMaterializer(runtime)(
                    raw,
                    device=torch.device("cpu"),
                )

                self.assertIsInstance(batch, ModelBatch)
                self.assertEqual(batch.tasks, [task])
                self.assertEqual(runtime.codec.calls, [(expected_shape, 4)])

    def test_pair_waveform_fallback_can_mix_prepared_and_raw_samples(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        runtime.codec = _EncodingCodec()
        raw = Collator(
            runtime,
            {Task.S2ST: 1.0},
            encode_missing_codes=True,
        )([_raw_sample(), _raw_pair_waveform_sample()])

        batch = OnDeviceCodecMaterializer(runtime)(raw, device=torch.device("cpu"))

        self.assertIsInstance(raw, RawSpeechBatch)
        self.assertIsInstance(batch, ModelBatch)
        self.assertEqual(batch.tasks, [Task.S2ST, Task.S2ST])
        self.assertEqual(len(runtime.codec.calls), 2)

    def test_datamodule_can_select_single_shape_without_changing_pair_default(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(),
            shape=DataShape.SINGLE,
            encode_missing_codes=True,
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )

        with patch(
            "speech_to_speech.datamodule.module.load_dataset",
            return_value=[_raw_single_waveform_sample()],
        ):
            datamodule.setup()
            batch = next(iter(datamodule.train_dataloader()))

        self.assertIsInstance(batch, RawSpeechBatch)
        self.assertEqual(
            datamodule.loader_specs["train"].speech_config.shape,
            DataShape.SINGLE,
        )

    def test_datamodule_wires_waveform_fallback_for_pair_shape(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(),
            shape=DataShape.PAIR,
            encode_missing_codes=True,
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.S2ST: 1.0})},
        )

        with patch(
            "speech_to_speech.datamodule.module.load_dataset",
            return_value=[_raw_pair_waveform_sample()],
        ):
            datamodule.setup()
            batch = next(iter(datamodule.train_dataloader()))

        self.assertIsInstance(batch, RawSpeechBatch)
        self.assertEqual(batch.tasks, [Task.S2ST])

    def test_full_codec_sequence_flattens_complete_codes_without_acoustic_target(self):
        tokenizer = FlattenedAudioTokenizer(
            codebook_sizes=(8, 10),
            codec_name="longcat",
        )
        audio_start = 10
        runtime = SimpleNamespace(
            audio_view=AudioView.LONGCAT,
            codec_frame_rate=50.0,
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
            semantic_codec_artifact=None,
            acoustic_layout=AcousticLayout.FRAME_ALIGNED,
            acoustic_unit_length=None,
            text_tokenizer=_ChatTokenizer(10),
            audio_tokenizer=tokenizer,
            layout=Layout(
                text=(0, audio_start),
                audio=(audio_start, audio_start + tokenizer.vocab_size + 3),
            ),
            pad_token_id=0,
            eos_token_id=1,
            boa_token_id=audio_start + tokenizer.vocab_size,
            eoa_token_id=audio_start + tokenizer.vocab_size + 1,
            mask_token_id=audio_start + tokenizer.vocab_size + 2,
        )
        raw = _raw_sample()

        pair = parse_sample(raw, runtime)
        sample = build_sample(pair, Task.S2ST, runtime)

        source_codes = raw[(Role.SOURCE, Modality.AUDIO)].views[AudioView.LONGCAT]
        target_codes = raw[(Role.TARGET, Modality.AUDIO)].views[AudioView.LONGCAT]
        self.assertTrue(torch.equal(pair.source.semantic_codes, source_codes))
        self.assertIsNone(pair.source.acoustic_codes)
        self.assertTrue(torch.equal(pair.target.semantic_codes, target_codes))
        self.assertIsNone(pair.target.acoustic_codes)
        self.assertTrue(torch.equal(pair.target.audio_token_ids, tokenizer.encode(target_codes)))
        self.assertEqual(int(pair.target.audio_token_spans.sum().item()), target_codes.size(0))
        self.assertIsNone(sample.acoustic_target)

        supervised = sample.token_labels[sample.token_labels.ne(-100)]
        expected = torch.cat(
            [
                pair.target.audio_token_ids + audio_start,
                torch.tensor([runtime.eoa_token_id]),
            ]
        )
        self.assertTrue(torch.equal(supervised, expected))



if __name__ == "__main__":
    unittest.main()
