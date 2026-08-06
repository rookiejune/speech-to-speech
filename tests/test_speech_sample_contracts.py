from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _contracts_helpers import *
from speech_to_speech.audio import AudioCodes
from speech_to_speech.datamodule.builder import build_speech_sample, ctc_target
from speech_to_speech.datamodule.sample import Speech, SpeechTaskSample
from speech_to_speech.datamodule.loader import ARFraming
from speech_to_speech.runtime.audio_schema import AudioTokenSpec
from speech_to_speech.task import TARGET_COT, PredictionModality, resolve_response


_base_data_runtime = _data_runtime
_base_bicodec_data_runtime = _bicodec_data_runtime


def _with_audio_schema(runtime):
    codec_name = getattr(runtime, "codec_name", None)
    if codec_name is None:
        codec_name = runtime.input_codec_name
    runtime.codec_name = codec_name
    runtime.input_audio_decoupled = getattr(runtime, "input_audio_decoupled", False)
    runtime.input_codec_name = getattr(runtime, "input_codec_name", codec_name)
    runtime.input_audio_view = getattr(runtime, "input_audio_view", runtime.audio_view)
    runtime.input_codec_frame_rate = getattr(
        runtime,
        "input_codec_frame_rate",
        runtime.codec_frame_rate,
    )
    runtime.input_audio_tokenizer = getattr(
        runtime,
        "input_audio_tokenizer",
        runtime.audio_tokenizer,
    )
    runtime.bos_token_id = getattr(runtime, "bos_token_id", 2)
    runtime.audio_schema_token_id = runtime.mask_token_id + 1
    runtime.input_audio_block_name = getattr(
        runtime,
        "input_audio_block_name",
        Modality.AUDIO.value,
    )
    runtime.input_boa_token_id = getattr(
        runtime,
        "input_boa_token_id",
        runtime.boa_token_id,
    )
    runtime.input_eoa_token_id = getattr(
        runtime,
        "input_eoa_token_id",
        runtime.eoa_token_id,
    )
    runtime.input_audio_schema_token_id = (
        runtime.audio_schema_token_id
        if not runtime.input_audio_decoupled
        else runtime.input_eoa_token_id + 1
    )
    runtime.input_codec_audio_range = getattr(
        runtime,
        "input_codec_audio_range",
        (runtime.layout.blocks[runtime.input_audio_block_name][0], runtime.input_boa_token_id),
    )
    blocks = dict(runtime.layout.blocks)
    audio_start, audio_end = blocks[Modality.AUDIO.value]
    blocks[Modality.AUDIO.value] = (
        audio_start,
        max(audio_end, runtime.audio_schema_token_id + 1),
    )
    if runtime.input_audio_decoupled:
        input_start, input_end = blocks[runtime.input_audio_block_name]
        blocks[runtime.input_audio_block_name] = (
            input_start,
            max(input_end, runtime.input_audio_schema_token_id + 1),
        )
    runtime.layout = Layout(**blocks)
    output_spec = AudioTokenSpec.create(
        codec_name=runtime.codec_name,
        sequence_layout=runtime.audio_sequence_layout.value,
        tokenizer=runtime.audio_tokenizer,
    )
    input_spec = (
        output_spec
        if not runtime.input_audio_decoupled
        else AudioTokenSpec.create(
            codec_name=runtime.input_codec_name,
            sequence_layout=runtime.audio_sequence_layout.value,
            tokenizer=runtime.input_audio_tokenizer,
        )
    )
    runtime.audio_token_spec = output_spec
    runtime.output_audio_token_spec = output_spec
    runtime.input_audio_token_spec = input_spec
    return runtime


def _data_runtime():
    return _with_audio_schema(_base_data_runtime())


def _bicodec_data_runtime():
    return _with_audio_schema(_base_bicodec_data_runtime())


class SpeechSampleContractTest(unittest.TestCase):
    def test_ctc_rejects_runtime_control_rows(self):
        runtime = _data_runtime()
        speech = parse_sample(_raw_sample(), runtime).source
        speech.text_token_ids = torch.tensor([runtime.lexical_text_vocab_size])

        with self.assertRaisesRegex(ValueError, "lexical text vocabulary"):
            ctc_target(torch.tensor([0, 1]), speech, runtime)

    def test_raw_text_is_encoded_at_the_datamodule_boundary(self):
        tokenizer = _Tokenizer(10)
        audio_tokenizer = NativeAudioTokenizer(vocab_size=8)
        audio_start = 10
        boa_token_id = audio_start + audio_tokenizer.vocab_size
        runtime = SimpleNamespace(
            input_audio_decoupled=False,
            input_codec_name="longcat",
            input_audio_view=AudioView.LONGCAT,
            input_codec_frame_rate=50.0,
            audio_view=AudioView.LONGCAT,
            codec_frame_rate=50.0,
            audio_sequence_layout=AudioSequenceLayout.SEMANTIC,
            acoustic_generator_artifact=None,
            text_tokenizer=tokenizer,
            input_audio_tokenizer=audio_tokenizer,
            audio_tokenizer=audio_tokenizer,
            layout=Layout(text=(0, audio_start), audio=(audio_start, boa_token_id + 3)),
            pad_token_id=0,
            eos_token_id=1,
            boa_token_id=boa_token_id,
            eoa_token_id=boa_token_id + 1,
            mask_token_id=boa_token_id + 2,
            input_audio_block_name="audio",
            input_boa_token_id=boa_token_id,
            input_eoa_token_id=boa_token_id + 1,
            input_codec_audio_range=(audio_start, boa_token_id),
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

    def test_speech_task_prediction_is_derived_from_trace(self):
        pair = parse_sample(_raw_sample(), _data_runtime())

        sample = SpeechTaskSample(
            source=pair.source,
            target=pair.target,
            task=Task.S2ST,
            trace=TARGET_COT,
        )

        self.assertNotIn("prediction", vars(sample))
        self.assertIs(sample.prediction, PredictionModality.PARALLEL)

    def test_build_sample_uses_inferred_audio_seconds_for_audio_tasks(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        pair = parse_sample(_raw_sample_without_duration(), runtime)

        tts = build_sample(pair, Task.TTS, runtime)
        clone = build_sample(pair, Task.TTS_VOICE_CLONE, runtime)
        s2st = build_sample(pair, Task.S2ST, runtime)

        self.assertEqual(tts.audio_seconds, 0.04)
        self.assertEqual(clone.audio_seconds, 0.08)
        self.assertEqual(s2st.audio_seconds, 0.08)
        self.assertNotIn("target_language", tts.request)
        self.assertNotIn("target_language", s2st.request)

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
            Task.TTS_VOICE_CLONE: (False, False),
            Task.MT: (False, False),
        }
        for task, (source_expected, target_expected) in expected.items():
            with self.subTest(task=task):
                sample = build_sample(pair, task, runtime)
                expected_language = (
                    "en" if resolve_response(task).requires_target_language else None
                )
                self.assertEqual(sample.target_language, expected_language)
                self.assertEqual(
                    "target_language" in sample.request,
                    expected_language is not None,
                )
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
        self.assertNotIn("target_language", sample.request)
        self.assertEqual(batch.tasks, [Task.TTS])
        self.assertEqual(batch.target_languages, [None])
        self.assertIsNotNone(batch.acoustic_target)
        supervised = batch.token_labels[batch.token_labels.ne(-100)]
        self.assertTrue(
            torch.equal(
                supervised,
                torch.tensor([24, 27, 16, 17, 25]),
            )
        )
        self.assertAlmostEqual(float(batch.audio_seconds[0].item()), 0.04)

    def test_single_collator_builds_asr_from_the_same_utterance_shape(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)

        batch = SingleCollator(runtime, {Task.ASR: 1.0})([_raw_single_sample()])

        self.assertEqual(batch.tasks, [Task.ASR])
        self.assertEqual(batch.target_languages, [None])
        self.assertIsNone(batch.acoustic_target)
        supervised = batch.token_labels[batch.token_labels.ne(-100)]
        self.assertTrue(torch.equal(supervised, torch.tensor([10, 1, 2, 11])))

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
        self.assertEqual(batch.target_languages, [None])
        self.assertEqual(runtime.codec.calls, [])

    def test_single_audio_ar_pretraining_supervises_the_full_audio_envelope(self):
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
        self.assertEqual(int(batch.input_ids[0, 0]), runtime.bos_token_id)
        self.assertEqual(int(batch.token_labels[0, 0]), -100)
        self.assertTrue(
            torch.equal(
                batch.token_labels[0, 1:],
                torch.tensor(
                    [
                        runtime.boa_token_id,
                        runtime.audio_schema_token_id,
                        16,
                        17,
                        runtime.eoa_token_id,
                    ]
                ),
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
        self.assertEqual(int(batch.input_ids[0, 0]), runtime.bos_token_id)
        supervised = batch.token_labels[0][batch.token_labels[0].ne(-100)]
        audio_start, _ = runtime.layout.blocks[Modality.AUDIO.value]
        self.assertEqual(int(supervised[0]), runtime.boa_token_id)
        self.assertEqual(int(supervised[1]), runtime.audio_schema_token_id)
        self.assertTrue(supervised[2:-1].ge(audio_start).all())
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
        self.assertEqual(int(batch.input_ids[0, 0]), runtime.bos_token_id)
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
            "speech_to_speech.callback.codec.global_codec",
        ) as global_backend:
            batch = OnDeviceCodecMaterializer(runtime)(raw, device=torch.device("cpu"))

        self.assertIsInstance(batch, ModelBatch)
        global_backend.assert_not_called()
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
        lexical_text_vocab_size = 10
        controls = ControlTokenLookup(lexical_text_vocab_size)
        audio_start = lexical_text_vocab_size + len(ControlToken)
        boa_token_id = audio_start + tokenizer.vocab_size
        runtime = _with_audio_schema(SimpleNamespace(
            input_audio_decoupled=False,
            input_codec_name="longcat",
            input_audio_view=AudioView.LONGCAT,
            input_codec_frame_rate=50.0,
            audio_view=AudioView.LONGCAT,
            codec_frame_rate=50.0,
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
            acoustic_generator_artifact=None,
            text_tokenizer=_ChatTokenizer(10),
            input_audio_tokenizer=tokenizer,
            audio_tokenizer=tokenizer,
            layout=Layout(
                text=(0, audio_start),
                audio=(audio_start, boa_token_id + 3),
            ),
            lexical_text_vocab_size=lexical_text_vocab_size,
            control_token_ids=controls.ids,
            control_token_id=controls,
            pad_token_id=0,
            eos_token_id=1,
            boa_token_id=boa_token_id,
            eoa_token_id=boa_token_id + 1,
            mask_token_id=boa_token_id + 2,
            input_audio_block_name="audio",
            input_boa_token_id=boa_token_id,
            input_eoa_token_id=boa_token_id + 1,
            input_codec_audio_range=(audio_start, boa_token_id),
        ))
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
                torch.tensor(
                    [runtime.boa_token_id, runtime.audio_schema_token_id]
                ),
                pair.target.audio_token_ids + audio_start,
                torch.tensor([runtime.eoa_token_id]),
            ]
        )
        self.assertTrue(torch.equal(supervised, expected))

    def test_bicodec_ctc_span_includes_internal_serialization_markers(self):
        runtime = _bicodec_data_runtime()
        tokenizer = runtime.audio_tokenizer
        codes = AudioCodes(
            semantic_codes=torch.tensor([[1], [2]]),
            global_codes=torch.tensor([[0], [1]]),
        )
        serialized = tokenizer.encode_full(codes)
        spans = tokenizer.frame_spans(serialized)
        self.assertIsInstance(spans, torch.Tensor)
        speech = Speech(
            semantic_codes=codes.semantic_codes,
            acoustic_codes=None,
            text_token_ids=torch.tensor([1, 2]),
            audio_token_ids=serialized,
            audio_token_spans=spans,
            language=Language.EN,
            duration_seconds=0.04,
            global_codes=codes.global_codes,
        )

        sample = build_speech_sample(
            speech,
            speech,
            Task.T2ST,
            runtime,
            prompt="translate $$$PLACEHOLDER$$$ now",
        )

        self.assertIsNotNone(sample.target_ctc)
        assert sample.target_ctc is not None
        positions = sample.target_ctc["token_positions"]
        audio_start, _ = runtime.layout.blocks[Modality.AUDIO.value]
        local_ids = sample.input_ids.index_select(0, positions) - audio_start
        self.assertTrue(torch.equal(local_ids, serialized))
        self.assertEqual(int(local_ids[0]), tokenizer.global_token_id)
        self.assertTrue(local_ids.eq(tokenizer.semantic_token_id).any())
        runtime.audio_token_spec.validate_complete(local_ids)



if __name__ == "__main__":
    unittest.main()
