from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast

import torch
from anydataset.types import (
    AudioMeta,
    AudioView,
    Lang,
    Modality,
    Role,
    TextMeta,
    TextView,
)
from anydataset.types import AudioItem, TextItem
from anytrain.module.idspace import Layout

from speech_to_speech.datamodule.builder import build_task_sample
from speech_to_speech.datamodule.parse import parse_task_sample
from speech_to_speech.datamodule.sample import (
    AudioContextSample,
)
from speech_to_speech.runtime import AudioSequenceLayout
from speech_to_speech.runtime.audio_schema import AudioTokenSpec
from speech_to_speech.runtime.audio_tokenizer import (
    BiCodecAudioTokenizer,
    NativeAudioTokenizer,
)
from speech_to_speech.task import ControlToken, Task


class PairReferenceContextTest(unittest.TestCase):
    def test_tts_generates_global_when_audio_is_not_a_task_input(self) -> None:
        runtime = _bicodec_runtime(AudioSequenceLayout.FLATTENED)
        parsed = parse_task_sample(_pair_sample(), Task.TTS, cast(object, runtime))
        self.assertIsNone(parsed.audio_context)
        built = build_task_sample(parsed, cast(object, runtime))
        response = built.labels.response_ids
        local = response[2:-1] - runtime.layout.blocks["audio"][0]
        self.assertEqual(int(local[0]), runtime.audio_tokenizer.global_token_id)
        self.assertNotIn("audio_context", built.request)

    def test_asr_does_not_materialize_decode_context(self) -> None:
        runtime = _bicodec_runtime(AudioSequenceLayout.FLATTENED)
        parsed = parse_task_sample(_pair_sample(), Task.ASR, cast(object, runtime))
        self.assertIsNone(parsed.audio_context)
        built = build_task_sample(parsed, cast(object, runtime))
        self.assertNotIn("audio_context", built.request)

    def test_s2st_predicts_complete_target_audio(self) -> None:
        runtime = _bicodec_runtime(AudioSequenceLayout.FLATTENED)
        parsed = parse_task_sample(_pair_sample(), Task.S2ST, cast(object, runtime))
        built = build_task_sample(parsed, cast(object, runtime))
        response = built.labels.response_ids
        local = response[2:-1] - runtime.layout.blocks["audio"][0]
        self.assertEqual(int(local[0]), runtime.audio_tokenizer.global_token_id)
        decoded = runtime.audio_tokenizer.decode_full(local)
        self.assertIsNotNone(decoded.global_codes)
        self.assertIsNotNone(decoded.semantic_codes)

    def test_explicit_audio_context_is_rejected(self) -> None:
        runtime = _bicodec_runtime(AudioSequenceLayout.FLATTENED)
        pair = _pair_sample()
        context_cell = {
            (Role.DEFAULT, Modality.AUDIO): _audio(99),
            (Role.DEFAULT, Modality.TEXT): TextItem(
                views={TextView.TEXT: "context text"},
                meta={TextMeta.LANG: Lang.EN},
            ),
        }
        wrapped = AudioContextSample(sample=pair, audio_context=context_cell)
        with self.assertRaisesRegex(ValueError, "TTS_VOICE_CLONE"):
            parse_task_sample(wrapped, Task.TTS, cast(object, runtime))

    def test_voice_clone_uses_source_audio_and_predicts_full_target(self) -> None:
        runtime = _bicodec_runtime(AudioSequenceLayout.FLATTENED)
        parsed = parse_task_sample(
            _pair_sample(),
            Task.TTS_VOICE_CLONE,
            cast(object, runtime),
        )

        built = build_task_sample(parsed, cast(object, runtime))

        positions = built.audio_input_positions
        self.assertIsNotNone(positions)
        assert positions is not None
        prompt_audio = built.request["prompt_ids"].index_select(0, positions)
        audio_start = runtime.layout.blocks[runtime.input_audio_block_name][0]
        source_local = prompt_audio - audio_start
        source_codes = runtime.input_audio_tokenizer.decode_full(source_local)
        torch.testing.assert_close(source_codes.semantic_codes, parsed.source.semantic_codes)
        response = built.labels.response_ids
        output_start = runtime.layout.blocks["audio"][0]
        target_codes = runtime.audio_tokenizer.decode_full(response[2:-1] - output_start)
        torch.testing.assert_close(target_codes.semantic_codes, parsed.target.semantic_codes)
        torch.testing.assert_close(target_codes.global_codes, parsed.target.global_codes)

    def test_voice_clone_supports_asymmetric_audio_runtime(self) -> None:
        runtime = _glm4_bicodec_runtime()
        parsed = parse_task_sample(
            _pair_sample(),
            Task.TTS_VOICE_CLONE,
            cast(object, runtime),
        )

        built = build_task_sample(parsed, cast(object, runtime))

        positions = built.audio_input_positions
        self.assertIsNotNone(positions)
        assert positions is not None
        prompt = built.request["prompt_ids"]
        input_start, input_end = runtime.layout.blocks["audio_input"]
        prompt_audio = prompt.index_select(0, positions)
        self.assertTrue(prompt_audio.ge(input_start).all())
        self.assertTrue(prompt_audio.lt(input_end).all())
        input_boa = (prompt == runtime.input_boa_token_id).nonzero().flatten()
        self.assertEqual(input_boa.numel(), 1)
        start = int(input_boa[0]) - parsed.target.text_token_ids.numel()
        torch.testing.assert_close(
            prompt[start : int(input_boa[0])],
            parsed.target.text_token_ids,
        )
        output_start = runtime.layout.blocks["audio"][0]
        response_local = built.labels.response_ids[2:-1] - output_start
        decoded = runtime.audio_tokenizer.decode_full(response_local)
        torch.testing.assert_close(decoded.semantic_codes, parsed.target.semantic_codes)
        torch.testing.assert_close(decoded.global_codes, parsed.target.global_codes)


def _pair_sample() -> dict:
    return {
        (Role.SOURCE, Modality.AUDIO): _audio(0),
        (Role.SOURCE, Modality.TEXT): TextItem(
            views={TextView.TEXT: "source text"},
            meta={TextMeta.LANG: Lang.ZH},
        ),
        (Role.TARGET, Modality.AUDIO): _audio(1),
        (Role.TARGET, Modality.TEXT): TextItem(
            views={TextView.TEXT: "target text"},
            meta={TextMeta.LANG: Lang.EN},
        ),
    }


def _audio(offset: int) -> AudioItem:
    frames = 4
    unit_length = 2
    semantic = ((torch.arange(frames) + offset) % 8).unsqueeze(-1)
    global_codes = torch.stack(
        [
            (torch.arange(unit_length) + offset) % 3,
            (torch.arange(unit_length) + offset + 1) % 3,
        ],
        dim=-1,
    )
    return AudioItem(
        views={
            AudioView.GLM4: semantic,
            AudioView.BICODEC: {
                "semantic": semantic,
                "global": global_codes,
            }
        },
        meta={AudioMeta.DURATION: frames / 50.0},
    )


def _bicodec_runtime(audio_sequence_layout: AudioSequenceLayout):
    tokenizer = BiCodecAudioTokenizer(
        semantic_codebook_size=8,
        global_codebook_sizes=(3, 3),
        global_unit_length=2,
    )
    lexical_text_vocab_size = 10
    control_token_ids = tuple(
        range(lexical_text_vocab_size, lexical_text_vocab_size + len(ControlToken))
    )
    audio_start = lexical_text_vocab_size + len(ControlToken)
    boa = audio_start + tokenizer.vocab_size
    spec = AudioTokenSpec.create(
        codec_name="bicodec",
        sequence_layout=audio_sequence_layout.value,
        tokenizer=tokenizer,
    )
    runtime = SimpleNamespace(
        input_audio_decoupled=False,
        input_codec_name="bicodec",
        input_audio_view=AudioView.BICODEC,
        input_codec_frame_rate=50.0,
        codec_name="bicodec",
        audio_view=AudioView.BICODEC,
        codec_frame_rate=50.0,
        audio_sequence_layout=audio_sequence_layout,
        acoustic_generator_artifact=None,
        text_tokenizer=_ChatTokenizer(),
        input_audio_tokenizer=tokenizer,
        audio_tokenizer=tokenizer,
        input_audio_token_spec=spec,
        audio_token_spec=spec,
        output_audio_token_spec=spec,
        layout=Layout(text=(0, audio_start), audio=(audio_start, boa + 4)),
        lexical_text_vocab_size=lexical_text_vocab_size,
        control_token_ids=control_token_ids,
        control_token_id=lambda token: control_token_ids[
            list(ControlToken).index(token)
        ],
        pad_token_id=0,
        eos_token_id=1,
        boa_token_id=boa,
        eoa_token_id=boa + 1,
        mask_token_id=boa + 2,
        audio_schema_token_id=boa + 3,
        input_audio_block_name="audio",
        input_boa_token_id=boa,
        input_eoa_token_id=boa + 1,
        input_audio_schema_token_id=boa + 3,
        input_codec_audio_range=(audio_start, boa),
    )
    return runtime


def _glm4_bicodec_runtime():
    input_tokenizer = NativeAudioTokenizer(vocab_size=8)
    output_tokenizer = BiCodecAudioTokenizer(
        semantic_codebook_size=8,
        global_codebook_sizes=(3, 3),
        global_unit_length=2,
    )
    lexical_text_vocab_size = 10
    control_token_ids = tuple(
        range(lexical_text_vocab_size, lexical_text_vocab_size + len(ControlToken))
    )
    input_start = lexical_text_vocab_size + len(ControlToken)
    input_boa = input_start + input_tokenizer.vocab_size
    audio_start = input_boa + 4
    boa = audio_start + output_tokenizer.vocab_size
    return SimpleNamespace(
        input_audio_decoupled=True,
        input_codec_name="glm4",
        input_audio_view=AudioView.GLM4,
        input_codec_frame_rate=12.5,
        codec_name="bicodec",
        audio_view=AudioView.BICODEC,
        codec_frame_rate=50.0,
        audio_sequence_layout=AudioSequenceLayout.FLATTENED,
        acoustic_generator_artifact=None,
        text_tokenizer=_ChatTokenizer(),
        input_audio_tokenizer=input_tokenizer,
        audio_tokenizer=output_tokenizer,
        layout=Layout(
            text=(0, input_start),
            audio_input=(input_start, audio_start),
            audio=(audio_start, boa + 4),
        ),
        lexical_text_vocab_size=lexical_text_vocab_size,
        control_token_ids=control_token_ids,
        control_token_id=lambda token: control_token_ids[
            list(ControlToken).index(token)
        ],
        pad_token_id=0,
        eos_token_id=1,
        boa_token_id=boa,
        eoa_token_id=boa + 1,
        mask_token_id=boa + 2,
        audio_schema_token_id=boa + 3,
        input_audio_block_name="audio_input",
        input_boa_token_id=input_boa,
        input_eoa_token_id=input_boa + 1,
        input_audio_schema_token_id=input_boa + 3,
        input_codec_audio_range=(input_start, input_boa),
    )


class _ChatTokenizer:
    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        self.encoded.append(text)
        return [2, 3]

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return " ".join(message["content"] for message in messages)


if __name__ == "__main__":
    unittest.main()
