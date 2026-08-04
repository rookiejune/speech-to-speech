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
from anytrain.codec import AcousticLayout
from anytrain.module.idspace import Layout

from speech_to_speech.audio_stream import AudioStream
from speech_to_speech.datamodule.build.sample import build_task_sample
from speech_to_speech.datamodule.parse.parser import parse_task_sample
from speech_to_speech.datamodule.types import AudioContextSample, Speech
from speech_to_speech.runtime import AudioSequenceLayout
from speech_to_speech.runtime.audio_tokenizer import BiCodecAudioTokenizer
from speech_to_speech.task import Task


class PairReferenceContextTest(unittest.TestCase):
    def test_tts_reuse_route_uses_source_audio_as_context(self) -> None:
        runtime = _bicodec_runtime(AudioSequenceLayout.SEMANTIC)
        sample = _pair_sample()

        parsed = parse_task_sample(sample, Task.TTS, cast(object, runtime))
        self.assertIsInstance(parsed.audio_context, Speech)
        assert isinstance(parsed.audio_context, Speech)
        torch.testing.assert_close(
            parsed.audio_context.acoustic_codes,
            sample[(Role.SOURCE, Modality.AUDIO)].views[AudioView.BICODEC]["acoustic"],
        )

        built = build_task_sample(parsed, cast(object, runtime))
        self.assertIsNotNone(built.request["audio_context"])
        assert built.request["audio_context"] is not None
        torch.testing.assert_close(
            built.request["audio_context"].acoustic,
            parsed.audio_context.acoustic_codes,
        )
        local_prompt = _prompt_acoustic_ids(built.input_ids, runtime)
        decoded = runtime.audio_tokenizer.decode_streams(
            local_prompt,
            (AudioStream.ACOUSTIC,),
        )
        self.assertIsNone(decoded.semantic)
        torch.testing.assert_close(
            decoded.acoustic,
            parsed.audio_context.acoustic_codes,
        )

    def test_generate_route_does_not_bind_pair_source_context(self) -> None:
        runtime = _bicodec_runtime(AudioSequenceLayout.FLATTENED)
        parsed = parse_task_sample(_pair_sample(), Task.TTS, cast(object, runtime))
        self.assertIsNone(parsed.audio_context)

    def test_asr_with_reuse_route_does_not_put_context_in_batch(self) -> None:
        runtime = _bicodec_runtime(AudioSequenceLayout.SEMANTIC)
        parsed = parse_task_sample(_pair_sample(), Task.ASR, cast(object, runtime))
        self.assertIsInstance(parsed.audio_context, Speech)
        built = build_task_sample(parsed, cast(object, runtime))
        self.assertIsNone(built.request["audio_context"])

    def test_audio_context_sample_wins_over_pair_source(self) -> None:
        runtime = _bicodec_runtime(AudioSequenceLayout.SEMANTIC)
        pair = _pair_sample()
        context_cell = {
            (Role.DEFAULT, Modality.AUDIO): _audio(99),
            (Role.DEFAULT, Modality.TEXT): TextItem(
                views={TextView.TEXT: "context text"},
                meta={TextMeta.LANG: Lang.EN},
            ),
        }
        wrapped = AudioContextSample(sample=pair, audio_context=context_cell)
        parsed = parse_task_sample(wrapped, Task.TTS, cast(object, runtime))
        self.assertIsInstance(parsed.audio_context, Speech)
        assert isinstance(parsed.audio_context, Speech)
        torch.testing.assert_close(
            parsed.audio_context.acoustic_codes,
            context_cell[(Role.DEFAULT, Modality.AUDIO)].views[AudioView.BICODEC][
                "acoustic"
            ],
        )

    def test_reuse_route_fails_without_source_audio(self) -> None:
        runtime = _bicodec_runtime(AudioSequenceLayout.SEMANTIC)
        sample = {
            (Role.TARGET, Modality.AUDIO): _audio(1),
            (Role.TARGET, Modality.TEXT): TextItem(
                views={TextView.TEXT: "target"},
                meta={TextMeta.LANG: Lang.EN},
            ),
        }
        with self.assertRaisesRegex(ValueError, "missing source/audio"):
            parse_task_sample(sample, Task.TTS, cast(object, runtime))


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
    acoustic = torch.stack(
        [
            (torch.arange(unit_length) + offset) % 3,
            (torch.arange(unit_length) + offset + 1) % 3,
        ],
        dim=-1,
    )
    return AudioItem(
        views={AudioView.BICODEC: {"semantic": semantic, "acoustic": acoustic}},
        meta={AudioMeta.DURATION: frames / 50.0},
    )


def _bicodec_runtime(audio_sequence_layout: AudioSequenceLayout):
    tokenizer = BiCodecAudioTokenizer(
        semantic_codebook_size=8,
        acoustic_codebook_sizes=(3, 3),
        acoustic_unit_length=2,
    )
    audio_start = 10
    boa = audio_start + tokenizer.vocab_size
    runtime = SimpleNamespace(
        codec_name="bicodec",
        audio_view=AudioView.BICODEC,
        codec_frame_rate=50.0,
        audio_sequence_layout=audio_sequence_layout,
        semantic_codec_artifact=None,
        acoustic_layout=AcousticLayout.FIXED_LENGTH,
        acoustic_unit_length=2,
        text_tokenizer=_ChatTokenizer(),
        audio_tokenizer=tokenizer,
        layout=Layout(text=(0, 10), audio=(audio_start, boa + 3)),
        pad_token_id=0,
        eos_token_id=1,
        boa_token_id=boa,
        eoa_token_id=boa + 1,
        mask_token_id=boa + 2,
    )
    return runtime


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


def _prompt_acoustic_ids(input_ids: torch.Tensor, runtime) -> torch.Tensor:
    row = input_ids
    boa_positions = (row == runtime.boa_token_id).nonzero(as_tuple=False).flatten()
    if boa_positions.numel() < 1:
        raise AssertionError("expected BOA-wrapped reference prompt in TTS input")
    prompt_boa = int(boa_positions[0].item())
    prompt_eoa = int(
        (row[prompt_boa + 1 :] == runtime.eoa_token_id)
        .nonzero(as_tuple=False)[0]
        .item()
    ) + prompt_boa + 1
    return row[prompt_boa + 1 : prompt_eoa] - runtime.layout.blocks["audio"][0]


if __name__ == "__main__":
    unittest.main()
