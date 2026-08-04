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
    Speech,
)
from speech_to_speech.runtime import AudioSequenceLayout
from speech_to_speech.runtime.audio_tokenizer import BiCodecAudioTokenizer
from speech_to_speech.task import Task


class PairReferenceContextTest(unittest.TestCase):
    def test_tts_generates_global_when_audio_is_not_a_task_input(self) -> None:
        runtime = _bicodec_runtime(AudioSequenceLayout.FLATTENED)
        parsed = parse_task_sample(_pair_sample(), Task.TTS, cast(object, runtime))
        self.assertIsNone(parsed.audio_context)
        built = build_task_sample(parsed, cast(object, runtime))
        response = built.labels.response_ids
        local = response[:-1] - runtime.layout.blocks["audio"][0]
        self.assertEqual(int(local[0]), runtime.audio_tokenizer.global_token_id)
        self.assertNotIn("audio_context", built.request)

    def test_asr_does_not_materialize_decode_context(self) -> None:
        runtime = _bicodec_runtime(AudioSequenceLayout.FLATTENED)
        parsed = parse_task_sample(_pair_sample(), Task.ASR, cast(object, runtime))
        self.assertIsNone(parsed.audio_context)
        built = build_task_sample(parsed, cast(object, runtime))
        self.assertNotIn("audio_context", built.request)

    def test_s2st_reuses_visible_source_global(self) -> None:
        runtime = _bicodec_runtime(AudioSequenceLayout.FLATTENED)
        parsed = parse_task_sample(_pair_sample(), Task.S2ST, cast(object, runtime))
        built = build_task_sample(parsed, cast(object, runtime))
        response = built.labels.response_ids
        local = response[:-1] - runtime.layout.blocks["audio"][0]
        self.assertEqual(int(local[0]), runtime.audio_tokenizer.semantic_token_id)
        prompt_global = _prompt_global_ids(built.request["prompt_ids"], runtime)
        decoded = runtime.audio_tokenizer.decode_streams(prompt_global)
        self.assertIsNotNone(decoded.global_codes)

    def test_explicit_audio_context_is_serialized_into_prompt(self) -> None:
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
        parsed = parse_task_sample(wrapped, Task.TTS, cast(object, runtime))
        self.assertIsInstance(parsed.audio_context, Speech)
        assert isinstance(parsed.audio_context, Speech)
        torch.testing.assert_close(
            parsed.audio_context.global_codes,
            context_cell[(Role.DEFAULT, Modality.AUDIO)].views[AudioView.BICODEC][
                "global"
            ],
        )
        built = build_task_sample(parsed, cast(object, runtime))
        self.assertNotIn("audio_context", built.request)
        response = built.labels.response_ids
        local = response[:-1] - runtime.layout.blocks["audio"][0]
        self.assertEqual(int(local[0]), runtime.audio_tokenizer.semantic_token_id)


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
    audio_start = 10
    boa = audio_start + tokenizer.vocab_size
    runtime = SimpleNamespace(
        codec_name="bicodec",
        audio_view=AudioView.BICODEC,
        codec_frame_rate=50.0,
        audio_sequence_layout=audio_sequence_layout,
        acoustic_generator_artifact=None,
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


def _prompt_global_ids(input_ids: torch.Tensor, runtime) -> torch.Tensor:
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
