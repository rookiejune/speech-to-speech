from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast

import torch
from anydataset.types import AudioView, Modality
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from anytrain.module.idspace import Layout
from torch import Tensor, nn

from speech_to_speech.audio_stream import AudioStream
from speech_to_speech.generation.chat import (
    ChatRequest,
    completion_from_result,
    create,
    materialize_codes,
    to_request,
)
from speech_to_speech.generation.protocol import TokenGenerator
from speech_to_speech.runtime import AudioSequenceLayout
from speech_to_speech.runtime.audio_tokenizer import BiCodecAudioTokenizer
from speech_to_speech.runtime.protocol import GenerationRuntime
from speech_to_speech.runtime.types import Backbone
from speech_to_speech.task import Task


class _TextTokenizer:
    def __init__(self) -> None:
        self.conversations: list[list[dict[str, str]]] = []
        self.encoded: list[str] = []

    def apply_chat_template(self, conversation, **kwargs) -> str:
        del kwargs
        normalized = [dict(message) for message in conversation]
        self.conversations.append(normalized)
        body = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in normalized
        )
        return f"{body}<assistant>"

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        self.encoded.append(text)
        if "hello world" in text:
            return [2, 3]
        if "hello" in text:
            return [2]
        return [1]

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        values = [int(value) for value in token_ids]
        if skip_special_tokens:
            values = [value for value in values if value != 7]
        return "decoded:" + ",".join(str(value) for value in values)


class _StructuredCodec:
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_codebook = torch.zeros(8, 4)
    semantic_codebook_sizes = (8,)
    acoustic_codebook_sizes = (3,)
    acoustic_layout = AcousticLayout.FIXED_LENGTH
    acoustic_unit_length = 2
    acoustic_feature_dim = 4

    def tokenize(self, audio: Tensor, sample_rate: int) -> SemanticAcousticCodes:
        del sample_rate
        if audio.dim() != 3:
            raise ValueError("expected batched waveform")
        frames = max(audio.size(-1) // 2, 1)
        return SemanticAcousticCodes(
            semantic=torch.arange(frames, dtype=torch.long).view(1, frames, 1),
            acoustic=torch.zeros(1, 2, 1, dtype=torch.long),
        )

    def detokenize(self, codes: object) -> Tensor:
        if not isinstance(codes, SemanticAcousticCodes):
            raise TypeError("codes must be SemanticAcousticCodes")
        return codes.semantic[..., 0].float()

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        return acoustic_codes.float()

    def decode_features(
        self,
        semantic_codes: Tensor,
        acoustic_features: Tensor,
    ) -> Tensor:
        del acoustic_features
        return semantic_codes[..., 0].float()


def _codes() -> SemanticAcousticCodes:
    return SemanticAcousticCodes(
        semantic=torch.tensor([[1], [2], [3]], dtype=torch.int32),
        acoustic=torch.tensor([[0], [1]], dtype=torch.int64),
    )


def _runtime(
    *,
    audio_sequence_layout: AudioSequenceLayout = AudioSequenceLayout.SEMANTIC,
    codec_name: str = "bicodec",
    codec: object | None = None,
) -> GenerationRuntime:
    tokenizer = BiCodecAudioTokenizer(
        semantic_vocab_size=8,
        acoustic_codebook_sizes=(3,),
        acoustic_unit_length=2,
    )
    runtime = SimpleNamespace(
        audio_sequence_layout=audio_sequence_layout,
        audio_tokenizer=tokenizer,
        text_tokenizer=_TextTokenizer(),
        layout=Layout(
            text=(0, 8),
            audio=(8, 8 + tokenizer.vocab_size + 3),
        ),
        boa_token_id=8 + tokenizer.vocab_size,
        eoa_token_id=8 + tokenizer.vocab_size + 1,
        eos_token_id=7,
        pad_token_id=0,
        codec_audio_range=(8, 8 + tokenizer.vocab_size),
        structured_full_sequence=True,
        acoustic_side_channel=False,
        codec_name=codec_name,
        audio_view=AudioView.BICODEC,
        codec=codec,
    )
    return cast(GenerationRuntime, runtime)


class _RouteModel:
    def __init__(
        self,
        runtime: GenerationRuntime,
        output: SemanticAcousticCodes,
    ) -> None:
        self.runtime = runtime
        embedding = nn.Embedding(1, 1)
        self.backbone = cast(
            Backbone,
            cast(
                object,
                SimpleNamespace(get_input_embeddings=lambda: embedding),
            ),
        )
        streams = (
            (AudioStream.ACOUSTIC, AudioStream.SEMANTIC)
            if runtime.audio_sequence_layout is AudioSequenceLayout.FLATTENED
            else (AudioStream.SEMANTIC,)
        )
        local_ids = runtime.audio_tokenizer.encode_streams(
            output,
            streams,
        )
        self.response = torch.cat(
            (
                runtime.layout.to_global(Modality.AUDIO.value, local_ids),
                torch.tensor([runtime.eoa_token_id]),
            )
        )
        self.audio_token_frame_spans = torch.ones(
            runtime.audio_tokenizer.vocab_size,
            dtype=torch.long,
        )

    def generation_step(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("structured route should use generate_tokens")

    def select_audio_head_cache(self, past_key_values, indices):
        del past_key_values, indices
        return None

    def generate_tokens(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        prompt_attention_mask: Tensor | None = None,
        audio_input_positions: Tensor | None = None,
        stop_token_id: int | None = None,
        generation_modality: Modality | None = None,
        allowed_token_ids=None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> Tensor:
        del (
            max_new_tokens,
            temperature,
            top_p,
            prompt_attention_mask,
            audio_input_positions,
            stop_token_id,
            generation_modality,
            allowed_token_ids,
            do_sample,
            use_cache,
        )
        response = self.response.to(device=prompt_ids.device).expand(
            prompt_ids.size(0),
            -1,
        )
        return torch.cat((prompt_ids, response), dim=1)


class _TextModel:
    def __init__(self, runtime: GenerationRuntime) -> None:
        self.runtime = runtime
        embedding = nn.Embedding(1, 1)
        self.backbone = cast(
            Backbone,
            cast(
                object,
                SimpleNamespace(get_input_embeddings=lambda: embedding),
            ),
        )
        self.audio_token_frame_spans = torch.ones(1, dtype=torch.long)

    def generation_step(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("text model should use generate_tokens")

    def select_audio_head_cache(self, past_key_values, indices):
        del past_key_values, indices
        return None

    def generate_tokens(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        prompt_attention_mask: Tensor | None = None,
        audio_input_positions: Tensor | None = None,
        stop_token_id: int | None = None,
        generation_modality: Modality | None = None,
        allowed_token_ids=None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> Tensor:
        del (
            max_new_tokens,
            temperature,
            top_p,
            prompt_attention_mask,
            audio_input_positions,
            generation_modality,
            allowed_token_ids,
            do_sample,
            use_cache,
        )
        eos = (
            prompt_ids.new_tensor([stop_token_id])
            if stop_token_id is not None
            else prompt_ids.new_tensor([7])
        )
        response = prompt_ids.new_tensor([[4, 5]])
        return torch.cat((prompt_ids, response.expand(prompt_ids.size(0), -1), eos.expand(prompt_ids.size(0), 1)), dim=1)


class ChatAdapterTest(unittest.TestCase):
    def test_text_only_messages_build_private_request(self) -> None:
        runtime = _runtime(audio_sequence_layout=AudioSequenceLayout.FLATTENED)
        request: ChatRequest = {
            "messages": [{"role": "user", "content": "hello world"}],
            "task": Task.T2TT,
            "language": "English",
        }
        private = to_request(request, runtime)
        self.assertIs(private["task"], Task.T2TT)
        self.assertIsNone(private["audio_context"])
        self.assertGreater(private["prompt_ids"].numel(), 0)

    def test_messages_history_is_preserved_and_encoded_once(self) -> None:
        runtime = _runtime(audio_sequence_layout=AudioSequenceLayout.FLATTENED)
        request: ChatRequest = {
            "messages": [
                {"role": "system", "content": "Use terse wording."},
                {"role": "user", "content": "Earlier turn."},
                {"role": "assistant", "content": "Acknowledged."},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hello world"}],
                },
            ],
            "task": Task.T2TT,
            "language": "English",
        }

        to_request(request, runtime)

        tokenizer = cast(_TextTokenizer, runtime.text_tokenizer)
        self.assertEqual(len(tokenizer.conversations), 1)
        conversation = tokenizer.conversations[0]
        self.assertEqual(
            [message["role"] for message in conversation],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(conversation[0]["content"], "Use terse wording.")
        self.assertEqual(conversation[1]["content"], "Earlier turn.")
        self.assertEqual(conversation[2]["content"], "Acknowledged.")
        self.assertIn("hello world", conversation[3]["content"])
        self.assertNotEqual(conversation[3]["content"], "hello world")
        self.assertEqual(len(tokenizer.encoded), 1)
        self.assertIn("<system>Use terse wording.</system>", tokenizer.encoded[0])
        self.assertIn("<assistant>Acknowledged.</assistant>", tokenizer.encoded[0])

    def test_codec_codes_passthrough_matches_bicodec_builder(self) -> None:
        runtime = _runtime()
        codes = _codes()
        request: ChatRequest = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello world"},
                        {
                            "type": "codec_codes",
                            "codec": "bicodec",
                            "codes": codes,
                        },
                    ],
                }
            ],
            "task": Task.TTS,
            "language": "Chinese",
        }
        private = to_request(request, runtime)
        self.assertIs(private["audio_context"], codes)
        self.assertIs(private["task"], Task.TTS)

    def test_codec_name_mismatch_is_explicit(self) -> None:
        runtime = _runtime()
        with self.assertRaisesRegex(ValueError, "does not match runtime codec"):
            to_request(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "hello"},
                                {
                                    "type": "codec_codes",
                                    "codec": "longcat",
                                    "codes": _codes(),
                                },
                            ],
                        }
                    ],
                    "task": Task.TTS,
                },
                runtime,
            )

    def test_audio_part_encodes_with_structured_codec(self) -> None:
        codec = _StructuredCodec()
        runtime = _runtime(codec=codec)
        waveform = torch.zeros(8, dtype=torch.float32)
        codes = materialize_codes(
            {
                "type": "audio",
                "waveform": waveform,
                "sample_rate": 16_000,
            },
            runtime,
        )
        self.assertIsInstance(codes, SemanticAcousticCodes)
        assert isinstance(codes, SemanticAcousticCodes)
        self.assertEqual(codes.semantic.size(1), 1)
        self.assertEqual(codes.acoustic.size(0), 2)

    def test_create_returns_completion_for_text_task(self) -> None:
        runtime = _runtime(audio_sequence_layout=AudioSequenceLayout.FLATTENED)
        model = cast(TokenGenerator, _TextModel(runtime))
        completion = create(
            {
                "messages": [{"role": "user", "content": "hello world"}],
                "task": Task.T2TT,
            },
            model,
            max_new_tokens=4,
            do_sample=False,
        )
        self.assertEqual(len(completion["choices"]), 1)
        message = completion["choices"][0]["message"]
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "decoded:4,5")
        self.assertIsNone(message["audio"])

    def test_completion_projects_text_from_text_and_mixed_responses(self) -> None:
        runtime = _runtime(audio_sequence_layout=AudioSequenceLayout.FLATTENED)
        audio_start, _ = runtime.codec_audio_range
        cases = (
            (Task.T2TT, [4, 5], "decoded:4,5"),
            (
                Task.PARALLEL_AR,
                [
                    4,
                    runtime.eos_token_id,
                    runtime.boa_token_id,
                    audio_start,
                    runtime.eoa_token_id,
                ],
                "decoded:4",
            ),
            (
                Task.INTERLEAVED_AR,
                [
                    4,
                    runtime.boa_token_id,
                    audio_start,
                    runtime.eoa_token_id,
                    5,
                    runtime.eos_token_id,
                ],
                "decoded:4,5",
            ),
        )

        for task, response_ids, expected in cases:
            with self.subTest(task=task):
                completion = completion_from_result(
                    {
                        "response_ids": torch.tensor(response_ids),
                        "audio": None,
                    },
                    {
                        "messages": [{"role": "user", "content": "hello world"}],
                        "task": task,
                    },
                    runtime,
                )

                self.assertEqual(
                    completion["choices"][0]["message"]["content"],
                    expected,
                )

    def test_completion_preserves_audio_decode_error(self) -> None:
        runtime = _runtime(audio_sequence_layout=AudioSequenceLayout.FLATTENED)
        decode_error = {
            "type": "ValueError",
            "message": "invalid generated audio",
        }

        completion = completion_from_result(
            {
                "response_ids": torch.tensor([4, 5]),
                "audio": None,
                "decode_error": decode_error,
            },
            {
                "messages": [{"role": "user", "content": "hello world"}],
                "task": Task.T2TT,
            },
            runtime,
        )

        self.assertEqual(
            completion["choices"][0]["message"]["decode_error"],
            decode_error,
        )


if __name__ == "__main__":
    unittest.main()
