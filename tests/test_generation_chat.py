from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast

import torch
from anydataset.types import AudioView, Modality
from anytrain.codec import SemanticGlobalCodes
from anytrain.module.idspace import Layout
from torch import Tensor, nn

from speech_to_speech.audio import AudioCodes, AudioStream
from speech_to_speech.generation.chat import (
    ChatRequest,
    completion_from_result,
    create,
    materialize_codes,
    to_request,
)
from speech_to_speech.generation.contract import TokenGenerator
from speech_to_speech.generation.request import validate
from speech_to_speech.model.generation import GenerationStepResult
from speech_to_speech.runtime import AudioSequenceLayout
from speech_to_speech.runtime.audio_schema import AudioTokenSpec
from speech_to_speech.runtime.audio_tokenizer import (
    BiCodecAudioTokenizer,
    NativeAudioTokenizer,
)
from speech_to_speech.runtime.protocol import GenerationRuntime
from speech_to_speech.runtime.backbone.contract import Backbone
from speech_to_speech.task import (
    ControlToken,
    DIRECT,
    FULL_COT,
    TARGET_COT,
    PredictionModality,
    Task,
)


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


class _GlobalCodec:
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_codebook = torch.zeros(8, 4)
    semantic_codebook_sizes = (8,)
    global_codebook_sizes = (3,)
    global_unit_length = 2
    global_feature_dim = 4

    def tokenize(self, audio: Tensor, sample_rate: int) -> SemanticGlobalCodes:
        del sample_rate
        if audio.dim() != 3:
            raise ValueError("expected batched waveform")
        frames = max(audio.size(-1) // 2, 1)
        return SemanticGlobalCodes(
            semantic=torch.arange(frames, dtype=torch.long).view(1, frames, 1),
            global_codes=torch.zeros(1, 2, 1, dtype=torch.long),
        )

    def detokenize(self, codes: object) -> Tensor:
        if not isinstance(codes, SemanticGlobalCodes):
            raise TypeError("codes must be SemanticGlobalCodes")
        return codes.semantic[..., 0].float()

    def global_codes_to_features(self, global_codes: Tensor) -> Tensor:
        return global_codes.float()

    def decode_features(
        self,
        semantic_codes: Tensor,
        global_features: Tensor,
    ) -> Tensor:
        del global_features
        return semantic_codes[..., 0].float()


class _FrameTokenizerBackend:
    sample_rate = 16_000
    frame_rate = 12.5
    codebook_sizes = (16,)

    def __init__(self) -> None:
        self.calls: list[tuple[Tensor, int]] = []

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        self.calls.append((audio.detach().clone(), sample_rate))
        return torch.tensor(
            [[[1], [2], [3]]],
            dtype=torch.long,
            device=audio.device,
        )


class _BackendLoader:
    def __init__(self, backend: object, *, forbidden: bool = False) -> None:
        self.backend = backend
        self.forbidden = forbidden
        self.loads = 0

    def load(self) -> object:
        self.loads += 1
        if self.forbidden:
            raise AssertionError("unexpected audio tokenizer backend load")
        return self.backend


class _LazyBackendRuntime:
    def __init__(
        self,
        runtime: SimpleNamespace,
        *,
        input_loader: _BackendLoader,
        output_loader: _BackendLoader,
    ) -> None:
        self._runtime = runtime
        self._input_loader = input_loader
        self._output_loader = output_loader

    def __getattr__(self, name: str):
        return getattr(self._runtime, name)

    @property
    def input_audio_tokenizer_backend(self) -> object:
        return self._input_loader.load()

    @property
    def output_audio_tokenizer_backend(self) -> object:
        return self._output_loader.load()

    @property
    def input_codec(self) -> object:
        raise AssertionError("chat input must not use the deprecated input_codec alias")

    @property
    def codec(self) -> object:
        raise AssertionError("chat input must not load the output codec alias")


def _codes() -> AudioCodes:
    return AudioCodes(
        semantic_codes=torch.tensor([[1], [2], [3]], dtype=torch.int32),
        global_codes=torch.tensor([[0], [1]], dtype=torch.int64),
    )


def _runtime(
    *,
    audio_sequence_layout: AudioSequenceLayout = AudioSequenceLayout.FLATTENED,
    codec_name: str = "bicodec",
    codec: object | None = None,
    decoupled: bool = False,
) -> GenerationRuntime:
    tokenizer = BiCodecAudioTokenizer(
        semantic_codebook_size=8,
        global_codebook_sizes=(3,),
        global_unit_length=2,
    )
    input_tokenizer = NativeAudioTokenizer(vocab_size=16)
    input_size = input_tokenizer.vocab_size + 3 if decoupled else 0
    lexical_text_vocab_size = 8
    control_token_ids = tuple(range(8, 14))
    text_vocab_size = lexical_text_vocab_size + len(control_token_ids)
    audio_start = text_vocab_size + input_size
    layout = (
        Layout(
            text=(0, text_vocab_size),
            audio_input=(text_vocab_size, audio_start),
            audio=(audio_start, audio_start + tokenizer.vocab_size + 4),
        )
        if decoupled
        else Layout(
            text=(0, text_vocab_size),
            audio=(audio_start, audio_start + tokenizer.vocab_size + 4),
        )
    )
    def control_token_id(token: ControlToken) -> int:
        return control_token_ids[list(ControlToken).index(token)]

    def generation_allowed_ids(modality: Modality) -> tuple[int, ...]:
        if modality is not Modality.TEXT:
            raise ValueError("chat test runtime only exposes lexical text ids.")
        return tuple(range(1, lexical_text_vocab_size))

    output_spec = AudioTokenSpec.create(
        codec_name=codec_name,
        sequence_layout=audio_sequence_layout.value,
        tokenizer=tokenizer,
    )
    input_spec = (
        AudioTokenSpec.create(
            codec_name="glm4",
            sequence_layout=audio_sequence_layout.value,
            tokenizer=input_tokenizer,
        )
        if decoupled
        else output_spec
    )
    runtime = SimpleNamespace(
        audio_sequence_layout=audio_sequence_layout,
        audio_tokenizer=tokenizer,
        output_audio_tokenizer=tokenizer,
        input_audio_tokenizer=(input_tokenizer if decoupled else tokenizer),
        output_audio_token_spec=output_spec,
        input_audio_token_spec=input_spec,
        text_tokenizer=_TextTokenizer(),
        layout=layout,
        lexical_text_vocab_size=lexical_text_vocab_size,
        control_token_ids=control_token_ids,
        control_token_id=control_token_id,
        generation_allowed_ids=generation_allowed_ids,
        boa_token_id=audio_start + tokenizer.vocab_size,
        eoa_token_id=audio_start + tokenizer.vocab_size + 1,
        mask_token_id=audio_start + tokenizer.vocab_size + 2,
        audio_schema_token_id=audio_start + tokenizer.vocab_size + 3,
        input_boa_token_id=(text_vocab_size + input_tokenizer.vocab_size if decoupled else audio_start + tokenizer.vocab_size),
        input_eoa_token_id=(text_vocab_size + input_tokenizer.vocab_size + 1 if decoupled else audio_start + tokenizer.vocab_size + 1),
        input_audio_schema_token_id=(text_vocab_size + input_tokenizer.vocab_size + 2 if decoupled else audio_start + tokenizer.vocab_size + 3),
        input_audio_block_name=("audio_input" if decoupled else "audio"),
        input_audio_decoupled=decoupled,
        input_codec_name=("glm4" if decoupled else codec_name),
        input_audio_view=(AudioView.GLM4 if decoupled else AudioView.BICODEC),
        eos_token_id=7,
        pad_token_id=0,
        codec_audio_range=(audio_start, audio_start + tokenizer.vocab_size),
        input_codec_audio_range=(
            (text_vocab_size, text_vocab_size + input_tokenizer.vocab_size)
            if decoupled
            else (audio_start, audio_start + tokenizer.vocab_size)
        ),
        structured_full_sequence=True,
        acoustic_side_channel=False,
        codec_name=codec_name,
        audio_view=AudioView.BICODEC,
        output_audio_tokenizer_backend=codec,
        input_audio_tokenizer_backend=codec,
        codec=codec,
    )
    return cast(GenerationRuntime, runtime)


class _RouteModel:
    def __init__(
        self,
        runtime: GenerationRuntime,
        output: AudioCodes,
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
            (AudioStream.GLOBAL, AudioStream.SEMANTIC)
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
        self._step = 0

    def generation_step(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor,
        output_hidden_states: bool,
        token_ids: Tensor | None,
        token_kind: str | None = None,
        modality: Modality | None,
        past_key_values=None,
        use_cache: bool = False,
        audio_input_positions: Tensor | None = None,
        audio_head_past: object | None = None,
        input_modalities: frozenset[Modality] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
    ) -> GenerationStepResult:
        del (
            attention_mask,
            token_kind,
            modality,
            past_key_values,
            audio_input_positions,
            audio_head_past,
            input_modalities,
            validate_input,
            validate_audio_input_positions,
        )
        script = [
            self.runtime.control_token_id(ControlToken.MT_BEGIN),
            self.runtime.control_token_id(ControlToken.LANG_EN),
            4,
            5,
            self.runtime.control_token_id(ControlToken.MT_END),
        ]
        next_id = script[self._step]
        self._step += 1
        logits = torch.full(
            (*input_ids.shape, self.runtime.layout.vocab_size),
            float("-inf"),
        )
        logits[:, -1, next_id] = 0.0
        if token_ids is not None:
            logits = logits.index_select(-1, token_ids)
        return GenerationStepResult(
            logits=logits,
            past_key_values=SimpleNamespace() if use_cache else None,
            audio_head_past=None,
            hidden_states=(torch.zeros(*input_ids.shape, 1),)
            if output_hidden_states
            else None,
        )

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
    def test_decoupled_audio_source_uses_input_codes_and_output_boa(self) -> None:
        runtime = _runtime(decoupled=True)
        private = to_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "codec_codes",
                                "codec": "glm4",
                                "codes": torch.tensor([[1], [2], [3]]),
                            },
                        ],
                    }
                ],
                "task": Task.S2ST,
            },
            runtime,
        )

        positions = private["audio_input_positions"]
        self.assertIsNotNone(positions)
        assert positions is not None
        prompt = private["prompt_ids"]
        selected = prompt.index_select(0, positions)
        input_start, input_end = runtime.input_codec_audio_range
        self.assertTrue(bool(selected.ge(input_start).all()))
        self.assertTrue(bool(selected.lt(input_end).all()))
        self.assertNotEqual(int(prompt[-1]), runtime.boa_token_id)
        self.assertIn(runtime.input_boa_token_id, prompt.tolist())
        self.assertIn(runtime.input_eoa_token_id, prompt.tolist())
        self.assertNotEqual(int(prompt[-1]), runtime.input_eoa_token_id)

    def test_voice_clone_chat_places_target_text_before_source_audio(self) -> None:
        runtime = _runtime(decoupled=True)
        private = to_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello world"},
                            {
                                "type": "codec_codes",
                                "codec": "glm4",
                                "codes": torch.tensor([[1], [2], [3]]),
                            },
                        ],
                    }
                ],
                "task": Task.TTS_VOICE_CLONE,
            },
            runtime,
        )

        self.assertIs(private["task"], Task.TTS_VOICE_CLONE)
        positions = private["audio_input_positions"]
        self.assertIsNotNone(positions)
        assert positions is not None
        prompt = private["prompt_ids"]
        input_boa = (prompt == runtime.input_boa_token_id).nonzero().flatten()
        self.assertEqual(input_boa.numel(), 1)
        boundary = int(input_boa[0])
        torch.testing.assert_close(prompt[boundary - 2 : boundary], torch.tensor([2, 3]))
        selected = prompt.index_select(0, positions)
        input_start, input_end = runtime.input_codec_audio_range
        self.assertTrue(bool(selected.ge(input_start).all()))
        self.assertTrue(bool(selected.lt(input_end).all()))
        validate(private, cast(TokenGenerator, _RouteModel(runtime, _codes())))

    def test_decoupled_audio_source_rejects_output_codec_codes(self) -> None:
        runtime = _runtime(decoupled=True)
        base = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "codec_codes",
                            "codec": "bicodec",
                            "codes": torch.tensor([[1]]),
                        },
                    ],
                }
            ],
            "task": Task.S2ST,
        }
        with self.assertRaisesRegex(ValueError, "runtime input codec"):
            to_request(cast(ChatRequest, base), runtime)

    def test_decoupled_waveform_loads_only_input_frame_tokenizer(self) -> None:
        input_backend = _FrameTokenizerBackend()
        input_loader = _BackendLoader(input_backend)
        output_loader = _BackendLoader(_GlobalCodec(), forbidden=True)
        base = cast(SimpleNamespace, _runtime(decoupled=True))
        runtime = cast(
            GenerationRuntime,
            _LazyBackendRuntime(
                base,
                input_loader=input_loader,
                output_loader=output_loader,
            ),
        )

        private = to_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello"},
                            {
                                "type": "audio",
                                "waveform": torch.zeros(8),
                                "sample_rate": 16_000,
                            },
                        ],
                    }
                ],
                "task": Task.S2ST,
            },
            runtime,
        )

        self.assertEqual(input_loader.loads, 1)
        self.assertEqual(output_loader.loads, 0)
        self.assertEqual(len(input_backend.calls), 1)
        encoded_waveform, sample_rate = input_backend.calls[0]
        self.assertEqual(tuple(encoded_waveform.shape), (1, 1, 8))
        self.assertEqual(encoded_waveform.dtype, torch.float32)
        self.assertEqual(sample_rate, 16_000)
        positions = private["audio_input_positions"]
        self.assertIsNotNone(positions)
        assert positions is not None
        torch.testing.assert_close(
            private["prompt_ids"].index_select(0, positions),
            runtime.layout.to_global(
                runtime.input_audio_block_name,
                torch.tensor([1, 2, 3]),
            )
        )

    def test_text_only_messages_build_private_request(self) -> None:
        runtime = _runtime(audio_sequence_layout=AudioSequenceLayout.FLATTENED)
        request: ChatRequest = {
            "messages": [{"role": "user", "content": "hello world"}],
            "task": Task.T2TT,
            "language": "English",
        }
        private = to_request(request, runtime)
        self.assertIs(private["task"], Task.T2TT)
        self.assertNotIn("prediction", private)
        self.assertEqual(private["trace"], DIRECT)
        self.assertEqual(private["target_language"], "en")
        self.assertFalse(
            any(
                token_id in private["prompt_ids"].tolist()
                for token_id in runtime.control_token_ids
            )
        )
        self.assertNotIn("audio_context", private)
        self.assertGreater(private["prompt_ids"].numel(), 0)

    def test_explicit_full_trace_is_normalized_and_added_to_prompt(self) -> None:
        runtime = _runtime(audio_sequence_layout=AudioSequenceLayout.FLATTENED)
        private = to_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello world"},
                            {
                                "type": "codec_codes",
                                "codec": "bicodec",
                                "codes": _codes(),
                            },
                        ],
                    }
                ],
                "task": Task.S2TT,
                "trace": FULL_COT,
            },
            runtime,
        )

        self.assertNotIn("prediction", private)
        self.assertEqual(private["trace"], FULL_COT)
        self.assertNotIn(
            runtime.control_token_id(ControlToken.ASR_BEGIN),
            private["prompt_ids"].tolist(),
        )
        tokenizer = cast(_TextTokenizer, runtime.text_tokenizer)
        self.assertIn(
            "Respond in this exact order",
            tokenizer.conversations[0][-1]["content"],
        )

    def test_asr_does_not_bind_non_mt_language_to_control_vocabulary(self) -> None:
        runtime = _runtime(audio_sequence_layout=AudioSequenceLayout.FLATTENED)
        private = to_request(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "codec_codes",
                                "codec": "bicodec",
                                "codes": _codes(),
                            }
                        ],
                    }
                ],
                "task": Task.ASR,
                "language": "Esperanto",
            },
            runtime,
        )

        self.assertNotIn("target_language", private)
        self.assertNotIn(
            runtime.control_token_id(ControlToken.ASR_BEGIN),
            private["prompt_ids"].tolist(),
        )

    def test_prediction_override_is_rejected(self) -> None:
        runtime = _runtime(audio_sequence_layout=AudioSequenceLayout.FLATTENED)
        request = cast(
            ChatRequest,
            {
                "messages": [{"role": "user", "content": "hello world"}],
                "task": Task.T2TT,
                "prediction": PredictionModality.TEXT,
            },
        )

        with self.assertRaisesRegex(ValueError, "prediction override"):
            to_request(request, runtime)

    def test_bicodec_rejects_mixed_response_trace(self) -> None:
        runtime = _runtime()
        with self.assertRaisesRegex(
            ValueError,
            "BiCodec chat does not support mixed response traces",
        ):
            to_request(
                {
                    "messages": [{"role": "user", "content": "hello world"}],
                    "task": Task.T2ST,
                    "trace": TARGET_COT,
                },
                runtime,
            )

        with self.assertRaisesRegex(
            ValueError,
            "BiCodec chat does not support mixed response traces",
        ):
            to_request(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "hello world"},
                                {
                                    "type": "codec_codes",
                                    "codec": "bicodec",
                                    "codes": _codes(),
                                },
                            ],
                        }
                    ],
                    "task": Task.S2ST,
                    "trace": FULL_COT,
                },
                runtime,
            )

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

    def test_plain_tts_rejects_reference_codec_codes(self) -> None:
        runtime = _runtime()
        request: ChatRequest = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello world"},
                        {
                            "type": "codec_codes",
                            "codec": "bicodec",
                            "codes": _codes(),
                        },
                    ],
                }
            ],
            "task": Task.TTS,
            "language": "Chinese",
        }
        with self.assertRaisesRegex(ValueError, "audio/codec_codes are not supported"):
            to_request(request, runtime)

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
        codec = _GlobalCodec()
        input_loader = _BackendLoader(object(), forbidden=True)
        output_loader = _BackendLoader(codec)
        runtime = cast(
            GenerationRuntime,
            _LazyBackendRuntime(
                cast(SimpleNamespace, _runtime()),
                input_loader=input_loader,
                output_loader=output_loader,
            ),
        )
        waveform = torch.zeros(8, dtype=torch.float32)
        codes = materialize_codes(
            {
                "type": "audio",
                "waveform": waveform,
                "sample_rate": 16_000,
            },
            runtime,
        )
        self.assertEqual(input_loader.loads, 0)
        self.assertEqual(output_loader.loads, 1)
        self.assertIsInstance(codes, AudioCodes)
        assert isinstance(codes, AudioCodes)
        self.assertEqual(codes.semantic_codes.size(1), 1)
        self.assertIsNotNone(codes.global_codes)
        assert codes.global_codes is not None
        self.assertEqual(codes.global_codes.size(0), 2)

    def test_create_returns_completion_for_text_task(self) -> None:
        runtime = _runtime(audio_sequence_layout=AudioSequenceLayout.FLATTENED)
        model = cast(TokenGenerator, _TextModel(runtime))
        completion = create(
            {
                "messages": [{"role": "user", "content": "hello world"}],
                "task": Task.T2TT,
            },
            model,
            max_new_tokens=6,
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
            (
                Task.T2TT,
                [
                    runtime.control_token_id(ControlToken.MT_BEGIN),
                    runtime.control_token_id(ControlToken.LANG_EN),
                    4,
                    5,
                    runtime.control_token_id(ControlToken.MT_END),
                ],
                "decoded:4,5",
            ),
            (
                Task.PARALLEL_AR,
                [
                    4,
                    runtime.eos_token_id,
                    runtime.boa_token_id,
                    runtime.audio_schema_token_id,
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
                    runtime.audio_schema_token_id,
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
                self.assertNotIn(
                    "trace",
                    completion["choices"][0]["message"],
                )

    def test_completion_returns_target_text_and_structured_trace(self) -> None:
        runtime = _runtime(audio_sequence_layout=AudioSequenceLayout.FLATTENED)
        audio_start, _ = runtime.codec_audio_range
        completion = completion_from_result(
            {
                "response_ids": torch.tensor(
                    [
                        runtime.control_token_id(ControlToken.ASR_BEGIN),
                        4,
                        runtime.control_token_id(ControlToken.ASR_END),
                        runtime.control_token_id(ControlToken.MT_BEGIN),
                        runtime.control_token_id(ControlToken.LANG_EN),
                        5,
                        runtime.control_token_id(ControlToken.MT_END),
                        runtime.boa_token_id,
                        runtime.audio_schema_token_id,
                        audio_start,
                        runtime.eoa_token_id,
                    ]
                ),
                "audio": None,
            },
            {
                "messages": [{"role": "user", "content": "hello world"}],
                "task": Task.S2ST,
                "trace": FULL_COT,
            },
            runtime,
        )

        message = completion["choices"][0]["message"]
        self.assertEqual(message["content"], "decoded:5")
        self.assertEqual(
            message["trace"],
            [
                {
                    "index": 0,
                    "role": "source",
                    "modality": "text",
                    "content": "decoded:4",
                }
            ],
        )

    def test_completion_preserves_audio_decode_error(self) -> None:
        runtime = _runtime(audio_sequence_layout=AudioSequenceLayout.FLATTENED)
        decode_error = {
            "type": "ValueError",
            "message": "invalid generated audio",
        }

        completion = completion_from_result(
            {
                "response_ids": torch.tensor(
                    [
                        runtime.control_token_id(ControlToken.MT_BEGIN),
                        runtime.control_token_id(ControlToken.LANG_EN),
                        4,
                        5,
                        runtime.control_token_id(ControlToken.MT_END),
                    ]
                ),
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

    def test_asr_completion_does_not_normalize_non_mt_language(self) -> None:
        runtime = _runtime(audio_sequence_layout=AudioSequenceLayout.FLATTENED)
        completion = completion_from_result(
            {
                "response_ids": torch.tensor(
                    [
                        runtime.control_token_id(ControlToken.ASR_BEGIN),
                        4,
                        runtime.control_token_id(ControlToken.ASR_END),
                    ]
                ),
                "audio": None,
            },
            {
                "messages": [{"role": "user", "content": "audio"}],
                "task": Task.ASR,
                "language": "Esperanto",
            },
            runtime,
        )

        self.assertEqual(
            completion["choices"][0]["message"]["content"],
            "decoded:4",
        )


if __name__ == "__main__":
    unittest.main()
