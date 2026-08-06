from __future__ import annotations

import unittest
from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import torch
from anydataset.types import Modality
from anytrain.codec import SemanticGlobalCodes
from anytrain.module.idspace import Layout
from torch import Tensor, nn

from speech_to_speech.audio import AudioCodes, AudioStream
from speech_to_speech.generation import (
    Request,
    generate_responses,
)
from speech_to_speech.generation.audio import decode_token_audio_results
from speech_to_speech.generation.request import validate
from speech_to_speech.model.generation import GenerationStepResult
from speech_to_speech.runtime import (
    AudioOutputConfig,
    AudioSequenceLayout,
    Config,
    Runtime,
)
from speech_to_speech.runtime.audio_schema import AudioTokenSpec
from speech_to_speech.runtime.audio_tokenizer import BiCodecAudioTokenizer
from speech_to_speech.runtime.protocol import GenerationRuntime
from speech_to_speech.runtime.backbone.contract import Backbone
from speech_to_speech.task import ControlToken, Task


class _TextTokenizer:
    def __init__(self) -> None:
        self.encoded: list[str] = []
        self.conversations: list[list[dict[str, str]]] = []

    def apply_chat_template(self, conversation, **kwargs) -> str:
        del kwargs
        normalized = [dict(message) for message in conversation]
        self.conversations.append(normalized)
        body = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>" for message in normalized
        )
        return f"{body}<assistant>"

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        self.encoded.append(text)
        if "hello world" in text:
            return [2, 3]
        return [1]


def _runtime() -> GenerationRuntime:
    tokenizer = BiCodecAudioTokenizer(
        semantic_codebook_size=8,
        global_codebook_sizes=(3,),
        global_unit_length=2,
    )
    lexical_text_vocab_size = 8
    control_token_ids = tuple(
        range(lexical_text_vocab_size, lexical_text_vocab_size + len(ControlToken))
    )
    audio_start = lexical_text_vocab_size + len(ControlToken)
    boa_token_id = audio_start + tokenizer.vocab_size
    spec = AudioTokenSpec.create(
        codec_name="bicodec",
        sequence_layout=AudioSequenceLayout.FLATTENED.value,
        tokenizer=tokenizer,
    )
    runtime = SimpleNamespace(
        audio_sequence_layout=AudioSequenceLayout.FLATTENED,
        audio_tokenizer=tokenizer,
        output_audio_tokenizer=tokenizer,
        input_audio_tokenizer=tokenizer,
        output_audio_token_spec=spec,
        input_audio_token_spec=spec,
        text_tokenizer=_TextTokenizer(),
        layout=Layout(
            text=(0, audio_start),
            audio=(audio_start, audio_start + tokenizer.vocab_size + 4),
        ),
        lexical_text_vocab_size=lexical_text_vocab_size,
        control_token_ids=control_token_ids,
        control_token_id=lambda token: control_token_ids[list(ControlToken).index(token)],
        boa_token_id=boa_token_id,
        eoa_token_id=boa_token_id + 1,
        mask_token_id=boa_token_id + 2,
        audio_schema_token_id=boa_token_id + 3,
        eos_token_id=7,
        pad_token_id=0,
        codec_audio_range=(audio_start, boa_token_id),
        input_codec_audio_range=(audio_start, boa_token_id),
        input_audio_decoupled=False,
        input_audio_block_name=Modality.AUDIO.value,
        input_boa_token_id=boa_token_id,
        input_eoa_token_id=boa_token_id + 1,
        input_audio_schema_token_id=boa_token_id + 3,
        structured_full_sequence=True,
        acoustic_side_channel=False,
        codec=None,
    )
    runtime.audio_generation_allowed_ids = (
        boa_token_id,
        boa_token_id + 3,
        *range(audio_start, boa_token_id),
        boa_token_id + 1,
    )
    runtime.generation_allowed_ids = lambda modality: (
        tuple(range(lexical_text_vocab_size))
        if modality is Modality.TEXT
        else runtime.audio_generation_allowed_ids
    )
    return cast(GenerationRuntime, runtime)


def _codes() -> AudioCodes:
    return AudioCodes(
        semantic_codes=torch.tensor([[1], [2], [3]], dtype=torch.int32),
        global_codes=torch.tensor([[0], [1]], dtype=torch.int64),
    )


def _tts_request() -> Request:
    return Request(
        prompt_ids=torch.tensor([2, 3]),
        task=Task.TTS,
        audio_input_positions=None,
    )


class BiCodecRequestInputTest(unittest.TestCase):
    def test_plain_tts_request_uses_text_only_prompt(self) -> None:
        runtime = _runtime()
        request = _tts_request()

        self.assertNotIn("audio_context", request)
        self.assertNotIn("prediction", request)
        self.assertIs(request["task"], Task.TTS)
        torch.testing.assert_close(request["prompt_ids"], torch.tensor([2, 3]))
        validate(request, _RouteModel(runtime, _codes()))

    def test_service_rejects_out_of_band_audio_context(self) -> None:
        runtime = _runtime()
        request = _tts_request()
        cast(dict[str, object], request)["audio_context"] = AudioCodes(
            semantic_codes=torch.tensor([[1], [2], [3]], dtype=torch.long),
            global_codes=torch.tensor([[2], [2]], dtype=torch.long),
        )

        with self.assertRaisesRegex(ValueError, "audio_context is not supported"):
            validate(request, _RouteModel(runtime, _codes()))

    def test_text_service_rejects_audio_context(self) -> None:
        runtime = _runtime()
        text_request = Request(
            prompt_ids=torch.tensor([1]),
            task=Task.T2TT,
            audio_input_positions=None,
        )
        cast(dict[str, object], text_request)["audio_context"] = _codes()
        with self.assertRaisesRegex(ValueError, "audio_context is not supported"):
            validate(text_request, _RouteModel(runtime, _codes()))

    def test_interleaved_generation_rejects_structured_bicodec(self) -> None:
        runtime = _runtime()
        model = _MixedRouteModel(runtime, _codes())

        with self.assertRaisesRegex(
            ValueError,
            "INTERLEAVED generation does not support structured BiCodec",
        ):
            generate_responses(
                [
                    Request(
                        prompt_ids=torch.tensor([1]),
                        task=Task.INTERLEAVED_AR,
                    )
                ],
                model,
                max_new_tokens=16,
                do_sample=False,
            )

    def test_parallel_generation_decodes_one_structured_bicodec_span(self) -> None:
        runtime = _runtime()
        output = AudioCodes(
            semantic_codes=torch.tensor([[4], [5]], dtype=torch.long),
            global_codes=torch.tensor([[2], [1]], dtype=torch.long),
        )
        codec = _GlobalCodec()
        cast(SimpleNamespace, cast(object, runtime)).codec = codec

        result = generate_responses(
            [Request(prompt_ids=torch.tensor([1]), task=Task.PARALLEL_AR)],
            _MixedRouteModel(runtime, output),
            max_new_tokens=16,
            do_sample=False,
        )[0]

        audio = result["audio"]
        if audio is None or audio["codes"] is None:
            self.fail("parallel BiCodec generation did not decode its audio span")
        torch.testing.assert_close(
            audio["codes"].semantic_codes,
            output.semantic_codes,
        )
        torch.testing.assert_close(audio["codes"].global_codes, output.global_codes)

    def test_parallel_codes_only_generation_returns_structured_codes(self) -> None:
        runtime = _runtime()
        output = AudioCodes(
            semantic_codes=torch.tensor([[4], [5]], dtype=torch.long),
            global_codes=torch.tensor([[2], [1]], dtype=torch.long),
        )
        cast(SimpleNamespace, cast(object, runtime)).output_audio_detokenizer = None

        result = generate_responses(
            [Request(prompt_ids=torch.tensor([1]), task=Task.PARALLEL_AR)],
            _MixedRouteModel(runtime, output),
            max_new_tokens=16,
            do_sample=False,
        )[0]

        audio = result["audio"]
        codes = None if audio is None else audio["codes"]
        if not isinstance(codes, AudioCodes):
            self.fail("codes-only parallel generation did not return AudioCodes")
        torch.testing.assert_close(codes.semantic_codes, output.semantic_codes)
        torch.testing.assert_close(codes.global_codes, output.global_codes)
        self.assertIsNone(audio["waveform"])
        self.assertIsNone(audio["sample_rate"])

    def test_generated_response_requires_global_and_semantic(self) -> None:
        runtime = _runtime()
        output = AudioCodes(
            semantic_codes=torch.tensor([[4], [5]], dtype=torch.long),
            global_codes=torch.tensor([[0], [0]], dtype=torch.long),
        )
        with self.assertRaises(ValueError):
            generate_responses(
                [_tts_request()],
                _RouteModel(runtime, output, streams=(AudioStream.SEMANTIC,)),
                max_new_tokens=8,
                do_sample=False,
            )

    def test_unconditioned_request_generates_global_and_semantic(self) -> None:
        runtime = _runtime()
        output = AudioCodes(
            semantic_codes=torch.tensor([[4], [5]], dtype=torch.long),
            global_codes=torch.tensor([[2], [1]], dtype=torch.long),
        )
        codec = _GlobalCodec()
        cast(SimpleNamespace, cast(object, runtime)).codec = codec

        result = generate_responses(
            [_tts_request()],
            _RouteModel(runtime, output),
            max_new_tokens=12,
            do_sample=False,
        )[0]

        audio = result["audio"]
        if audio is None or audio["codes"] is None:
            self.fail("unconditioned generation did not return structured audio codes")
        torch.testing.assert_close(
            audio["codes"].semantic_codes,
            output.semantic_codes,
        )
        torch.testing.assert_close(audio["codes"].global_codes, output.global_codes)

    def test_unconditioned_codes_only_generation_skips_detokenizer(self) -> None:
        runtime = _runtime()
        output = AudioCodes(
            semantic_codes=torch.tensor([[4], [5]], dtype=torch.long),
            global_codes=torch.tensor([[2], [1]], dtype=torch.long),
        )
        cast(SimpleNamespace, cast(object, runtime)).output_audio_detokenizer = None

        result = generate_responses(
            [_tts_request()],
            _RouteModel(runtime, output),
            max_new_tokens=12,
            do_sample=False,
        )[0]

        audio = result["audio"]
        codes = None if audio is None else audio["codes"]
        if not isinstance(codes, AudioCodes):
            self.fail("codes-only BiCodec generation did not return AudioCodes")
        torch.testing.assert_close(codes.semantic_codes, output.semantic_codes)
        torch.testing.assert_close(codes.global_codes, output.global_codes)
        self.assertIsNone(audio["waveform"])
        self.assertIsNone(audio["sample_rate"])

    def test_canonical_codes_only_decode_does_not_load_output_backend(self) -> None:
        runtime = Runtime(
            Config(
                audio_output=AudioOutputConfig(
                    tokenizer="bicodec",
                    detokenizer=None,
                )
            ),
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
        )
        runtime.__dict__["text_tokenizer"] = SimpleNamespace(vocab_size=8)
        codes = AudioCodes(
            semantic_codes=torch.tensor([[1], [2]], dtype=torch.long),
            global_codes=torch.zeros(32, 1, dtype=torch.long),
        )
        local_ids = runtime.audio_tokenizer.encode_full(codes)
        global_ids = runtime.layout.to_global(Modality.AUDIO.value, local_ids)
        response = torch.cat(
            (
                global_ids.new_tensor(
                    [runtime.boa_token_id, runtime.audio_schema_token_id]
                ),
                global_ids,
                global_ids.new_tensor([runtime.eoa_token_id]),
            )
        )

        result = decode_token_audio_results(
            [response],
            _RouteModel(runtime, codes),
        )[0]

        audio = result["audio"]
        self.assertIsNotNone(audio)
        if audio is None or not isinstance(audio["codes"], AudioCodes):
            self.fail("canonical codes-only decode did not return AudioCodes")
        self.assertNotIn("output_audio_tokenizer_backend", runtime.__dict__)


class _GlobalCodec:
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_codebook = torch.zeros(8, 4)
    semantic_codebook_sizes = (8,)
    global_codebook_sizes = (3,)
    global_unit_length = 2
    global_feature_dim = 4

    def __init__(self) -> None:
        self.codes: SemanticGlobalCodes | None = None

    def tokenize(self, audio: Tensor, sample_rate: int) -> object:
        del audio, sample_rate
        raise NotImplementedError

    def detokenize(self, codes: object) -> Tensor:
        if not isinstance(codes, SemanticGlobalCodes):
            raise TypeError("codes must be SemanticGlobalCodes")
        self.codes = codes
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


class _RouteModel:
    def __init__(
        self,
        runtime: GenerationRuntime,
        output: AudioCodes,
        *,
        streams: Sequence[AudioStream] = (
            AudioStream.GLOBAL,
            AudioStream.SEMANTIC,
        ),
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
        local_ids = runtime.audio_tokenizer.encode_streams(
            output,
            streams,
        )
        self.response = torch.cat(
            (
                torch.tensor([runtime.boa_token_id, runtime.audio_schema_token_id]),
                runtime.layout.to_global(Modality.AUDIO.value, local_ids),
                torch.tensor([runtime.eoa_token_id]),
            )
        )
        self.audio_token_frame_spans = torch.ones(
            runtime.audio_tokenizer.vocab_size,
            dtype=torch.long,
        )

        self._script = self.response.tolist()
        self._step = 0

    def generation_step(
        self,
        input_ids: Tensor,
        *,
        token_ids: Tensor | None,
        use_cache: bool,
        **kwargs,
    ) -> GenerationStepResult:
        del kwargs
        if self._step >= len(self._script):
            raise RuntimeError("structured generation script exhausted")
        next_id = self._script[self._step]
        self._step += 1
        if token_ids is None:
            raise AssertionError("structured generation must provide candidate ids")
        match = token_ids.eq(next_id).nonzero(as_tuple=False)
        if match.numel() == 0:
            raise AssertionError(f"scripted token {next_id} is outside candidates")
        logits = torch.full(
            (input_ids.size(0), input_ids.size(1), token_ids.numel()),
            float("-inf"),
        )
        logits[:, -1, int(match[0, 0])] = 0.0
        return GenerationStepResult(
            logits=logits,
            past_key_values=SimpleNamespace() if use_cache else None,
            audio_head_past=None,
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
        allowed_token_ids: Sequence[int] | Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> Tensor:
        del prompt_ids, max_new_tokens, temperature, top_p, prompt_attention_mask
        del audio_input_positions, stop_token_id, generation_modality, allowed_token_ids
        del do_sample, use_cache
        raise AssertionError("controlled audio response must use generation_step")


class _MixedRouteModel(_RouteModel):
    def __init__(
        self,
        runtime: GenerationRuntime,
        output: AudioCodes,
    ) -> None:
        super().__init__(runtime, output)
        self._script = [2, runtime.eos_token_id, *self.response.tolist()]
        self._step = 0

    def generation_step(
        self,
        input_ids: Tensor,
        *,
        token_ids: Tensor | None,
        use_cache: bool,
        **kwargs,
    ) -> GenerationStepResult:
        del kwargs
        next_id = self._script[self._step]
        self._step += 1
        if token_ids is None:
            raise AssertionError("mixed generation must provide its token union")
        match = token_ids.eq(next_id).nonzero()
        if match.numel() == 0:
            raise AssertionError("scripted mixed token is outside the token union")
        logits = torch.full(
            (input_ids.size(0), input_ids.size(1), token_ids.numel()),
            float("-inf"),
            device=input_ids.device,
        )
        logits[:, -1, int(match[0, 0])] = 0.0
        cache = SimpleNamespace(batch_select_indices=lambda indices: None) if use_cache else None
        return GenerationStepResult(
            logits=logits,
            past_key_values=cache,
            audio_head_past=None,
        )


if __name__ == "__main__":
    unittest.main()
