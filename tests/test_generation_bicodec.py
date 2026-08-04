from __future__ import annotations

import unittest
from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast

import torch
from anydataset.types import Modality
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from anytrain.module.idspace import Layout
from torch import Tensor, nn

from speech_to_speech.audio import AudioCodes, AudioStream
from speech_to_speech.generation import (
    Request,
    generate_responses,
)
from speech_to_speech.generation.bicodec import prepare_bicodec_tts_request
from speech_to_speech.generation._request import validate
from speech_to_speech.model.generation import GenerationStepResult
from speech_to_speech.runtime import AudioSequenceLayout
from speech_to_speech.runtime.audio_tokenizer import BiCodecAudioTokenizer
from speech_to_speech.runtime.protocol import GenerationRuntime
from speech_to_speech.runtime.types import Backbone
from speech_to_speech.task import Task


class _TextTokenizer:
    def __init__(self) -> None:
        self.encoded: list[str] = []
        self.conversations: list[list[dict[str, str]]] = []

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
        return [1]

def _runtime() -> GenerationRuntime:
    tokenizer = BiCodecAudioTokenizer(
        semantic_codebook_size=8,
        global_codebook_sizes=(3,),
        global_unit_length=2,
    )
    runtime = SimpleNamespace(
        audio_sequence_layout=AudioSequenceLayout.FLATTENED,
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
        codec=None,
    )
    return cast(GenerationRuntime, runtime)


def _codes() -> AudioCodes:
    return AudioCodes(
        semantic_codes=torch.tensor([[1], [2], [3]], dtype=torch.int32),
        global_codes=torch.tensor([[0], [1]], dtype=torch.int64),
    )


class BiCodecRequestInputTest(unittest.TestCase):
    def test_reference_request_serializes_text_and_global_stream(self) -> None:
        runtime = _runtime()
        codes = _codes()

        request = prepare_bicodec_tts_request(
            "hello world",
            runtime,
            reference_codes=codes,
            language="Chinese",
        )

        self.assertNotIn("audio_context", request)
        self.assertIs(request["task"], Task.TTS)
        self.assertEqual(len(runtime.text_tokenizer.encoded), 1)
        self.assertIn("hello world", runtime.text_tokenizer.encoded[0])
        local_audio = runtime.audio_tokenizer.encode_streams(
            codes,
            (AudioStream.GLOBAL,),
        )
        expected_suffix = torch.cat(
            (
                torch.tensor([runtime.boa_token_id]),
                runtime.layout.to_global(Modality.AUDIO.value, local_audio),
                torch.tensor([runtime.eoa_token_id, runtime.boa_token_id]),
            )
        )
        torch.testing.assert_close(
            request["prompt_ids"][-expected_suffix.numel() :],
            expected_suffix,
        )
        validate(request, _RouteModel(runtime, _codes()))

    def test_unconditioned_request_starts_full_output(self) -> None:
        runtime = _runtime()
        request = prepare_bicodec_tts_request("hello world", runtime)

        self.assertNotIn("audio_context", request)
        self.assertIs(request["task"], Task.TTS)
        torch.testing.assert_close(
            request["prompt_ids"],
            torch.tensor([2, 3, runtime.boa_token_id]),
        )
        self.assertEqual(len(runtime.text_tokenizer.encoded), 1)

    def test_reference_request_rejects_non_audio_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "text-to-audio"):
            prepare_bicodec_tts_request(
                "hello",
                _runtime(),
                reference_codes=_codes(),
                task=Task.T2TT,
            )

    def test_reference_request_validates_code_shapes(self) -> None:
        codes = _codes()
        malformed = AudioCodes(
            semantic_codes=codes.semantic_codes,
            global_codes=torch.tensor([[0, 1], [1, 2]]),
        )

        with self.assertRaisesRegex(ValueError, "global codes must have shape"):
            prepare_bicodec_tts_request(
                "hello",
                _runtime(),
                reference_codes=malformed,
            )

    def test_service_rejects_out_of_band_audio_context(self) -> None:
        runtime = _runtime()
        request = prepare_bicodec_tts_request(
            "hello",
            runtime,
            reference_codes=_codes(),
        )
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
        codec = _StructuredCodec()
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

    def test_reference_request_generates_semantic_and_reuses_prompt_global(self) -> None:
        runtime = _runtime()
        context = _codes()
        output = AudioCodes(
            semantic_codes=torch.tensor([[4], [5]], dtype=torch.long),
            global_codes=torch.tensor([[0], [0]], dtype=torch.long),
        )
        codec = _StructuredCodec()
        cast(SimpleNamespace, cast(object, runtime)).codec = codec

        result = generate_responses(
            [
                prepare_bicodec_tts_request(
                    "hello",
                    runtime,
                    reference_codes=context,
                )
            ],
            _RouteModel(runtime, output, streams=(AudioStream.SEMANTIC,)),
            max_new_tokens=4,
            do_sample=False,
        )[0]

        audio = result["audio"]
        if audio is None or audio["codes"] is None:
            self.fail("route generation did not return structured audio codes")
        torch.testing.assert_close(
            audio["codes"].semantic_codes,
            output.semantic_codes,
        )
        torch.testing.assert_close(audio["codes"].global_codes, context.global_codes)
        if codec.codes is None:
            self.fail("structured codec did not receive resolved route codes")
        torch.testing.assert_close(
            codec.codes.semantic,
            output.semantic_codes.unsqueeze(0),
        )
        torch.testing.assert_close(
            codec.codes.acoustic,
            cast(Tensor, context.global_codes).unsqueeze(0),
        )

    def test_unconditioned_request_generates_global_and_semantic(self) -> None:
        runtime = _runtime()
        output = AudioCodes(
            semantic_codes=torch.tensor([[4], [5]], dtype=torch.long),
            global_codes=torch.tensor([[2], [1]], dtype=torch.long),
        )
        codec = _StructuredCodec()
        cast(SimpleNamespace, cast(object, runtime)).codec = codec

        result = generate_responses(
            [prepare_bicodec_tts_request("hello", runtime)],
            _RouteModel(runtime, output),
            max_new_tokens=8,
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


class _StructuredCodec:
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_codebook = torch.zeros(8, 4)
    semantic_codebook_sizes = (8,)
    acoustic_codebook_sizes = (3,)
    acoustic_layout = AcousticLayout.FIXED_LENGTH
    acoustic_unit_length = 2
    acoustic_feature_dim = 4

    def __init__(self) -> None:
        self.codes: SemanticAcousticCodes | None = None

    def tokenize(self, audio: Tensor, sample_rate: int) -> object:
        del audio, sample_rate
        raise NotImplementedError

    def detokenize(self, codes: object) -> Tensor:
        if not isinstance(codes, SemanticAcousticCodes):
            raise TypeError("codes must be SemanticAcousticCodes")
        self.codes = codes
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
                runtime.layout.to_global(Modality.AUDIO.value, local_ids),
                torch.tensor([runtime.eoa_token_id]),
            )
        )
        self.audio_token_frame_spans = torch.ones(
            runtime.audio_tokenizer.vocab_size,
            dtype=torch.long,
        )

    def generation_step(self, *args, **kwargs) -> GenerationStepResult:
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
        allowed_token_ids: Sequence[int] | Tensor | None = None,
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


class _MixedRouteModel(_RouteModel):
    def __init__(
        self,
        runtime: GenerationRuntime,
        output: AudioCodes,
    ) -> None:
        super().__init__(runtime, output)
        runtime_object = cast(SimpleNamespace, cast(object, runtime))
        start, end = runtime.codec_audio_range
        runtime_object.audio_generation_allowed_ids = (
            *range(start, end),
            runtime.eoa_token_id,
        )
        runtime_object.generation_allowed_ids = lambda modality: (
            tuple(range(8))
            if modality is Modality.TEXT
            else runtime_object.audio_generation_allowed_ids
        )
        self._script = [2, runtime.eos_token_id, 2, *self.response.tolist()]
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
        cache = (
            SimpleNamespace(batch_select_indices=lambda indices: None)
            if use_cache
            else None
        )
        return GenerationStepResult(
            logits=logits,
            past_key_values=cache,
            audio_head_past=None,
        )


if __name__ == "__main__":
    unittest.main()
