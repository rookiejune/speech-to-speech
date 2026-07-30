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

from speech_to_speech.audio_route import (
    BICODEC_GENERATE_GLOBAL,
    BICODEC_REUSE_PROMPT_GLOBAL,
)
from speech_to_speech.generation import (
    Request,
    generate_responses,
    prepare_bicodec_global_tts_request,
    prepare_bicodec_tts_request,
)
from speech_to_speech.generation._request import validate
from speech_to_speech.runtime import AudioRepresentation
from speech_to_speech.runtime.audio_tokenizer import BiCodecAudioTokenizer
from speech_to_speech.runtime.protocol import GenerationRuntime
from speech_to_speech.runtime.types import Backbone
from speech_to_speech.task import Task


class _TextTokenizer:
    def __init__(self) -> None:
        self.encoded: list[str] = []

    def apply_chat_template(self, conversation, **kwargs) -> str:
        del kwargs
        return f"<user>{conversation[0]['content']}</user><assistant>"

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        self.encoded.append(text)
        if text == "hello world":
            return [2, 3]
        return [1]


def _runtime(*, route=BICODEC_REUSE_PROMPT_GLOBAL) -> GenerationRuntime:
    tokenizer = BiCodecAudioTokenizer(
        semantic_vocab_size=8,
        acoustic_codebook_sizes=(3,),
        acoustic_unit_length=2,
    )
    return cast(
        GenerationRuntime,
        SimpleNamespace(
            audio_route=route,
            audio_representation=AudioRepresentation.FULL_CODEC_SEQUENCE,
            audio_tokenizer=tokenizer,
            text_tokenizer=_TextTokenizer(),
            layout=Layout(
                text=(0, 8),
                audio=(8, 8 + tokenizer.vocab_size + 2),
            ),
            boa_token_id=8 + tokenizer.vocab_size,
            eoa_token_id=8 + tokenizer.vocab_size + 1,
            eos_token_id=7,
            pad_token_id=0,
            codec_audio_range=(8, 8 + tokenizer.vocab_size),
            structured_full_sequence=True,
            acoustic_side_channel=False,
            codec=None,
        ),
    )


def _codes() -> SemanticAcousticCodes:
    return SemanticAcousticCodes(
        semantic=torch.tensor([[1], [2], [3]], dtype=torch.int32),
        acoustic=torch.tensor([[0], [1]], dtype=torch.int64),
    )


class BiCodecRequestInputTest(unittest.TestCase):
    def test_reference_request_serializes_text_and_context(self) -> None:
        runtime = _runtime()
        codes = _codes()

        request = prepare_bicodec_tts_request(
            "hello world",
            codes,
            runtime,
            language="Chinese",
        )

        self.assertIs(request["audio_context"], codes)
        self.assertIs(request["task"], Task.TTS)
        self.assertEqual(runtime.text_tokenizer.encoded[1], "hello world")
        local_audio = runtime.audio_tokenizer.encode_streams(
            codes,
            BICODEC_REUSE_PROMPT_GLOBAL.prompt.canonical_streams,
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

    def test_global_request_starts_output_without_audio_context(self) -> None:
        runtime = _runtime(route=BICODEC_GENERATE_GLOBAL)
        request = prepare_bicodec_global_tts_request("hello world", runtime)

        self.assertIsNone(request["audio_context"])
        self.assertIs(request["task"], Task.TTS)
        torch.testing.assert_close(
            request["prompt_ids"],
            torch.tensor([1, 2, 3, 1, runtime.boa_token_id]),
        )

    def test_builders_require_their_exact_route(self) -> None:
        with self.assertRaisesRegex(ValueError, "no-reference global route"):
            prepare_bicodec_global_tts_request("hello", _runtime())
        with self.assertRaisesRegex(ValueError, "global-only reference route"):
            prepare_bicodec_tts_request(
                "hello",
                _codes(),
                _runtime(route=BICODEC_GENERATE_GLOBAL),
            )

    def test_reference_request_rejects_non_audio_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "text-to-audio"):
            prepare_bicodec_tts_request(
                "hello",
                _codes(),
                _runtime(),
                task=Task.T2TT,
            )

    def test_reference_request_validates_code_shapes(self) -> None:
        codes = _codes()
        malformed = SemanticAcousticCodes(
            semantic=codes.semantic,
            acoustic=codes.acoustic.unsqueeze(0),
        )

        with self.assertRaisesRegex(ValueError, "acoustic reference codes"):
            prepare_bicodec_tts_request("hello", malformed, _runtime())

    def test_service_rejects_context_that_does_not_match_prompt(self) -> None:
        runtime = _runtime()
        request = prepare_bicodec_tts_request("hello", _codes(), runtime)
        request["audio_context"] = SemanticAcousticCodes(
            semantic=torch.tensor([[1], [2], [3]], dtype=torch.long),
            acoustic=torch.tensor([[2], [2]], dtype=torch.long),
        )

        with self.assertRaisesRegex(ValueError, "does not serialize"):
            validate(request, _RouteModel(runtime, _codes()))

    def test_service_rejects_invalid_or_unexpected_context(self) -> None:
        runtime = _runtime()
        request = prepare_bicodec_tts_request("hello", _codes(), runtime)
        request["audio_context"] = cast(SemanticAcousticCodes, {"acoustic": []})
        with self.assertRaisesRegex(TypeError, "SemanticAcousticCodes"):
            validate(request, _RouteModel(runtime, _codes()))

        global_runtime = _runtime(route=BICODEC_GENERATE_GLOBAL)
        global_request = prepare_bicodec_global_tts_request("hello", global_runtime)
        global_request["audio_context"] = _codes()
        with self.assertRaisesRegex(ValueError, "without prompt streams"):
            validate(
                global_request,
                _RouteModel(global_runtime, _codes()),
            )

        text_request = Request(
            prompt_ids=torch.tensor([1]),
            task=Task.T2TT,
            audio_context=_codes(),
        )
        with self.assertRaisesRegex(ValueError, "text generation"):
            validate(text_request, _RouteModel(runtime, _codes()))

    def test_reference_request_generates_semantic_and_reuses_prompt_global(self) -> None:
        runtime = _runtime()
        context = _codes()
        output = SemanticAcousticCodes(
            semantic=torch.tensor([[4], [5]], dtype=torch.long),
            acoustic=torch.tensor([[0], [0]], dtype=torch.long),
        )
        codec = _StructuredCodec()
        cast(SimpleNamespace, cast(object, runtime)).codec = codec

        result = generate_responses(
            [prepare_bicodec_tts_request("hello", context, runtime)],
            _RouteModel(runtime, output),
            max_new_tokens=4,
            do_sample=False,
        )[0]

        audio = result["audio"]
        if audio is None or audio["codes"] is None:
            self.fail("route generation did not return structured audio codes")
        torch.testing.assert_close(audio["codes"].semantic, output.semantic)
        torch.testing.assert_close(audio["codes"].acoustic, context.acoustic)
        if codec.codes is None:
            self.fail("structured codec did not receive resolved route codes")
        torch.testing.assert_close(codec.codes.semantic, output.semantic.unsqueeze(0))
        torch.testing.assert_close(codec.codes.acoustic, context.acoustic.unsqueeze(0))

    def test_global_request_generates_global_and_semantic(self) -> None:
        runtime = _runtime(route=BICODEC_GENERATE_GLOBAL)
        output = SemanticAcousticCodes(
            semantic=torch.tensor([[4], [5]], dtype=torch.long),
            acoustic=torch.tensor([[2], [1]], dtype=torch.long),
        )
        codec = _StructuredCodec()
        cast(SimpleNamespace, cast(object, runtime)).codec = codec

        result = generate_responses(
            [prepare_bicodec_global_tts_request("hello", runtime)],
            _RouteModel(runtime, output),
            max_new_tokens=8,
            do_sample=False,
        )[0]

        audio = result["audio"]
        if audio is None or audio["codes"] is None:
            self.fail("global route generation did not return structured audio codes")
        torch.testing.assert_close(audio["codes"].semantic, output.semantic)
        torch.testing.assert_close(audio["codes"].acoustic, output.acoustic)


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
        route = runtime.audio_route
        if route is None:
            raise ValueError("route model requires an audio route")
        local_ids = runtime.audio_tokenizer.encode_streams(
            output,
            route.output.canonical_streams,
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

    def generate_full_codec_sequence(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        prompt_attention_mask: Tensor | None = None,
        audio_input_positions: Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> Tensor:
        del (
            max_new_tokens,
            temperature,
            top_p,
            prompt_attention_mask,
            audio_input_positions,
            do_sample,
            use_cache,
        )
        response = self.response.to(device=prompt_ids.device).expand(
            prompt_ids.size(0),
            -1,
        )
        return torch.cat((prompt_ids, response), dim=1)

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
            prompt_ids,
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
        raise AssertionError("structured route must use constrained generation")


if __name__ == "__main__":
    unittest.main()
