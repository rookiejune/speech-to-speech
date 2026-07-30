from __future__ import annotations

import unittest

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from types import SimpleNamespace

from speech_to_speech.audio_route import (
    AudioStream,
    BICODEC_GENERATE_GLOBAL,
    BICODEC_REUSE_PROMPT_GLOBAL,
)
from speech_to_speech.generation.decode import (
    decode_generated_bicodec_full,
    decode_generated_bicodec_route,
)
from speech_to_speech.generation._request import validate
from speech_to_speech.task import Task
from anytrain.module.idspace import Layout
from speech_to_speech.runtime import AudioRepresentation, Config
from speech_to_speech.runtime.audio_tokenizer import BiCodecAudioTokenizer
from speech_to_speech.model._generation import generate_bicodec_sequence


class BiCodecTokenizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = BiCodecAudioTokenizer(
            semantic_vocab_size=16,
            acoustic_codebook_sizes=(5, 7),
            acoustic_unit_length=3,
        )

    def test_semantic_only_roundtrip(self):
        semantic = torch.tensor([[2], [4], [6]])
        tokens = self.tokenizer.encode(semantic)

        torch.testing.assert_close(tokens, torch.tensor([2, 4, 6]))
        torch.testing.assert_close(self.tokenizer.decode(tokens), semantic)
        self.assertEqual(self.tokenizer.frame_spans(tokens).tolist(), [1, 1, 1])

    def test_full_sequence_roundtrip_is_slot_major(self):
        codes = SemanticAcousticCodes(
            semantic=torch.tensor([[2], [4], [6]]),
            acoustic=torch.tensor([[0, 1], [2, 3], [4, 5]]),
        )

        tokens = self.tokenizer.encode_full(codes)
        expected = torch.tensor(
            [
                self.tokenizer.acoustic_token_id,
                16,
                22,
                18,
                24,
                20,
                26,
                self.tokenizer.semantic_token_id,
                2,
                4,
                6,
                self.tokenizer.end_token_id,
            ]
        )
        torch.testing.assert_close(tokens, expected)

        decoded = self.tokenizer.decode_full(tokens)
        torch.testing.assert_close(decoded.semantic, codes.semantic)
        torch.testing.assert_close(decoded.acoustic, codes.acoustic)

    def test_full_sequence_preserves_independent_semantic_and_acoustic_lengths(self):
        codes = SemanticAcousticCodes(
            semantic=torch.tensor([[1], [2], [3], [4], [5]]),
            acoustic=torch.tensor([[0, 0], [1, 1], [2, 2]]),
        )

        decoded = self.tokenizer.decode_full(self.tokenizer.encode_full(codes))
        self.assertEqual(decoded.semantic.shape, (5, 1))
        self.assertEqual(decoded.acoustic.shape, (3, 2))

    def test_stream_routes_serialize_only_the_owned_streams(self):
        codes = SemanticAcousticCodes(
            semantic=torch.tensor([[2], [4]]),
            acoustic=torch.tensor([[0, 1], [2, 3], [4, 5]]),
        )

        semantic_tokens = self.tokenizer.encode_streams(
            codes,
            BICODEC_REUSE_PROMPT_GLOBAL.output.canonical_streams,
        )
        decoded_semantic = self.tokenizer.decode_streams(
            semantic_tokens,
            BICODEC_REUSE_PROMPT_GLOBAL.output.canonical_streams,
        )
        self.assertIsNone(decoded_semantic.acoustic)
        torch.testing.assert_close(decoded_semantic.semantic, codes.semantic)

        full_tokens = self.tokenizer.encode_streams(
            codes,
            BICODEC_GENERATE_GLOBAL.output.canonical_streams,
        )
        decoded_full = self.tokenizer.decode_streams(
            full_tokens,
            BICODEC_GENERATE_GLOBAL.output.canonical_streams,
        )
        torch.testing.assert_close(decoded_full.semantic, codes.semantic)
        torch.testing.assert_close(decoded_full.acoustic, codes.acoustic)

    def test_global_only_uses_structured_codes(self):
        global_codes = torch.tensor([[0, 1], [2, 3], [4, 5]])
        codes = SemanticAcousticCodes(
            semantic=torch.tensor([[2], [4]]),
            acoustic=global_codes,
        )
        global_tokens = self.tokenizer.encode_streams(
            codes,
            (AudioStream.GLOBAL,),
        )

        decoded = self.tokenizer.decode_streams(
            global_tokens,
            (AudioStream.GLOBAL,),
        )
        self.assertIsNone(decoded.semantic)
        torch.testing.assert_close(decoded.acoustic, global_codes)

    def test_global_only_does_not_require_semantic_codes(self):
        codes = SemanticAcousticCodes(
            semantic=torch.empty((0, 1), dtype=torch.long),
            acoustic=torch.tensor([[1, 2], [3, 4], [0, 5]]),
        )

        tokens = self.tokenizer.encode_global(codes)

        decoded = self.tokenizer.decode_streams(tokens, (AudioStream.GLOBAL,))
        self.assertIsNone(decoded.semantic)
        torch.testing.assert_close(decoded.acoustic, codes.acoustic)

    def test_bicodec_rejects_acoustic_stream_name(self):
        codes = SemanticAcousticCodes(
            semantic=torch.tensor([[2], [4]]),
            acoustic=torch.tensor([[1, 2], [3, 4], [0, 5]]),
        )
        with self.assertRaisesRegex(ValueError, "use global instead of acoustic"):
            self.tokenizer.encode_streams(
                codes,
                (AudioStream.ACOUSTIC,),
            )


class BiCodecDecodeTest(unittest.TestCase):
    def test_reference_route_requires_prompt_context(self):
        model = SimpleNamespace(
            runtime=SimpleNamespace(
                audio_route=BICODEC_REUSE_PROMPT_GLOBAL,
                layout=Layout(text=(0, 4), audio=(4, 30)),
            )
        )

        with self.assertRaisesRegex(ValueError, "structured prompt audio context"):
            validate(
                {"prompt_ids": torch.tensor([1]), "task": Task.TTS},
                model,
            )

    def test_full_decode_passes_structured_codes_to_backend(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_vocab_size=8,
            acoustic_codebook_sizes=(3,),
            acoustic_unit_length=2,
        )
        expected = SemanticAcousticCodes(
            semantic=torch.tensor([[1], [2]]),
            acoustic=torch.tensor([[0], [1]]),
        )
        tokens = tokenizer.encode_full(expected)

        codec = _StructuredCodec()
        waveform = decode_generated_bicodec_full(
            tokens[None] + 10,
            codec=codec,
            audio_tokenizer=tokenizer,
            audio_token_range=(10, 10 + tokenizer.vocab_size),
        )

        self.assertEqual(waveform.shape, (1, 1, 2))
        self.assertIsNotNone(codec.codes)
        torch.testing.assert_close(codec.codes.semantic, expected.semantic[None])
        torch.testing.assert_close(codec.codes.acoustic, expected.acoustic[None])

    def test_route_decode_uses_both_output_streams(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_vocab_size=8,
            acoustic_codebook_sizes=(3,),
            acoustic_unit_length=2,
        )
        output = SemanticAcousticCodes(
            semantic=torch.tensor([[1], [2]]),
            acoustic=torch.tensor([[0], [1]]),
        )
        tokens = tokenizer.encode_streams(
            output,
            BICODEC_GENERATE_GLOBAL.output.canonical_streams,
        )
        _, resolved = decode_generated_bicodec_route(
            tokens + 10,
            SemanticAcousticCodes(
                semantic=torch.tensor([[7]]),
                acoustic=torch.tensor([[2], [2]]),
            ),
            route=BICODEC_GENERATE_GLOBAL,
            codec=_StructuredCodec(),
            audio_tokenizer=tokenizer,
            audio_token_range=(10, 10 + tokenizer.vocab_size),
        )

        torch.testing.assert_close(resolved.semantic, output.semantic)
        torch.testing.assert_close(resolved.acoustic, output.acoustic)

    def test_global_reference_route_decodes_prompt_global_and_output_semantic(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_vocab_size=8,
            acoustic_codebook_sizes=(3,),
            acoustic_unit_length=2,
        )
        context = SemanticAcousticCodes(
            semantic=torch.tensor([[7]]),
            acoustic=torch.tensor([[2], [1]]),
        )
        output = SemanticAcousticCodes(
            semantic=torch.tensor([[1], [2]]),
            acoustic=torch.tensor([[0], [1]]),
        )
        tokens = tokenizer.encode_streams(
            output,
            BICODEC_REUSE_PROMPT_GLOBAL.output.canonical_streams,
        )

        _, resolved = decode_generated_bicodec_route(
            tokens + 10,
            context,
            route=BICODEC_REUSE_PROMPT_GLOBAL,
            codec=_StructuredCodec(),
            audio_tokenizer=tokenizer,
            audio_token_range=(10, 10 + tokenizer.vocab_size),
        )

        torch.testing.assert_close(resolved.semantic, output.semantic)
        torch.testing.assert_close(resolved.acoustic, context.acoustic)

    def test_state_machine_emits_a_complete_structured_sequence(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_vocab_size=8,
            acoustic_codebook_sizes=(3,),
            acoustic_unit_length=2,
        )
        model = _GenerationModel(30)
        output = generate_bicodec_sequence(
            model,
            torch.tensor([[1]]),
            tokenizer=tokenizer,
            streams=(AudioStream.GLOBAL, AudioStream.SEMANTIC),
            max_new_tokens=32,
            temperature=1.0,
            top_p=1.0,
            prompt_attention_mask=None,
            do_sample=False,
            use_cache=False,
        )

        decoded = tokenizer.decode_full(output[0, 1:-1] - 10)
        self.assertEqual(decoded.semantic.shape, (1, 1))
        self.assertEqual(decoded.acoustic.shape, (2, 1))

    def test_state_machine_can_generate_semantic_only_output(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_vocab_size=8,
            acoustic_codebook_sizes=(3,),
            acoustic_unit_length=2,
        )
        model = _GenerationModel(30)
        output = generate_bicodec_sequence(
            model,
            torch.tensor([[1]]),
            tokenizer=tokenizer,
            streams=(AudioStream.SEMANTIC,),
            max_new_tokens=10,
            temperature=1.0,
            top_p=1.0,
            prompt_attention_mask=None,
            do_sample=False,
            use_cache=False,
        )

        decoded = tokenizer.decode_streams(
            output[0, 1:-1] - 10,
            (AudioStream.SEMANTIC,),
        )
        self.assertIsNone(decoded.acoustic)
        self.assertEqual(decoded.semantic.shape, (1, 1))

    def test_state_machine_generates_global_and_semantic_output(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_vocab_size=8,
            acoustic_codebook_sizes=(3,),
            acoustic_unit_length=2,
        )
        model = _GenerationModel(30)
        output = generate_bicodec_sequence(
            model,
            torch.tensor([[1]]),
            tokenizer=tokenizer,
            streams=BICODEC_GENERATE_GLOBAL.output.canonical_streams,
            max_new_tokens=32,
            temperature=1.0,
            top_p=1.0,
            prompt_attention_mask=None,
            do_sample=False,
            use_cache=False,
        )

        decoded = tokenizer.decode_full(output[0, 1:-1] - 10)
        self.assertEqual(decoded.semantic.shape, (1, 1))
        self.assertEqual(decoded.acoustic.shape, (2, 1))


class BiCodecConfigTest(unittest.TestCase):
    def test_semantic_only_requires_artifact(self):
        config = Config(
            codec="bicodec",
            audio_representation=AudioRepresentation.DECOUPLED,
            semantic_codec_artifact="/tmp/bicodec-semantic",
        )
        self.assertEqual(config.codec, "bicodec")

    def test_full_sequence_does_not_require_artifact(self):
        config = Config(
            codec="bicodec",
            audio_representation=AudioRepresentation.FULL_CODEC_SEQUENCE,
        )
        self.assertIsNone(config.semantic_codec_artifact)


class _StructuredCodec:
    acoustic_layout = AcousticLayout.FIXED_LENGTH
    acoustic_unit_length = 2
    semantic_codebook_sizes = (8,)
    acoustic_codebook_sizes = (3,)
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_codebook = torch.zeros(8, 4)
    codes: SemanticAcousticCodes | None = None

    def detokenize(self, codes: SemanticAcousticCodes) -> torch.Tensor:
        self.codes = codes
        return torch.zeros(codes.semantic.size(0), 1, codes.acoustic.size(1))


class _GenerationModel:
    def __init__(self, eoa_token_id: int) -> None:
        self.runtime = SimpleNamespace(eoa_token_id=eoa_token_id, audio_head_range=(10, 100))
        self._semantic_steps = 0

    def generation_step(self, input_ids, *, token_ids, **kwargs):
        del input_ids, kwargs
        logits = torch.zeros(1, 1, token_ids.numel())
        if token_ids.numel() > 1 and self._semantic_steps > 0:
            logits[0, 0, -1] = 1.0
        self._semantic_steps += 1
        return SimpleNamespace(logits=logits, past_key_values=None)


if __name__ == "__main__":
    unittest.main()
