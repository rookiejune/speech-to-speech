from __future__ import annotations

import unittest

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from types import SimpleNamespace

from speech_to_speech.audio_stream import AudioStream
from speech_to_speech.generation.decode import (
    decode_generated_bicodec_full,
    decode_generated_bicodec_full_row,
    decode_generated_bicodec_semantic_with_reference,
)
from speech_to_speech.generation._request import validate
from speech_to_speech.task import Task
from anytrain.module.idspace import Layout
from speech_to_speech.runtime import AudioSequenceLayout, Config, Runtime
from speech_to_speech.runtime.audio_tokenizer import BiCodecAudioTokenizer

from _constrained_codec_generation import (
    generate_marker_stream_bicodec_sequence_for_test,
)


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
            (AudioStream.SEMANTIC,),
        )
        decoded_semantic = self.tokenizer.decode_streams(
            semantic_tokens,
            (AudioStream.SEMANTIC,),
        )
        self.assertIsNone(decoded_semantic.acoustic)
        torch.testing.assert_close(decoded_semantic.semantic, codes.semantic)

        full_tokens = self.tokenizer.encode_streams(
            codes,
            (AudioStream.ACOUSTIC, AudioStream.SEMANTIC),
        )
        decoded_full = self.tokenizer.decode_streams(
            full_tokens,
            (AudioStream.ACOUSTIC, AudioStream.SEMANTIC),
        )
        torch.testing.assert_close(decoded_full.semantic, codes.semantic)
        torch.testing.assert_close(decoded_full.acoustic, codes.acoustic)

    def test_acoustic_only_uses_structured_codes(self):
        acoustic_codes = torch.tensor([[0, 1], [2, 3], [4, 5]])
        codes = SemanticAcousticCodes(
            semantic=torch.tensor([[2], [4]]),
            acoustic=acoustic_codes,
        )
        acoustic_tokens = self.tokenizer.encode_streams(
            codes,
            (AudioStream.ACOUSTIC,),
        )

        decoded = self.tokenizer.decode_streams(
            acoustic_tokens,
            (AudioStream.ACOUSTIC,),
        )
        self.assertIsNone(decoded.semantic)
        torch.testing.assert_close(decoded.acoustic, acoustic_codes)

    def test_acoustic_only_does_not_require_semantic_codes(self):
        codes = SemanticAcousticCodes(
            semantic=torch.empty((0, 1), dtype=torch.long),
            acoustic=torch.tensor([[1, 2], [3, 4], [0, 5]]),
        )

        tokens = self.tokenizer.encode_acoustic(codes)

        decoded = self.tokenizer.decode_streams(tokens, (AudioStream.ACOUSTIC,))
        self.assertIsNone(decoded.semantic)
        torch.testing.assert_close(decoded.acoustic, codes.acoustic)

class BiCodecDecodeTest(unittest.TestCase):
    def test_reference_route_requires_prompt_context(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_vocab_size=8,
            acoustic_codebook_sizes=(3,),
            acoustic_unit_length=2,
        )
        runtime = SimpleNamespace(
            audio_sequence_layout=AudioSequenceLayout.SEMANTIC,
            audio_tokenizer=tokenizer,
            layout=Layout(text=(0, 4), audio=(4, 30)),
            boa_token_id=4 + tokenizer.vocab_size,
            eoa_token_id=4 + tokenizer.vocab_size + 1,
        )
        model = SimpleNamespace(runtime=runtime)

        with self.assertRaisesRegex(ValueError, "requires audio context"):
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
            (AudioStream.ACOUSTIC, AudioStream.SEMANTIC),
        )
        _, resolved = decode_generated_bicodec_full_row(
            tokens + 10,
            codec=_StructuredCodec(),
            audio_tokenizer=tokenizer,
            audio_token_range=(10, 10 + tokenizer.vocab_size),
        )

        torch.testing.assert_close(resolved.semantic, output.semantic)
        torch.testing.assert_close(resolved.acoustic, output.acoustic)

    def test_reference_route_decodes_prompt_acoustic_and_output_semantic(self):
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
            (AudioStream.SEMANTIC,),
        )

        _, resolved = decode_generated_bicodec_semantic_with_reference(
            tokens + 10,
            context,
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
        output = generate_marker_stream_bicodec_sequence_for_test(
            model,
            torch.tensor([[1]]),
            tokenizer=tokenizer,
            streams=(AudioStream.ACOUSTIC, AudioStream.SEMANTIC),
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
        output = generate_marker_stream_bicodec_sequence_for_test(
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

    def test_state_machine_generates_acoustic_and_semantic_output(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_vocab_size=8,
            acoustic_codebook_sizes=(3,),
            acoustic_unit_length=2,
        )
        model = _GenerationModel(30)
        output = generate_marker_stream_bicodec_sequence_for_test(
            model,
            torch.tensor([[1]]),
            tokenizer=tokenizer,
            streams=(AudioStream.ACOUSTIC, AudioStream.SEMANTIC),
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
    def test_semantic_layout_requires_artifact(self):
        with self.assertRaisesRegex(ValueError, "semantic_codec_artifact"):
            Runtime(
                Config(codec="bicodec"),
                audio_sequence_layout=AudioSequenceLayout.SEMANTIC,
            )
        config = Config(
            codec="bicodec",
            semantic_codec_artifact="/tmp/bicodec-semantic",
        )
        runtime = Runtime(config, audio_sequence_layout=AudioSequenceLayout.SEMANTIC)
        self.assertEqual(runtime.codec_name, "bicodec")

    def test_flattened_layout_does_not_require_artifact(self):
        runtime = Runtime(
            Config(codec="bicodec"),
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
        )
        self.assertIsNone(runtime.semantic_codec_artifact)


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
