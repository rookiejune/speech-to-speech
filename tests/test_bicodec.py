from __future__ import annotations

import unittest

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from types import SimpleNamespace

from speech_to_speech.generation.decode import decode_generated_bicodec_full
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
                self.tokenizer.codec_token_id,
                self.tokenizer.semantic_token_id,
                2,
                4,
                6,
                self.tokenizer.acoustic_token_id,
                16,
                22,
                18,
                24,
                20,
                26,
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


class BiCodecDecodeTest(unittest.TestCase):
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
            semantic_range=tokenizer.semantic_token_range,
            codec_token_id=tokenizer.codec_token_id,
            semantic_token_id=tokenizer.semantic_token_id,
            acoustic_token_id=tokenizer.acoustic_token_id,
            end_token_id=tokenizer.end_token_id,
            acoustic_offsets=tokenizer.acoustic_offsets,
            acoustic_sizes=tokenizer.acoustic_codebook_sizes,
            acoustic_unit_length=tokenizer.acoustic_unit_length,
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
