from __future__ import annotations

import json
import hashlib
import unittest
from types import SimpleNamespace
from typing import cast

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from anytrain.tokenizer import CodecBPE
from torch import Tensor

from speech_to_speech.audio_stream import AudioStream
from speech_to_speech.codes import AudioCodes
from speech_to_speech.generation.decode import (
    decode_generated_bicodec_full,
    decode_generated_bicodec_row,
)
from speech_to_speech.runtime import AudioSequenceLayout, Config, Runtime
from speech_to_speech.runtime.audio_tokenizer import (
    BiCodecAudioTokenizer,
    TorchCodecBPE,
)

from _constrained_codec_generation import (
    generate_marker_stream_bicodec_sequence_for_test,
)


class BiCodecTokenizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = BiCodecAudioTokenizer(
            semantic_codebook_size=16,
            global_codebook_sizes=(5, 7),
            global_unit_length=3,
        )

    def test_semantic_only_roundtrip(self):
        semantic = torch.tensor([[2], [4], [6]])
        tokens = self.tokenizer.encode(semantic)

        torch.testing.assert_close(tokens, torch.tensor([2, 4, 6]))
        torch.testing.assert_close(self.tokenizer.decode(tokens), semantic)
        self.assertEqual(self.tokenizer.frame_spans(tokens).tolist(), [1, 1, 1])

    def test_contract_state_is_json_safe_and_structural(self):
        state = self.tokenizer.contract_state()

        self.assertEqual(
            state,
            {
                "grammar": "bicodec-v2",
                "semantic_codebook_size": 16,
                "semantic_vocab_size": 16,
                "semantic_tokenizer": {
                    "grammar": "native-v1",
                    "vocab_size": 16,
                },
                "semantic_token_range": [0, 16],
                "global_codebook_sizes": [5, 7],
                "global_offsets": [16, 21],
                "global_token_ranges": [[16, 21], [21, 28]],
                "global_unit_length": 3,
                "semantic_token_id": 28,
                "global_token_id": 29,
                "end_token_id": 30,
                "vocab_size": 31,
            },
        )
        json.dumps(state)
        self.assertIsNot(
            state["global_token_ranges"],
            self.tokenizer.contract_state()["global_token_ranges"],
        )

    def test_semantic_bpe_roundtrip_moves_structured_offsets_after_bpe_vocab(self):
        semantic_tokenizer = semantic_bpe()
        tokenizer = BiCodecAudioTokenizer(
            semantic_codebook_size=16,
            global_codebook_sizes=(5, 7),
            global_unit_length=3,
            semantic_tokenizer=semantic_tokenizer,
        )
        codes = AudioCodes(
            semantic_codes=torch.tensor([[1], [2], [3]]),
            global_codes=torch.tensor([[0, 1], [2, 3], [4, 5]]),
        )

        tokens = tokenizer.encode_full(codes)

        self.assertEqual(tokenizer.semantic_codebook_size, 16)
        self.assertEqual(tokenizer.semantic_vocab_size, 18)
        self.assertIs(tokenizer.semantic_tokenizer, semantic_tokenizer)
        torch.testing.assert_close(
            tokens,
            torch.tensor([31, 18, 24, 20, 26, 22, 28, 30, 17, 32]),
        )
        decoded = tokenizer.decode_full(tokens)
        torch.testing.assert_close(decoded.semantic_codes, codes.semantic_codes)
        torch.testing.assert_close(decoded.global_codes, codes.global_codes)
        torch.testing.assert_close(
            tokenizer.frame_spans(tokens),
            torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 3, 0]),
        )
        state = tokenizer.contract_state()
        self.assertEqual(state["grammar"], "bicodec-v2")
        nested = state["semantic_tokenizer"]
        self.assertIsInstance(nested, dict)
        assert isinstance(nested, dict)
        self.assertEqual(nested["grammar"], "codec-bpe-v1")
        self.assertNotEqual(
            contract_hash(self.tokenizer.contract_state()),
            contract_hash(state),
        )

    def test_semantic_bpe_must_match_raw_codebook(self):
        base = CodecBPE.train(
            lambda: [[[1], [2], [1], [2]]],
            codebook_sizes=(8,),
            vocab_size=3,
            show_progress=False,
        )

        with self.assertRaisesRegex(ValueError, "codebook sizes"):
            BiCodecAudioTokenizer(
                semantic_codebook_size=16,
                global_codebook_sizes=(5,),
                global_unit_length=1,
                semantic_tokenizer=TorchCodecBPE.wrap(base),
            )

    def test_full_sequence_roundtrip_is_slot_major(self):
        codes = AudioCodes(
            semantic_codes=torch.tensor([[2], [4], [6]]),
            global_codes=torch.tensor([[0, 1], [2, 3], [4, 5]]),
        )

        tokens = self.tokenizer.encode_full(codes)
        expected = torch.tensor(
            [
                self.tokenizer.global_token_id,
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
        torch.testing.assert_close(decoded.semantic_codes, codes.semantic_codes)
        torch.testing.assert_close(decoded.global_codes, codes.global_codes)

    def test_full_sequence_preserves_independent_semantic_and_global_lengths(self):
        codes = AudioCodes(
            semantic_codes=torch.tensor([[1], [2], [3], [4], [5]]),
            global_codes=torch.tensor([[0, 0], [1, 1], [2, 2]]),
        )

        decoded = self.tokenizer.decode_full(self.tokenizer.encode_full(codes))
        self.assertEqual(decoded.semantic_codes.shape, (5, 1))
        self.assertEqual(cast(Tensor, decoded.global_codes).shape, (3, 2))

    def test_stream_routes_serialize_only_the_owned_streams(self):
        codes = AudioCodes(
            semantic_codes=torch.tensor([[2], [4]]),
            global_codes=torch.tensor([[0, 1], [2, 3], [4, 5]]),
        )

        semantic_tokens = self.tokenizer.encode_streams(
            codes,
            (AudioStream.SEMANTIC,),
        )
        decoded_semantic = self.tokenizer.decode_streams(
            semantic_tokens,
            (AudioStream.SEMANTIC,),
        )
        self.assertIsNone(decoded_semantic.global_codes)
        torch.testing.assert_close(
            decoded_semantic.semantic_codes,
            codes.semantic_codes,
        )

        full_tokens = self.tokenizer.encode_streams(
            codes,
            (AudioStream.GLOBAL, AudioStream.SEMANTIC),
        )
        decoded_full = self.tokenizer.decode_streams(
            full_tokens,
            (AudioStream.GLOBAL, AudioStream.SEMANTIC),
        )
        torch.testing.assert_close(decoded_full.semantic_codes, codes.semantic_codes)
        torch.testing.assert_close(decoded_full.global_codes, codes.global_codes)

    def test_global_only_uses_structured_codes(self):
        global_codes = torch.tensor([[0, 1], [2, 3], [4, 5]])
        codes = AudioCodes(
            semantic_codes=torch.tensor([[2], [4]]),
            global_codes=global_codes,
        )
        global_tokens = self.tokenizer.encode_streams(
            codes,
            (AudioStream.GLOBAL,),
        )

        decoded = self.tokenizer.decode_streams(
            global_tokens,
            (AudioStream.GLOBAL,),
        )
        self.assertIsNone(decoded.semantic_codes)
        torch.testing.assert_close(decoded.global_codes, global_codes)

    def test_global_only_does_not_require_semantic_codes(self):
        codes = AudioCodes(
            global_codes=torch.tensor([[1, 2], [3, 4], [0, 5]]),
        )

        tokens = self.tokenizer.encode_global(codes)

        decoded = self.tokenizer.decode_streams(tokens, (AudioStream.GLOBAL,))
        self.assertIsNone(decoded.semantic_codes)
        torch.testing.assert_close(decoded.global_codes, codes.global_codes)


class BiCodecDecodeTest(unittest.TestCase):
    def test_full_decode_passes_structured_codes_to_backend(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_codebook_size=8,
            global_codebook_sizes=(3,),
            global_unit_length=2,
        )
        expected = AudioCodes(
            semantic_codes=torch.tensor([[1], [2]]),
            global_codes=torch.tensor([[0], [1]]),
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
        torch.testing.assert_close(
            codec.codes.semantic,
            expected.semantic_codes[None],
        )
        torch.testing.assert_close(
            codec.codes.acoustic,
            cast(Tensor, expected.global_codes)[None],
        )

    def test_route_decode_uses_both_output_streams(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_codebook_size=8,
            global_codebook_sizes=(3,),
            global_unit_length=2,
        )
        output = AudioCodes(
            semantic_codes=torch.tensor([[1], [2]]),
            global_codes=torch.tensor([[0], [1]]),
        )
        tokens = tokenizer.encode_streams(
            output,
            (AudioStream.GLOBAL, AudioStream.SEMANTIC),
        )
        boa = 10 + tokenizer.vocab_size
        _, resolved = decode_generated_bicodec_row(
            tokens + 10,
            torch.tensor([1, boa]),
            codec=_StructuredCodec(),
            audio_tokenizer=tokenizer,
            audio_token_range=(10, 10 + tokenizer.vocab_size),
            boa_token_id=boa,
            eoa_token_id=boa + 1,
        )

        torch.testing.assert_close(resolved.semantic_codes, output.semantic_codes)
        torch.testing.assert_close(resolved.global_codes, output.global_codes)

    def test_reference_decode_reuses_prompt_global_and_output_semantic(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_codebook_size=8,
            global_codebook_sizes=(3,),
            global_unit_length=2,
        )
        context = AudioCodes(
            semantic_codes=torch.tensor([[7]]),
            global_codes=torch.tensor([[2], [1]]),
        )
        output = AudioCodes(
            semantic_codes=torch.tensor([[1], [2]]),
            global_codes=torch.tensor([[0], [1]]),
        )
        tokens = tokenizer.encode_streams(
            output,
            (AudioStream.SEMANTIC,),
        )

        boa = 10 + tokenizer.vocab_size
        eoa = boa + 1
        prompt_global = tokenizer.encode_global(context) + 10
        prompt = torch.cat(
            (
                torch.tensor([1, boa]),
                prompt_global,
                torch.tensor([eoa, boa]),
            )
        )
        _, resolved = decode_generated_bicodec_row(
            tokens + 10,
            prompt,
            codec=_StructuredCodec(),
            audio_tokenizer=tokenizer,
            audio_token_range=(10, 10 + tokenizer.vocab_size),
            boa_token_id=boa,
            eoa_token_id=eoa,
        )

        torch.testing.assert_close(resolved.semantic_codes, output.semantic_codes)
        torch.testing.assert_close(resolved.global_codes, context.global_codes)

    def test_route_decode_requires_exactly_one_global_owner(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_codebook_size=8,
            global_codebook_sizes=(3,),
            global_unit_length=2,
        )
        codes = AudioCodes(
            semantic_codes=torch.tensor([[1], [2]]),
            global_codes=torch.tensor([[0], [1]]),
        )
        boa = 10 + tokenizer.vocab_size
        eoa = boa + 1
        prompt = torch.cat(
            (
                torch.tensor([1, boa]),
                tokenizer.encode_global(codes) + 10,
                torch.tensor([eoa, boa]),
            )
        )
        cases = (
            (tokenizer.encode_full(codes) + 10, prompt),
            (
                tokenizer.encode_streams(codes, (AudioStream.SEMANTIC,)) + 10,
                None,
            ),
        )

        for response, prompt_ids in cases:
            with self.subTest(prompt=prompt_ids is not None), self.assertRaisesRegex(
                ValueError,
                "exactly one global stream owner",
            ):
                decode_generated_bicodec_row(
                    response,
                    prompt_ids,
                    codec=_StructuredCodec(),
                    audio_tokenizer=tokenizer,
                    audio_token_range=(10, 10 + tokenizer.vocab_size),
                    boa_token_id=boa,
                    eoa_token_id=eoa,
                )

    def test_state_machine_emits_a_complete_structured_sequence(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_codebook_size=8,
            global_codebook_sizes=(3,),
            global_unit_length=2,
        )
        model = _GenerationModel(30)
        output = generate_marker_stream_bicodec_sequence_for_test(
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
        self.assertEqual(decoded.semantic_codes.shape, (1, 1))
        self.assertEqual(cast(Tensor, decoded.global_codes).shape, (2, 1))

    def test_state_machine_can_generate_semantic_only_output(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_codebook_size=8,
            global_codebook_sizes=(3,),
            global_unit_length=2,
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
        self.assertIsNone(decoded.global_codes)
        self.assertEqual(cast(Tensor, decoded.semantic_codes).shape, (1, 1))

    def test_state_machine_generates_global_and_semantic_output(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_codebook_size=8,
            global_codebook_sizes=(3,),
            global_unit_length=2,
        )
        model = _GenerationModel(30)
        output = generate_marker_stream_bicodec_sequence_for_test(
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
        self.assertEqual(decoded.semantic_codes.shape, (1, 1))
        self.assertEqual(cast(Tensor, decoded.global_codes).shape, (2, 1))


class BiCodecConfigTest(unittest.TestCase):
    def test_bicodec_requires_one_structured_layout(self):
        with self.assertRaisesRegex(ValueError, "self-describing structured"):
            Runtime(
                Config(codec="bicodec"),
                audio_sequence_layout=AudioSequenceLayout.SEMANTIC,
            )

        with self.assertRaisesRegex(ValueError, "cannot use"):
            Runtime(
                Config(
                    codec="bicodec",
                    semantic_codec_artifact="/tmp/bicodec-semantic",
                ),
                audio_sequence_layout=AudioSequenceLayout.FLATTENED,
            )

    def test_flattened_layout_does_not_require_artifact(self):
        runtime = Runtime(
            Config(codec="bicodec", audio_tokenizer="/tmp/bicodec-semantic-bpe"),
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
        )
        self.assertIsNone(runtime.semantic_codec_artifact)

    def test_frame_codec_flattened_layout_still_rejects_bpe(self):
        with self.assertRaisesRegex(ValueError, "frame-code codecs"):
            Runtime(
                Config(codec="longcat", audio_tokenizer="/tmp/longcat-bpe"),
                audio_sequence_layout=AudioSequenceLayout.FLATTENED,
            )


def semantic_bpe() -> TorchCodecBPE:
    base = CodecBPE.train(
        lambda: [
            [[1], [2], [1], [2], [3]],
            [[1], [2], [3]],
        ],
        codebook_sizes=(16,),
        vocab_size=18,
        show_progress=False,
    )
    return TorchCodecBPE.wrap(base)


def contract_hash(state: dict[str, object]) -> str:
    payload = json.dumps(
        state,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
