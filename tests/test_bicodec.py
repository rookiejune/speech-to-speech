from __future__ import annotations

import json
import hashlib
import unittest
from typing import cast

import torch
from anytrain.codec import SemanticGlobalCodes
from anytrain.tokenizer import CodecBPE
from torch import Tensor

from speech_to_speech.audio import AudioCodes, AudioStream
from speech_to_speech.generation.decode import (
    decode_generated_bicodec_full,
    decode_generated_bicodec_row,
)
from speech_to_speech.runtime import AudioSequenceLayout, Config, Runtime
from speech_to_speech.runtime.audio_schema import AudioTokenSpec
from speech_to_speech.runtime.audio_tokenizer import (
    BiCodecAudioTokenizer,
    TorchCodecBPE,
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
                "grammar": "bicodec-v3",
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
                "vocab_size": 30,
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
            torch.tensor([31, 18, 24, 20, 26, 22, 28, 30, 17]),
        )
        decoded = tokenizer.decode_full(tokens)
        torch.testing.assert_close(decoded.semantic_codes, codes.semantic_codes)
        torch.testing.assert_close(decoded.global_codes, codes.global_codes)
        torch.testing.assert_close(
            tokenizer.frame_spans(tokens),
            torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 3]),
        )
        state = tokenizer.contract_state()
        self.assertEqual(state["grammar"], "bicodec-v3")
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
            ]
        )
        torch.testing.assert_close(tokens, expected)

        decoded = self.tokenizer.decode_full(tokens)
        torch.testing.assert_close(decoded.semantic_codes, codes.semantic_codes)
        torch.testing.assert_close(decoded.global_codes, codes.global_codes)

    def test_schema_grammar_declares_stream_order_without_internal_end(self):
        codes = AudioCodes(
            semantic_codes=torch.tensor([[2], [4], [6]]),
            global_codes=torch.tensor([[0, 1], [2, 3], [4, 5]]),
        )
        spec = AudioTokenSpec.create(
            codec_name="bicodec",
            sequence_layout="flattened",
            tokenizer=self.tokenizer,
        )
        tokens = self.tokenizer.encode_full(codes)

        self.assertEqual(
            spec.grammar.private_marker_ids,
            (self.tokenizer.semantic_token_id, self.tokenizer.global_token_id),
        )
        self.assertFalse(hasattr(self.tokenizer, "end_token_id"))
        self.assertEqual(int(tokens[-1]), 6)
        self.assertTrue(spec.allows_eoa(tokens))

        global_length = 1 + (
            self.tokenizer.global_unit_length
            * len(self.tokenizer.global_codebook_sizes)
        )
        global_prefix = tokens[:global_length]
        self.assertFalse(spec.allows_eoa(global_prefix))
        with self.assertRaisesRegex(ValueError, "EOA.*before"):
            spec.validate_next(global_prefix, 100, eoa_token_id=100)
        self.assertTrue(spec.allows_eoa(global_prefix, variant="global"))

        semantic = torch.tensor([self.tokenizer.semantic_token_id, 1])
        self.assertTrue(spec.allows_eoa(semantic, variant="semantic"))
        with self.assertRaisesRegex(ValueError, "global.*marker"):
            spec.validate_prefix(
                torch.tensor(
                    [
                        self.tokenizer.semantic_token_id,
                        1,
                        self.tokenizer.global_token_id,
                    ]
                )
            )

        self.assertEqual(
            spec.grammar.generation_variants,
            ("global_semantic", "semantic"),
        )
        self.assertEqual(
            spec.next_candidates([]).marker_ids,
            (self.tokenizer.semantic_token_id, self.tokenizer.global_token_id),
        )
        self.assertEqual(
            spec.next_candidates(
                [],
                variants=("global_semantic",),
            ).marker_ids,
            (self.tokenizer.global_token_id,),
        )
        self.assertEqual(spec.generation_variant(()), "global_semantic")
        self.assertEqual(spec.generation_variant((global_prefix,)), "semantic")
        self.assertEqual(spec.generation_variant((tokens,)), "semantic")

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

        codec = _GlobalCodec()
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
            codec.codes.global_codes,
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
        schema = boa + 2
        _, resolved = decode_generated_bicodec_row(
            tokens + 10,
            None,
            codec=_GlobalCodec(),
            audio_tokenizer=tokenizer,
            audio_token_range=(10, 10 + tokenizer.vocab_size),
            boa_token_id=boa,
            eoa_token_id=boa + 1,
            audio_schema_token_id=schema,
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
        schema = boa + 2
        prompt_global = tokenizer.encode_global(context) + 10
        prompt = torch.cat(
            (
                torch.tensor([1, boa, schema]),
                prompt_global,
                torch.tensor([eoa, boa, schema]),
            )
        )
        _, resolved = decode_generated_bicodec_row(
            tokens + 10,
            prompt,
            codec=_GlobalCodec(),
            audio_tokenizer=tokenizer,
            audio_token_range=(10, 10 + tokenizer.vocab_size),
            boa_token_id=boa,
            eoa_token_id=eoa,
            audio_schema_token_id=schema,
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
        schema = boa + 2
        prompt = torch.cat(
            (
                torch.tensor([1, boa, schema]),
                tokenizer.encode_global(codes) + 10,
                torch.tensor([eoa, boa, schema]),
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
                    codec=_GlobalCodec(),
                    audio_tokenizer=tokenizer,
                    audio_token_range=(10, 10 + tokenizer.vocab_size),
                    boa_token_id=boa,
                    eoa_token_id=eoa,
                    audio_schema_token_id=schema,
                )

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
                    acoustic_generator_artifact="/tmp/bicodec-semantic",
                ),
                audio_sequence_layout=AudioSequenceLayout.FLATTENED,
            )

    def test_flattened_layout_does_not_require_artifact(self):
        runtime = Runtime(
            Config(codec="bicodec", audio_tokenizer="/tmp/bicodec-semantic-bpe"),
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
        )
        self.assertIsNone(runtime.acoustic_generator_artifact)

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


class _GlobalCodec:
    global_unit_length = 2
    semantic_codebook_sizes = (8,)
    global_codebook_sizes = (3,)
    global_feature_dim = 4
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_codebook = torch.zeros(8, 4)
    codes: SemanticGlobalCodes | None = None

    def tokenize(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> SemanticGlobalCodes:
        del audio, sample_rate
        raise NotImplementedError

    def detokenize(self, codes: SemanticGlobalCodes) -> torch.Tensor:
        self.codes = codes
        return torch.zeros(codes.semantic.size(0), 1, codes.global_codes.size(1))

    def global_codes_to_features(self, global_codes: torch.Tensor) -> torch.Tensor:
        return global_codes.to(dtype=torch.float32)

    def decode_features(
        self,
        semantic_codes: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        del semantic_codes
        return global_features.new_zeros((1, 1, 1))


if __name__ == "__main__":
    unittest.main()
