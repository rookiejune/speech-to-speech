from __future__ import annotations

import json
import unittest

import torch

from anytrain.tokenizer import CodecBPE

from speech_to_speech.runtime.audio_schema import AudioTokenSpec
from speech_to_speech.runtime.audio_tokenizer import (
    FlattenedAudioTokenizer,
    NativeAudioTokenizer,
    TorchCodecBPE,
)

try:
    import tokenizers
except ImportError:
    tokenizers = None


def replay(corpus):
    return lambda: corpus


class NativeAudioTokenizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = NativeAudioTokenizer(vocab_size=4)

    def test_tensor_api_preserves_device_and_uses_vector_shapes(self):
        frames = torch.tensor([[1], [2]], dtype=torch.int32)

        token_ids = self.tokenizer.encode(frames)
        decoded = self.tokenizer.decode(token_ids)
        spans = self.tokenizer.frame_spans(token_ids)
        assert isinstance(decoded, torch.Tensor)
        assert isinstance(spans, torch.Tensor)

        self.assertEqual(token_ids.device, frames.device)
        self.assertEqual(decoded.device, frames.device)
        self.assertEqual(spans.device, frames.device)
        self.assertEqual(token_ids.dtype, torch.long)
        self.assertEqual(decoded.dtype, torch.long)
        self.assertTrue(torch.equal(token_ids, torch.tensor([1, 2])))
        self.assertTrue(torch.equal(decoded, torch.tensor([[1], [2]])))
        self.assertTrue(torch.equal(spans, torch.ones(2, dtype=torch.long)))

    def test_list_api_preserves_identity_contract(self):
        self.assertTrue(
            torch.equal(
                self.tokenizer.encode([(1,), (2,)]),
                torch.tensor([1, 2]),
            )
        )
        self.assertEqual(self.tokenizer.decode([1, 2]), [(1,), (2,)])
        self.assertEqual(self.tokenizer.frame_spans([1, 2]), [1, 1])

    def test_contract_state_is_json_safe_and_structural(self):
        state = self.tokenizer.contract_state()

        self.assertEqual(
            state,
            {
                "grammar": "native-v1",
                "vocab_size": 4,
            },
        )
        json.dumps(state)

    def test_schema_grammar_has_no_private_marker_and_rejects_early_eoa(self):
        spec = AudioTokenSpec.create(
            codec_name="longcat",
            sequence_layout="semantic",
            tokenizer=self.tokenizer,
        )

        self.assertEqual(spec.grammar.private_marker_ids, ())
        self.assertFalse(spec.allows_eoa([]))
        with self.assertRaisesRegex(ValueError, "EOA.*before"):
            spec.validate_next([], 99, eoa_token_id=99)

        spec.validate_next([], 1, eoa_token_id=99)
        self.assertTrue(spec.allows_eoa([1]))
        spec.validate_next([1], 99, eoa_token_id=99)
        with self.assertRaisesRegex(ValueError, r"\[0, 4\)"):
            spec.validate_prefix([4])

        first = spec.next_candidates([])
        self.assertEqual(first.marker_ids, ())
        self.assertEqual(first.token_ranges, ((0, 4),))
        self.assertFalse(first.allows_eoa)
        continued = spec.next_candidates([1])
        self.assertEqual(continued.token_ranges, ((0, 4),))
        self.assertTrue(continued.allows_eoa)
        self.assertEqual(spec.generation_variant(()), "payload")

    def test_rejects_non_integer_ids(self):
        for value in (True, 1.5, 1 + 0j):
            with self.subTest(api="encode-list", value=value):
                with self.assertRaisesRegex(TypeError, "integer ids"):
                    self.tokenizer.encode([(value,)])
            with self.subTest(api="decode-list", value=value):
                with self.assertRaisesRegex(TypeError, "integer ids"):
                    self.tokenizer.decode([value])
            with self.subTest(api="spans-list", value=value):
                with self.assertRaisesRegex(TypeError, "integer ids"):
                    self.tokenizer.frame_spans([value])

        for dtype in (torch.bool, torch.float32, torch.complex64):
            with self.subTest(api="encode-tensor", dtype=dtype):
                with self.assertRaisesRegex(TypeError, "integer ids"):
                    self.tokenizer.encode(torch.ones((1, 1), dtype=dtype))
            with self.subTest(api="decode-tensor", dtype=dtype):
                with self.assertRaisesRegex(TypeError, "integer ids"):
                    self.tokenizer.decode(torch.ones(1, dtype=dtype))
            with self.subTest(api="spans-tensor", dtype=dtype):
                with self.assertRaisesRegex(TypeError, "integer ids"):
                    self.tokenizer.frame_spans(torch.ones(1, dtype=dtype))

        for dtype in (torch.uint16, torch.uint64):
            with self.subTest(api="encode-tensor", dtype=dtype):
                with self.assertRaisesRegex(TypeError, "signed dtype"):
                    self.tokenizer.encode(torch.ones((1, 1), dtype=dtype))
            with self.subTest(api="decode-tensor", dtype=dtype):
                with self.assertRaisesRegex(TypeError, "signed dtype"):
                    self.tokenizer.decode(torch.ones(1, dtype=dtype))
            with self.subTest(api="spans-tensor", dtype=dtype):
                with self.assertRaisesRegex(TypeError, "signed dtype"):
                    self.tokenizer.frame_spans(torch.ones(1, dtype=dtype))

    def test_rejects_invalid_shapes_and_ranges(self):
        invalid_shapes = (
            lambda: self.tokenizer.encode(torch.tensor([1])),
            lambda: self.tokenizer.encode(torch.tensor([[1, 2]])),
            lambda: self.tokenizer.decode(torch.tensor([[1]])),
            lambda: self.tokenizer.frame_spans(torch.tensor([[1]])),
        )
        for call in invalid_shapes:
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "shape|expects"):
                    call()

        invalid_ranges = (
            lambda: self.tokenizer.encode([(-1,)]),
            lambda: self.tokenizer.encode(torch.tensor([[4]])),
            lambda: self.tokenizer.decode([-1]),
            lambda: self.tokenizer.decode(torch.tensor([4])),
            lambda: self.tokenizer.frame_spans([-1]),
            lambda: self.tokenizer.frame_spans(torch.tensor([4])),
        )
        for call in invalid_ranges:
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, r"\[0, 4\)"):
                    call()


class FlattenedAudioTokenizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = FlattenedAudioTokenizer(
            codebook_sizes=(4, 10),
            codec_name="longcat",
        )

    def test_block_layout_uses_codebook_markers(self):
        token_ids = self.tokenizer.encode(torch.tensor([[1, 5], [2, 6]]))

        self.assertTrue(
            torch.equal(token_ids, torch.tensor([14, 1, 2, 15, 9, 10]))
        )
        self.assertEqual(
            self.tokenizer.special_tokens,
            {
                "codec:longcat:codebook:0": 14,
                "codec:longcat:codebook:1": 15,
            },
        )
        self.assertEqual(self.tokenizer.codebook_ranges, ((0, 4), (4, 14)))

    def test_contract_state_is_json_safe_and_structural(self):
        state = self.tokenizer.contract_state()

        self.assertEqual(
            state,
            {
                "grammar": "flattened-v1",
                "codec_name": "longcat",
                "codebook_sizes": [4, 10],
                "codebook_ranges": [[0, 4], [4, 14]],
                "codebook_token_ids": [14, 15],
                "vocab_size": 16,
            },
        )
        json.dumps(state)
        self.assertIsNot(
            state["codebook_sizes"],
            self.tokenizer.contract_state()["codebook_sizes"],
        )

    def test_schema_grammar_declares_marker_range_order_and_eoa_gate(self):
        spec = AudioTokenSpec.create(
            codec_name="longcat",
            sequence_layout="flattened",
            tokenizer=self.tokenizer,
        )

        self.assertEqual(spec.grammar.private_marker_ids, (14, 15))
        blocks = spec.grammar.variants[0].blocks
        self.assertEqual(
            [
                (block.marker_id, block.token_ranges)
                for block in blocks
            ],
            [(14, ((0, 4),)), (15, ((4, 14),))],
        )
        incomplete = [14, 1, 2, 15, 9]
        spec.validate_prefix(incomplete)
        self.assertFalse(spec.allows_eoa(incomplete))
        with self.assertRaisesRegex(ValueError, "EOA.*before"):
            spec.validate_next(incomplete, 99, eoa_token_id=99)

        complete = [*incomplete, 10]
        self.assertTrue(spec.allows_eoa(complete))
        spec.validate_next(complete, 99, eoa_token_id=99)
        for invalid in ([15], [14, 1, 15, 9, 10, 11]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                spec.validate_prefix(invalid)

        first = spec.next_candidates([])
        self.assertEqual(first.marker_ids, (14,))
        after_first_frame = spec.next_candidates([14, 1])
        self.assertEqual(after_first_frame.marker_ids, (15,))
        self.assertEqual(after_first_frame.token_ranges, ((0, 4),))
        complete_candidates = spec.next_candidates([14, 1, 15, 9])
        self.assertEqual(complete_candidates.token_ranges, ())
        self.assertTrue(complete_candidates.allows_eoa)

    def test_single_codebook_flattened_grammar_has_one_generation_variant(self):
        tokenizer = FlattenedAudioTokenizer(
            codebook_sizes=(4,),
            codec_name="single",
        )
        spec = AudioTokenSpec.create(
            codec_name="single",
            sequence_layout="flattened",
            tokenizer=tokenizer,
        )
        marker = tokenizer.codebook_token_ids[0]

        self.assertEqual(spec.grammar.generation_variants, ("full",))
        self.assertEqual(spec.next_candidates([]).marker_ids, (marker,))
        candidates = spec.next_candidates([marker, 1])
        self.assertEqual(candidates.token_ranges, ((0, 4),))
        self.assertTrue(candidates.allows_eoa)

    def test_round_trip_preserves_full_codec_frames(self):
        frames = torch.tensor([[1, 5], [2, 6]], dtype=torch.int32)

        token_ids = self.tokenizer.encode(frames)
        decoded = self.tokenizer.decode(token_ids)
        spans = self.tokenizer.frame_spans(token_ids)

        self.assertTrue(torch.equal(decoded, frames.to(dtype=torch.long)))
        self.assertTrue(torch.equal(spans, torch.tensor([0, 1, 1, 0, 0, 0])))
        self.assertEqual(
            self.tokenizer.decode(token_ids.tolist()),
            [(1, 5), (2, 6)],
        )

    def test_vocab_span_lookup_marks_only_first_codebook_as_frames(self):
        spans = self.tokenizer.frame_spans(range(self.tokenizer.vocab_size))

        self.assertEqual(spans, [1, 1, 1, 1, *([0] * 12)])

    def test_rejects_invalid_flattened_grammar(self):
        invalid = (
            [14, 1, 2, 15, 9],
            [1, 2, 15, 9, 10],
            [14, 1, 2, 15],
            [14, 1, 2, 15, 9, 40],
        )
        for token_ids in invalid:
            with self.subTest(token_ids=token_ids):
                with self.assertRaises(ValueError):
                    self.tokenizer.decode(token_ids)


@unittest.skipIf(tokenizers is None, "tokenizers is not installed")
class TorchCodecBPETest(unittest.TestCase):
    def test_wrap_adds_tensor_support_for_multi_codebook_frames(self):
        base = CodecBPE.train(
            replay(
                [
                    [[1, 4], [2, 7], [1, 4], [2, 7], [3, 8]],
                    [[1, 4], [2, 7], [3, 8]],
                ]
            ),
            codebook_sizes=(4, 16),
            vocab_size=5,
        )
        tokenizer = TorchCodecBPE.wrap(base)

        token_ids = tokenizer.encode(torch.tensor([[1, 4], [2, 7], [3, 8]]))
        frames = tokenizer.decode(torch.tensor([4]))
        spans = tokenizer.frame_spans(torch.tensor([4]))

        self.assertTrue(torch.equal(token_ids, torch.tensor([4])))
        self.assertTrue(torch.equal(frames, torch.tensor([[1, 4], [2, 7], [3, 8]])))
        self.assertTrue(torch.equal(spans, torch.tensor([3])))

    def test_wrap_preserves_list_api(self):
        base = CodecBPE.train(
            replay([[[1], [2], [1], [2], [3]], [[1], [2], [3]]]),
            codebook_sizes=(16,),
            vocab_size=5,
        )
        tokenizer = TorchCodecBPE.wrap(base)
        frames = [[1], [2], [3]]
        token_ids = base.encode(frames)

        self.assertEqual(tokenizer.encode(frames), token_ids)
        self.assertEqual(tokenizer.decode(token_ids), base.decode(token_ids))
        self.assertEqual(
            tokenizer.frame_spans(token_ids),
            [len(base.decode([token_id])) for token_id in token_ids],
        )

    def test_contract_state_captures_effective_token_and_merge_mapping(self):
        base = CodecBPE.train(
            replay(
                [
                    [[1, 4], [2, 7], [1, 4], [2, 7], [3, 8]],
                    [[1, 4], [2, 7], [3, 8]],
                ]
            ),
            codebook_sizes=(4, 16),
            vocab_size=5,
            show_progress=False,
        )
        tokenizer = TorchCodecBPE.wrap(base)

        state = tokenizer.contract_state()

        self.assertEqual(
            state,
            {
                "grammar": "codec-bpe-v1",
                "codebook_sizes": [4, 16],
                "vocab_size": 5,
                "tokens": [
                    {"token_id": 0, "frames": [[1, 4]]},
                    {"token_id": 1, "frames": [[2, 7]]},
                    {"token_id": 2, "frames": [[3, 8]]},
                    {"token_id": 3, "frames": [[1, 4], [2, 7]]},
                    {
                        "token_id": 4,
                        "frames": [[1, 4], [2, 7], [3, 8]],
                    },
                ],
                "merges": [
                    {"left": 0, "right": 1, "token_id": 3},
                    {"left": 3, "right": 2, "token_id": 4},
                ],
            },
        )
        json.dumps(state)
        next_state = tokenizer.contract_state()
        self.assertIsNot(state["tokens"], next_state["tokens"])
        self.assertIsNot(state["merges"], next_state["merges"])


if __name__ == "__main__":
    unittest.main()
