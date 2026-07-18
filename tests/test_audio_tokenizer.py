from __future__ import annotations

import unittest

import torch

from anytrain.tokenizer import CodecBPE

from speech_to_speech.runtime.audio_tokenizer import (
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


if __name__ == "__main__":
    unittest.main()
