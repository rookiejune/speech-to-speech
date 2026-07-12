from __future__ import annotations

import unittest

import torch

from anytrain.tokenizer import CodecBPE

from speech_to_speech.runtime.audio_tokenizer import TorchCodecBPE

try:
    import tokenizers
except ImportError:
    tokenizers = None


def replay(corpus):
    return lambda: corpus


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

        self.assertTrue(torch.equal(token_ids, torch.tensor([4])))
        self.assertTrue(torch.equal(frames, torch.tensor([[1, 4], [2, 7], [3, 8]])))

    def test_wrap_preserves_list_api(self):
        base = CodecBPE.train(
            replay([[[1], [2], [1], [2], [3]], [[1], [2], [3]]]),
            codebook_sizes=(16,),
            vocab_size=5,
        )
        tokenizer = TorchCodecBPE.wrap(base)

        self.assertEqual(tokenizer.encode([[1], [2], [3]]), [4])
        self.assertEqual(tokenizer.decode([4]), [(1,), (2,), (3,)])


if __name__ == "__main__":
    unittest.main()
