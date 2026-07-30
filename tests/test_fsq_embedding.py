from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from speech_to_speech.model.adapter import AdapterType
from speech_to_speech.model.base import (
    _aligned_audio_adapter,
    _aligned_audio_output_adapter,
)
from speech_to_speech.model.audio_output import (
    AudioOutputAdapterConfig,
    AudioOutputAdapterType,
)
from speech_to_speech.model.embedding.fsq import (
    FsqAffineEmbedding,
    _level_scalars,
    _product_level_indices,
)
from speech_to_speech.model.embedding.audio import create_semantic_audio_embedding
from speech_to_speech.runtime.audio_tokenizer import FlattenedAudioTokenizer
from speech_to_speech.runtime.codec import StableCodec
from speech_to_speech.runtime.types import fsq_levels


class FsqAffineEmbeddingTest(unittest.TestCase):
    def test_product_level_indices_match_fsq_basis(self):
        levels = (3, 3, 3)
        indices = _product_level_indices(27, levels)
        basis = torch.tensor([1, 3, 9], dtype=torch.int64)
        level_tensor = torch.tensor(levels, dtype=torch.int64)
        expected = (
            torch.arange(27, dtype=torch.int64)[:, None] // basis
        ) % level_tensor
        torch.testing.assert_close(indices, expected)

    def test_affine_preserves_scalar_distance_along_one_dim(self):
        embed = FsqAffineEmbedding(
            codebook_sizes=(4,),
            fsq_levels=((2, 2),),
            num_embeddings=4 + 2,
            embedding_dim=8,
        )
        with torch.no_grad():
            embed.biases[0].zero_()
            embed.slopes[0].zero_()
            embed.slopes[0][0] = torch.arange(8, dtype=torch.float32)
            embed.slopes[0][1] = torch.arange(8, 16, dtype=torch.float32)

        # product ids 0=(0,0) and 1=(1,0) differ only on dim 0
        left = embed.code_embedding(torch.tensor([0]), stage=0)
        right = embed.code_embedding(torch.tensor([1]), stage=0)
        scalars = _level_scalars(_product_level_indices(4, (2, 2)), (2, 2))
        delta = abs(float(scalars[1, 0] - scalars[0, 0]))
        expected = delta * float(embed.slopes[0][0].detach().norm())
        self.assertAlmostEqual(float((left - right).detach().norm()), expected, places=5)

    def test_markers_use_free_rows(self):
        embed = FsqAffineEmbedding(
            codebook_sizes=(4,),
            fsq_levels=((2, 2),),
            num_embeddings=4 + 3,
            embedding_dim=4,
        )
        with torch.no_grad():
            embed.free.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
            embed.biases[0].zero_()
            embed.slopes[0].zero_()

        markers = embed(torch.tensor([4, 5, 6]))
        torch.testing.assert_close(markers, embed.free.weight)
        codes = embed(torch.tensor([0, 1, 2, 3]))
        torch.testing.assert_close(codes, torch.zeros(4, 4))

    def test_tied_logits_match_materialized_weight(self):
        embed = FsqAffineEmbedding(
            codebook_sizes=(9,),
            fsq_levels=((3, 3),),
            num_embeddings=9 + 2,
            embedding_dim=5,
        )
        hidden = torch.randn(2, 5)
        weight = embed.weight
        logits = F.linear(hidden, weight)
        torch.testing.assert_close(logits, hidden @ weight.T)
        torch.testing.assert_close(embed(torch.arange(11)), weight)

    def test_stable_codec_exposes_fsq_levels(self):
        codec = StableCodec(_StableSource())
        self.assertEqual(codec.semantic_feature_dim, 1)
        self.assertEqual(codec.fsq_levels, ((6, 6, 6, 6, 6, 6),))
        self.assertEqual(fsq_levels(codec), ((6, 6, 6, 6, 6, 6),))

    def test_create_semantic_audio_embedding_uses_fsq_affine(self):
        codec = StableCodec(_StableSource(codebook_sizes=(9,), fsq_levels=((3, 3),)))
        tokenizer = FlattenedAudioTokenizer(
            codebook_sizes=(9,),
            codec_name="stable_codec",
        )
        runtime = _Runtime(codec, tokenizer)
        embed = create_semantic_audio_embedding(
            runtime,
            reference=torch.empty(1),
            embedding_dim=16,
        )
        self.assertIsInstance(embed, FsqAffineEmbedding)
        self.assertEqual(embed.embedding_dim, 16)
        self.assertEqual(embed.num_embeddings, tokenizer.vocab_size + 3)
        self.assertEqual(embed.weight.shape, (tokenizer.vocab_size + 3, 16))

    def test_fsq_embedding_requires_backbone_aligned_dim(self):
        codec = StableCodec(_StableSource(codebook_sizes=(9,), fsq_levels=((3, 3),)))
        tokenizer = FlattenedAudioTokenizer(
            codebook_sizes=(9,),
            codec_name="stable_codec",
        )
        with self.assertRaisesRegex(ValueError, "embedding_dim"):
            create_semantic_audio_embedding(
                _Runtime(codec, tokenizer),
                reference=torch.empty(1),
            )

    def test_matched_dims_collapse_default_linear_adapters(self):
        self.assertIsNone(
            _aligned_audio_adapter(AdapterType.LINEAR, 64, 64)
        )
        self.assertIs(
            _aligned_audio_adapter(AdapterType.MLP, 64, 64),
            AdapterType.MLP,
        )
        self.assertIs(
            _aligned_audio_adapter(AdapterType.LINEAR, 4, 64),
            AdapterType.LINEAR,
        )
        output = _aligned_audio_output_adapter(
            AudioOutputAdapterConfig(type=AudioOutputAdapterType.LINEAR),
            64,
            64,
        )
        self.assertIs(output.type, AudioOutputAdapterType.NONE)

    def test_fsq_levels_requires_dim_one(self):
        codec = _DimCodec(4, ((2, 2),))
        self.assertIsNone(fsq_levels(codec))

    def test_unknown_stable_sizes_fail(self):
        with self.assertRaisesRegex(ValueError, "known FSQ level layout"):
            StableCodec(_StableSource(codebook_sizes=(10,)))


class _StableSource:
    sample_rate = 16_000
    frame_rate = 25.0

    def __init__(
        self,
        codebook_sizes: tuple[int, ...] = (46_656,),
        fsq_levels: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        self.codebook_sizes = codebook_sizes
        if fsq_levels is not None:
            self.fsq_levels = fsq_levels
        self.codes = torch.tensor([[[0]]], dtype=torch.long)
        self.waveform = torch.zeros(1, 1, 8)

    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        del audio, sample_rate
        return self.codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        del codes
        return self.waveform


class _DimCodec:
    def __init__(
        self,
        semantic_feature_dim: int,
        fsq_levels: tuple[tuple[int, ...], ...],
    ) -> None:
        self.semantic_feature_dim = semantic_feature_dim
        self.fsq_levels = fsq_levels
        self.codebook_sizes = tuple(math.prod(stage) for stage in fsq_levels)


class _Runtime:
    def __init__(self, codec: object, tokenizer: FlattenedAudioTokenizer) -> None:
        self.codec = codec
        self.audio_tokenizer = tokenizer


if __name__ == "__main__":
    unittest.main()
