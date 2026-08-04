from __future__ import annotations

import math
import unittest
from copy import deepcopy
from typing import Any, cast
from unittest.mock import patch

import torch
import torch.nn.functional as F

from speech_to_speech.model import AdapterType
from speech_to_speech.model._assembly import (
    aligned_audio_adapter,
    aligned_audio_output_adapter,
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
            embed.offsets[0].zero_()
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
            embed.offsets[0].zero_()
            embed.slopes[0].zero_()

        markers = embed(torch.tensor([4, 5, 6]))
        torch.testing.assert_close(markers, embed.free.weight)
        codes = embed(torch.tensor([0, 1, 2, 3]))
        torch.testing.assert_close(codes, torch.zeros(4, 4))

    def test_rows_and_forward_match_weight_without_materializing(self):
        embed = _multi_stage_embedding()
        input_ids = torch.tensor([[0, 3, 4], [9, 10, 12]])
        weight = embed.weight
        expected = F.embedding(input_ids, weight)

        with patch.object(
            FsqAffineEmbedding,
            "_materialize",
            side_effect=AssertionError("lookup materialized the table"),
        ):
            torch.testing.assert_close(embed.rows(input_ids), expected)
            torch.testing.assert_close(embed(input_ids), expected)
            self.assertEqual(embed(torch.tensor(4)).shape, (embed.embedding_dim,))
            self.assertEqual(
                embed(torch.empty(2, 0, dtype=torch.long)).shape,
                (2, 0, embed.embedding_dim),
            )

    def test_full_and_selected_logits_match_weight_without_materializing(self):
        embed = _multi_stage_embedding()
        hidden = torch.randn(2, 3, embed.embedding_dim, dtype=torch.float64)
        local_ids = torch.tensor([0, 4, 9, 10, 12])
        weight = embed.weight
        expected = F.linear(hidden.to(dtype=weight.dtype), weight)
        selected = F.linear(
            hidden.to(dtype=weight.dtype),
            weight.index_select(0, local_ids),
        )

        with patch.object(
            FsqAffineEmbedding,
            "_materialize",
            side_effect=AssertionError("logits materialized the table"),
        ):
            torch.testing.assert_close(embed.logits(hidden), expected)
            torch.testing.assert_close(embed.logits(hidden, local_ids), selected)

    def test_factorized_paths_match_materialized_gradients(self):
        torch.manual_seed(10)
        embed = _multi_stage_embedding()
        reference = deepcopy(embed)
        input_ids = torch.tensor([[0, 4, 10], [3, 9, 12]])
        row_gradient = torch.randn(2, 3, embed.embedding_dim)
        hidden = torch.randn(2, embed.embedding_dim, requires_grad=True)
        reference_hidden = hidden.detach().clone().requires_grad_()
        logit_gradient = torch.randn(2, embed.num_embeddings)

        actual_loss = (embed.rows(input_ids) * row_gradient).sum()
        actual_loss = actual_loss + (embed.logits(hidden) * logit_gradient).sum()
        expected_rows = F.embedding(input_ids, reference.weight)
        expected_logits = F.linear(reference_hidden, reference.weight)
        expected_loss = (expected_rows * row_gradient).sum()
        expected_loss = expected_loss + (expected_logits * logit_gradient).sum()
        actual_loss.backward()
        expected_loss.backward()

        torch.testing.assert_close(hidden.grad, reference_hidden.grad)
        actual_parameters = dict(embed.named_parameters())
        expected_parameters = dict(reference.named_parameters())
        self.assertEqual(actual_parameters.keys(), expected_parameters.keys())
        for name, parameter in actual_parameters.items():
            self.assertIsNotNone(parameter.grad, name)
            self.assertIsNotNone(expected_parameters[name].grad, name)
            torch.testing.assert_close(
                parameter.grad,
                expected_parameters[name].grad,
                atol=1e-5,
                rtol=1e-5,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    def test_offsets_and_free_rows_use_embedding_scale(self):
        torch.manual_seed(11)
        embedding_dim = 256
        embed = FsqAffineEmbedding(
            codebook_sizes=(4,),
            fsq_levels=((2, 2),),
            num_embeddings=4 + 1024,
            embedding_dim=embedding_dim,
        )

        self.assertEqual(embed.offsets[0].shape, (embedding_dim,))
        expected = embedding_dim**-0.5
        actual = float(embed.free.weight.detach().std(unbiased=False))
        self.assertAlmostEqual(actual, expected, delta=expected * 0.03)

    def test_strict_load_rejects_legacy_bias_keys(self):
        embed = _multi_stage_embedding()
        state = embed.state_dict()
        del state["offsets.0"]
        state["biases.0"] = torch.randn(1, embed.embedding_dim)

        with self.assertRaises(RuntimeError) as error:
            embed.load_state_dict(state, strict=True)
        self.assertIn("offsets.0", str(error.exception))
        self.assertIn("biases.0", str(error.exception))

    def test_topology_is_persistent_and_strictly_checked(self):
        source = FsqAffineEmbedding(
            codebook_sizes=(36,),
            fsq_levels=((6, 6),),
            num_embeddings=39,
            embedding_dim=4,
        )
        target = FsqAffineEmbedding(
            codebook_sizes=(36,),
            fsq_levels=((4, 9),),
            num_embeddings=39,
            embedding_dim=4,
        )
        state = source.state_dict()

        self.assertIn("topology", state)
        self.assertEqual(state["topology"].dtype, torch.int64)
        with self.assertRaisesRegex(RuntimeError, "FSQ topology"):
            target.load_state_dict(state, strict=True)

    def test_strict_load_rejects_missing_topology(self):
        embed = _multi_stage_embedding()
        state = embed.state_dict()
        del state["topology"]

        with self.assertRaisesRegex(RuntimeError, 'Missing key.*"topology"'):
            embed.load_state_dict(state, strict=True)

    def test_invalid_ids_preserve_embedding_contract(self):
        embed = _multi_stage_embedding()
        hidden = torch.randn(2, embed.embedding_dim)

        for input_ids in (torch.tensor([-1]), torch.tensor([embed.num_embeddings])):
            with self.subTest(input_ids=input_ids.tolist()):
                with self.assertRaisesRegex(IndexError, "index out of range"):
                    embed.rows(input_ids)
                with self.assertRaisesRegex(IndexError, "index out of range"):
                    embed(input_ids)
                with self.assertRaisesRegex(IndexError, "index out of range"):
                    embed.logits(hidden, input_ids)
        with self.assertRaisesRegex(RuntimeError, "Long, Int"):
            embed.rows(torch.tensor([0.0]))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            embed.logits(hidden, torch.tensor([[0]]))
        with self.assertRaisesRegex(ValueError, "stage 2 is out of range"):
            embed.code_embedding(torch.tensor([0]), stage=2)
        with self.assertRaisesRegex(ValueError, "outside the FSQ codebook"):
            embed.code_embedding(torch.tensor([4]), stage=0)

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
            cast(Any, runtime),
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
                cast(Any, _Runtime(codec, tokenizer)),
                reference=torch.empty(1),
            )

    def test_matched_dims_collapse_default_linear_adapters(self):
        self.assertIsNone(
            aligned_audio_adapter(AdapterType.LINEAR, 64, 64)
        )
        self.assertIs(
            aligned_audio_adapter(AdapterType.MLP, 64, 64),
            AdapterType.MLP,
        )
        self.assertIs(
            aligned_audio_adapter(AdapterType.LINEAR, 4, 64),
            AdapterType.LINEAR,
        )
        output = aligned_audio_output_adapter(
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


def _multi_stage_levels() -> tuple[tuple[int, ...], ...]:
    return ((2, 2), (3, 2))


def _multi_stage_embedding() -> FsqAffineEmbedding:
    levels = _multi_stage_levels()
    codebook_sizes = tuple(math.prod(stage) for stage in levels)
    return FsqAffineEmbedding(
        codebook_sizes=codebook_sizes,
        fsq_levels=levels,
        num_embeddings=sum(codebook_sizes) + 3,
        embedding_dim=7,
    )


if __name__ == "__main__":
    unittest.main()
