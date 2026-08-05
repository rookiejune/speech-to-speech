from __future__ import annotations

import math
import unittest
from copy import deepcopy
from typing import Any, cast
from unittest.mock import patch

import torch
import torch.nn.functional as F
from anytrain.module.idspace import Layout

from speech_to_speech.model import AdapterType
from speech_to_speech.model.factory import (
    aligned_audio_adapter,
    aligned_audio_output_adapter,
)
from speech_to_speech.model.audio_output import (
    AudioOutputAdapterConfig,
    AudioOutputAdapterType,
)
from speech_to_speech.model.embedding.audio import create_semantic_audio_embedding
from speech_to_speech.model.embedding.fsq import (
    FsqEmbedding,
    FsqEmbeddingConfig,
    FsqFeature,
    _product_level_indices,
    reference_rms,
)
from speech_to_speech.runtime import AudioSequenceLayout
from speech_to_speech.runtime.audio_schema import AudioTokenSpec
from speech_to_speech.runtime.audio_tokenizer import FlattenedAudioTokenizer
from speech_to_speech.runtime.codec import StableCodec
from speech_to_speech.runtime.codec_contract import (
    fsq_level_values,
    fsq_levels,
    fsq_radix_order,
)


class FsqEmbeddingTest(unittest.TestCase):
    def test_product_level_indices_match_first_fastest_basis(self) -> None:
        levels = (2, 3, 4)
        indices = _product_level_indices(math.prod(levels), levels)
        basis = torch.tensor([1, 2, 6], dtype=torch.int64)
        expected = (
            torch.arange(math.prod(levels), dtype=torch.int64)[:, None] // basis
        ) % torch.tensor(levels, dtype=torch.int64)
        torch.testing.assert_close(indices, expected)

    def test_onehot_digit_levels_are_unrestricted(self) -> None:
        embed = _embedding(levels=((2, 2),), embedding_dim=4)
        with torch.no_grad():
            embed.offsets[0].zero_()
            embed._tables(0)[0].copy_(torch.tensor([[1.0, 0, 0, 0], [0, 2.0, 0, 0]]))
            embed._tables(0)[1].copy_(torch.tensor([[0.0, 0, 3.0, 0], [0, 0, 0, 4.0]]))

        rows = embed.code_embedding(torch.arange(4), stage=0)
        first_delta = rows[1] - rows[0]
        second_delta = rows[2] - rows[0]
        self.assertFalse(
            torch.linalg.vector_norm(first_delta).isclose(torch.linalg.vector_norm(second_delta))
        )
        self.assertEqual(int(torch.linalg.matrix_rank(rows - rows.mean(0))), 2)

    def test_digit_value_requires_codec_values_and_preserves_them(self) -> None:
        config = FsqEmbeddingConfig(feature=FsqFeature.DIGIT_VALUE)
        with self.assertRaisesRegex(ValueError, "canonical level values"):
            _embedding(levels=((3,),), config=config)

        embed = _embedding(
            levels=((3,),),
            config=config,
            level_values=(((-2.0, 0.5, 3.0),),),
            embedding_dim=2,
        )
        with torch.no_grad():
            embed.offsets[0].zero_()
            embed._slopes(0)[0].copy_(torch.tensor([1.0, -2.0]))
        expected = torch.tensor([[-2.0, 4.0], [0.5, -1.0], [3.0, -6.0]])
        torch.testing.assert_close(
            embed.code_embedding(torch.arange(3), stage=0),
            expected,
        )

    def test_markers_use_free_rows(self) -> None:
        embed = _embedding(levels=((2, 2),), free=3, embedding_dim=4)
        with torch.no_grad():
            embed.free.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))

        torch.testing.assert_close(embed(torch.tensor([4, 5, 6])), embed.free.weight)

    def test_rows_and_forward_match_weight_without_materializing(self) -> None:
        embed = _multi_stage_embedding()
        input_ids = torch.tensor([[0, 3, 4], [9, 10, 12]])
        expected = F.embedding(input_ids, embed.weight)

        with patch.object(
            FsqEmbedding,
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

    def test_full_and_selected_logits_match_weight_without_materializing(self) -> None:
        for feature in (FsqFeature.DIGIT_ONEHOT, FsqFeature.DIGIT_VALUE):
            with self.subTest(feature=feature):
                embed = _multi_stage_embedding(feature=feature)
                hidden = torch.randn(2, 3, embed.embedding_dim, dtype=torch.float64)
                local_ids = torch.tensor([0, 4, 9, 10, 12])
                weight = embed.weight
                expected = F.linear(hidden.to(dtype=weight.dtype), weight)
                selected = F.linear(
                    hidden.to(dtype=weight.dtype),
                    weight.index_select(0, local_ids),
                )

                with patch.object(
                    FsqEmbedding,
                    "_materialize",
                    side_effect=AssertionError("logits materialized the table"),
                ):
                    torch.testing.assert_close(embed.logits(hidden), expected)
                    torch.testing.assert_close(embed.logits(hidden, local_ids), selected)

    def test_factorized_paths_match_materialized_gradients(self) -> None:
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
        for name, parameter in embed.named_parameters():
            expected_parameter = dict(reference.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertIsNotNone(expected_parameter.grad, name)
            torch.testing.assert_close(
                parameter.grad,
                expected_parameter.grad,
                atol=1e-5,
                rtol=1e-5,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    def test_initialization_matches_reference_rms(self) -> None:
        torch.manual_seed(11)
        reference = torch.arange(1, 129, dtype=torch.float32).reshape(16, 8) / 100
        target = reference_rms(reference, chunk_rows=3)
        embed = FsqEmbedding(
            codebook_sizes=(6,),
            fsq_levels=((2, 3),),
            num_embeddings=9,
            embedding_dim=64,
            target_rms=target,
        )

        code_rms = float(
            embed.code_embedding(torch.arange(6), stage=0).detach().square().mean().sqrt()
        )
        free_rms = float(embed.free.weight.detach().square().mean().sqrt())
        self.assertAlmostEqual(code_rms, target, places=6)
        self.assertAlmostEqual(free_rms, target, places=6)

        value = _embedding(
            levels=((3,),),
            free=0,
            embedding_dim=64,
            config=FsqEmbeddingConfig(feature=FsqFeature.DIGIT_VALUE),
            level_values=(((-2.0, 0.5, 3.0),),),
        )
        value_rms = float(
            value.code_embedding(torch.arange(3), stage=0).detach().square().mean().sqrt()
        )
        self.assertAlmostEqual(value_rms, 0.1, places=6)
        self.assertEqual(value.weight.shape, (3, 64))

    def test_neighbors_are_stage_local_normalized_and_exclude_free_rows(self) -> None:
        embed = _embedding(levels=((2, 3), (2, 2)), free=2, embedding_dim=4)
        neighbors = embed.neighbors(torch.tensor([0, 1, 4, 6, 9, 10, 11]))

        self.assertEqual(neighbors.token_ids.shape, (7, 4))
        self.assertEqual(neighbors.valid[0].nonzero().flatten().tolist(), [1, 3])
        self.assertEqual(neighbors.token_ids[0, [1, 3]].tolist(), [1, 2])
        self.assertEqual(neighbors.token_ids[3, neighbors.valid[3]].tolist(), [7, 8])
        self.assertEqual(neighbors.token_ids[4, neighbors.valid[4]].tolist(), [8, 7])
        torch.testing.assert_close(
            neighbors.weights.sum(dim=-1),
            torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
        )
        self.assertFalse(bool(neighbors.valid[-2:].any()))

    def test_topology_and_value_facts_are_strictly_checked(self) -> None:
        source = _embedding(levels=((6, 6),), embedding_dim=4)
        target = _embedding(levels=((4, 9),), embedding_dim=4)
        with self.assertRaisesRegex(RuntimeError, "FSQ topology"):
            target.load_state_dict(source.state_dict(), strict=True)

        value_config = FsqEmbeddingConfig(feature=FsqFeature.DIGIT_VALUE)
        source_value = _embedding(
            levels=((3,),),
            config=value_config,
            level_values=(((-1.0, 0.0, 1.0),),),
        )
        target_value = _embedding(
            levels=((3,),),
            config=value_config,
            level_values=(((-2.0, 0.0, 2.0),),),
        )
        with self.assertRaisesRegex(RuntimeError, "FSQ level values"):
            target_value.load_state_dict(source_value.state_dict(), strict=True)

        state = source.state_dict()
        del state["topology"]
        with self.assertRaisesRegex(RuntimeError, 'Missing key.*"topology"'):
            source.load_state_dict(state, strict=True)

    def test_state_roundtrip_preserves_factorized_rows_and_logits(self) -> None:
        ids = torch.tensor([0, 3, 4, 9, 12])
        hidden = torch.randn(2, 7)
        for feature in (FsqFeature.DIGIT_ONEHOT, FsqFeature.DIGIT_VALUE):
            with self.subTest(feature=feature):
                source = _multi_stage_embedding(feature=feature)
                target = _multi_stage_embedding(feature=feature)

                incompatible = target.load_state_dict(
                    source.state_dict(),
                    strict=True,
                )

                self.assertEqual(incompatible.missing_keys, [])
                self.assertEqual(incompatible.unexpected_keys, [])
                torch.testing.assert_close(target.rows(ids), source.rows(ids))
                torch.testing.assert_close(target.logits(hidden), source.logits(hidden))

    def test_invalid_ids_preserve_embedding_contract(self) -> None:
        embed = _multi_stage_embedding()
        hidden = torch.randn(2, embed.embedding_dim)

        for input_ids in (torch.tensor([-1]), torch.tensor([embed.num_embeddings])):
            with self.subTest(input_ids=input_ids.tolist()):
                with self.assertRaisesRegex(IndexError, "index out of range"):
                    embed.rows(input_ids)
                with self.assertRaisesRegex(IndexError, "index out of range"):
                    embed.logits(hidden, input_ids)
        with self.assertRaisesRegex(RuntimeError, "Long, Int"):
            embed.rows(torch.tensor([0.0]))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            embed.logits(hidden, torch.tensor([[0]]))
        with self.assertRaisesRegex(ValueError, "stage 2 is out of range"):
            embed.code_embedding(torch.tensor([0]), stage=2)

    def test_stable_codec_exposes_optional_canonical_values(self) -> None:
        values = tuple(
            tuple(tuple(float(index) for index in range(3)) for _ in range(2)) for _ in range(1)
        )
        codec = StableCodec(
            _StableSource(
                codebook_sizes=(9,),
                fsq_levels=((3, 3),),
                fsq_level_values=values,
            )
        )
        self.assertEqual(fsq_levels(codec), ((3, 3),))
        self.assertEqual(fsq_level_values(codec), values)
        self.assertEqual(fsq_radix_order(codec), "first_fastest")

    def test_create_semantic_audio_embedding_uses_default_onehot(self) -> None:
        codec = StableCodec(_StableSource(codebook_sizes=(9,), fsq_levels=((3, 3),)))
        tokenizer = FlattenedAudioTokenizer(
            codebook_sizes=(9,),
            codec_name="stable_codec",
        )
        reference = torch.full((5, 16), 0.25)
        embed = create_semantic_audio_embedding(
            cast(Any, _Runtime(codec, tokenizer)),
            reference=reference,
            embedding_dim=16,
        )
        self.assertIsInstance(embed, FsqEmbedding)
        fsq = cast(FsqEmbedding, embed)
        self.assertIs(fsq.config.feature, FsqFeature.DIGIT_ONEHOT)
        self.assertEqual(fsq.num_embeddings, tokenizer.vocab_size + 4)
        self.assertAlmostEqual(
            float(fsq.weight.detach().square().mean().sqrt()),
            0.25,
            places=5,
        )

    def test_create_digit_value_requires_runtime_alignment_capability(self) -> None:
        codec = StableCodec(_StableSource(codebook_sizes=(9,), fsq_levels=((3, 3),)))
        tokenizer = FlattenedAudioTokenizer(
            codebook_sizes=(9,),
            codec_name="stable_codec",
        )
        with self.assertRaisesRegex(ValueError, "canonical level values"):
            create_semantic_audio_embedding(
                cast(Any, _Runtime(codec, tokenizer)),
                reference=torch.ones(2, 4),
                embedding_dim=4,
                fsq=FsqEmbeddingConfig(feature=FsqFeature.DIGIT_VALUE),
            )

    def test_matched_dims_collapse_default_linear_adapters(self) -> None:
        self.assertIsNone(aligned_audio_adapter(AdapterType.LINEAR, 64, 64))
        self.assertIs(aligned_audio_adapter(AdapterType.MLP, 64, 64), AdapterType.MLP)
        output = aligned_audio_output_adapter(
            AudioOutputAdapterConfig(type=AudioOutputAdapterType.LINEAR),
            64,
            64,
        )
        self.assertIs(output.type, AudioOutputAdapterType.NONE)

    def test_fsq_levels_requires_dim_one(self) -> None:
        self.assertIsNone(fsq_levels(_DimCodec(4, ((2, 2),))))

    def test_unknown_stable_sizes_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "known FSQ level layout"):
            StableCodec(_StableSource(codebook_sizes=(10,)))


class _StableSource:
    sample_rate = 16_000
    frame_rate = 25.0

    def __init__(
        self,
        codebook_sizes: tuple[int, ...] = (46_656,),
        fsq_levels: tuple[tuple[int, ...], ...] | None = None,
        fsq_level_values: tuple[tuple[tuple[float, ...], ...], ...] | None = None,
    ) -> None:
        self.codebook_sizes = codebook_sizes
        if fsq_levels is not None:
            self.fsq_levels = fsq_levels
        if fsq_level_values is not None:
            self.fsq_level_values = fsq_level_values
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
        self.audio_sequence_layout = AudioSequenceLayout.FLATTENED
        self.output_audio_token_spec = AudioTokenSpec.create(
            codec_name="stable_codec",
            sequence_layout=self.audio_sequence_layout.value,
            tokenizer=tokenizer,
        )
        audio_start = 8
        audio_end = audio_start + tokenizer.vocab_size + 4
        self.layout = Layout(text=(0, audio_start), audio=(audio_start, audio_end))
        self.boa_token_id = audio_end - 4
        self.eoa_token_id = audio_end - 3
        self.mask_token_id = audio_end - 2
        self.audio_schema_token_id = audio_end - 1
        self.output_audio_schema_id = self.output_audio_token_spec.schema_id


def _embedding(
    *,
    levels: tuple[tuple[int, ...], ...],
    free: int = 3,
    embedding_dim: int = 7,
    config: FsqEmbeddingConfig | None = None,
    level_values: tuple[tuple[tuple[float, ...], ...], ...] | None = None,
) -> FsqEmbedding:
    sizes = tuple(math.prod(stage) for stage in levels)
    return FsqEmbedding(
        codebook_sizes=sizes,
        fsq_levels=levels,
        num_embeddings=sum(sizes) + free,
        embedding_dim=embedding_dim,
        target_rms=0.1,
        config=config,
        level_values=level_values,
    )


def _multi_stage_embedding(
    *,
    feature: FsqFeature = FsqFeature.DIGIT_ONEHOT,
) -> FsqEmbedding:
    levels = ((2, 2), (3, 2))
    values = (
        ((-1.0, 1.0), (-2.0, 2.0)),
        ((-1.0, 0.0, 1.0), (-0.5, 0.5)),
    )
    return _embedding(
        levels=levels,
        config=FsqEmbeddingConfig(feature=feature),
        level_values=values if feature is FsqFeature.DIGIT_VALUE else None,
    )


if __name__ == "__main__":
    unittest.main()
