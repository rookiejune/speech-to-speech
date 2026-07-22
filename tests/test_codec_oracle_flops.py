from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from lightning import pytorch as pl
from torch import Tensor, nn

from speech_to_speech._flops import adapter, linear, qwen_backbone, require_linear
from speech_to_speech.codec_oracle import (
    AcousticFlowScreening,
    AcousticRVQScreening,
    Initialization,
    TrainingFlops,
)
from speech_to_speech.model import AcousticFlow, AcousticRVQDecoder
from speech_to_speech.model.adapter import MLPAdapter


class CodecOracleFlopsTest(unittest.TestCase):
    def test_flow_uses_padded_batch_shape(self):
        module = _flow_module()
        short = _batch(torch.tensor([[1, 1, 1], [1, 0, 0]]).bool(), codebooks=2)
        same_shape = _batch(torch.tensor([[1, 0, 0], [1, 0, 0]]).bool(), codebooks=2)
        padded = _batch(torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]]).bool(), codebooks=2)

        short_flops = _flops(module, short)

        self.assertEqual(short_flops, 11_616.0)
        self.assertEqual(_flops(module, same_shape), short_flops)
        self.assertGreater(_flops(module, padded), short_flops)

    def test_rvq_uses_valid_frame_count(self):
        module = _rvq_module()
        sparse = _batch(torch.tensor([[1, 1, 0], [1, 0, 0]]).bool(), codebooks=2)
        dense = _batch(torch.ones((2, 3), dtype=torch.bool), codebooks=2)

        sparse_flops = _flops(module, sparse)
        dense_flops = _flops(module, dense)

        self.assertEqual(sparse_flops, 7_344.0)
        self.assertEqual(dense_flops, 14_688.0)
        self.assertEqual(dense_flops, sparse_flops * 2)

    def test_counts_linear_and_mlp_semantic_adapters(self):
        batch = _batch(torch.ones((1, 2), dtype=torch.bool), codebooks=2)
        identity = _flops(_flow_module(), batch)
        linear = _flops(_flow_module(adapter=nn.Linear(4, 4)), batch)
        mlp = _flops(_flow_module(adapter=MLPAdapter(4, 4)), batch)

        self.assertEqual(linear - identity, 192.0)
        self.assertEqual(mlp - identity, 1_584.0)

    def test_rejects_batch_and_model_mismatches(self):
        flow = _flow_module()
        wrong_codebooks = _batch(torch.ones((1, 2), dtype=torch.bool), codebooks=1)

        with self.assertRaisesRegex(ValueError, "model requires 2"):
            _flops(flow, wrong_codebooks)

        rvq = _rvq_module()
        rvq.model.runtime.codec.acoustic_codebook_sizes = (7,)
        with self.assertRaisesRegex(ValueError, "runtime codec"):
            _flops(rvq, _batch(torch.ones((1, 2), dtype=torch.bool), codebooks=2))

        with self.assertRaisesRegex(TypeError, "AcousticFlowScreening"):
            _flops(
                pl.LightningModule(),
                _batch(torch.ones((1, 2), dtype=torch.bool), codebooks=2),
            )

    def test_rejects_invalid_batches_and_unsupported_models(self):
        flow = _flow_module()
        with self.assertRaisesRegex(ValueError, r"\[B, F, Q\]"):
            _flops(
                flow,
                {
                    "codes": torch.zeros((2, 3), dtype=torch.long),
                    "mask": torch.ones((2, 3), dtype=torch.bool),
                },
            )

        with self.assertRaisesRegex(TypeError, "mask must be boolean"):
            _flops(
                flow,
                {
                    "codes": torch.zeros((1, 2, 3), dtype=torch.long),
                    "mask": torch.ones((1, 2)),
                },
            )

        with self.assertRaisesRegex(TypeError, "unsupported module"):
            _flops(
                _flow_module(adapter=nn.Sequential(nn.Linear(4, 4))),
                _batch(torch.ones((1, 2), dtype=torch.bool), codebooks=2),
            )

        with self.assertRaisesRegex(ValueError, "REPA"):
            _flops(
                _flow_module(repa_feature_dim=3),
                _batch(torch.ones((1, 2), dtype=torch.bool), codebooks=2),
            )

        rvq = _rvq_module()
        rvq.model.acoustic_decoder.decoder.config.layer_types = ["sliding_attention"]
        with self.assertRaisesRegex(ValueError, "full causal attention"):
            _flops(
                rvq,
                _batch(torch.ones((1, 2), dtype=torch.bool), codebooks=2),
            )

    def test_shared_qwen_helper_uses_padded_rows_and_valid_attention_lengths(self):
        core = _rvq_module().model.acoustic_decoder.decoder

        sparse = qwen_backbone(
            core,
            batch=2,
            sequence=3,
            lengths=(3, 1),
        )
        dense = qwen_backbone(
            core,
            batch=2,
            sequence=3,
            lengths=torch.tensor([3, 3]),
        )

        self.assertEqual(sparse, 2_032)
        self.assertEqual(dense, 2_112)
        self.assertEqual(dense - sparse, 80)

    def test_shared_linear_helpers_validate_shape_and_count(self):
        projection = nn.Linear(3, 5)

        require_linear(projection, 3, 5, "test")
        self.assertEqual(linear(projection, 7), 210)
        with self.assertRaisesRegex(ValueError, r"Linear\(3, 4\)"):
            require_linear(projection, 3, 4, "test")

    def test_shared_linear_helpers_reject_subclasses(self):
        projection = _LinearSubclass(3, 5)

        with self.assertRaisesRegex(ValueError, r"Linear\(3, 5\)"):
            require_linear(projection, 3, 5, "test")
        with self.assertRaisesRegex(TypeError, "exact nn.Linear"):
            linear(projection, 7)
        with self.assertRaisesRegex(TypeError, "unsupported module"):
            adapter(
                projection,
                rows=7,
                in_features=3,
                out_features=5,
                name="test adapter",
            )


class _FlowRuntime:
    def training_sample(self, target: Tensor, *, x_0=None):
        del x_0
        return SimpleNamespace(
            x_t=torch.zeros_like(target),
            velocity=torch.ones_like(target),
            t=torch.zeros(target.size(0)),
        )

    def sample(self, model, noise, **kwargs):
        del model, kwargs
        return SimpleNamespace(final=torch.zeros_like(noise))


class _LinearSubclass(nn.Linear):
    pass


class _FlowModel(nn.Module):
    def __init__(
        self,
        *,
        adapter: nn.Module | None = None,
        repa_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.runtime = SimpleNamespace(
            codec=SimpleNamespace(
                acoustic_codebook_sizes=(7, 9),
                acoustic_feature_dim=2,
            )
        )
        self.semantic_audio_embedding = nn.Embedding(16, 4)
        self.semantic_audio_adapter = nn.Identity() if adapter is None else adapter
        self.acoustic_flow = AcousticFlow(
            4,
            2,
            _FlowRuntime(),
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
            repa_feature_dim=repa_feature_dim,
        )

    @property
    def acoustic_decoder(self):
        return self.acoustic_flow.decoder


class _RVQModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.runtime = SimpleNamespace(
            codec=SimpleNamespace(acoustic_codebook_sizes=(7, 9))
        )
        self.semantic_audio_embedding = nn.Embedding(16, 4)
        self.semantic_audio_adapter = nn.Identity()
        self.acoustic_decoder = AcousticRVQDecoder(
            4,
            2,
            (7, 9),
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
        )


def _flow_module(
    *,
    adapter: nn.Module | None = None,
    repa_feature_dim: int | None = None,
) -> AcousticFlowScreening:
    return AcousticFlowScreening(
        _FlowModel(adapter=adapter, repa_feature_dim=repa_feature_dim),
        initialization=Initialization.CODEC,
        seed=0,
        flow_runtime=_FlowRuntime(),
        learning_rate=1e-3,
        weight_decay=0.0,
        target_mean=torch.zeros(1, 1, 2),
        target_std=torch.ones(1, 1, 2),
    )


def _rvq_module() -> AcousticRVQScreening:
    return AcousticRVQScreening(
        _RVQModel(),
        initialization=Initialization.CODEC,
        seed=0,
        learning_rate=1e-3,
        weight_decay=0.0,
    )


def _batch(mask: Tensor, *, codebooks: int) -> dict[str, Tensor]:
    codes = torch.zeros((*mask.shape, codebooks + 1), dtype=torch.long)
    codes = codes.masked_fill(~mask[..., None], -1)
    return {"codes": codes, "mask": mask}


def _flops(module: pl.LightningModule, batch: object) -> float:
    return TrainingFlops()(
        trainer=SimpleNamespace(),
        pl_module=module,
        outputs=None,
        batch=batch,
        batch_idx=0,
    )


if __name__ == "__main__":
    unittest.main()
