from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from anydataset.types import AudioItem, AudioView, Modality, Role
from omegaconf import OmegaConf

from speech_to_speech.codec_oracle import (
    FlowOracle,
    Initialization,
    Objective,
    TokenOracle,
    collate,
    embedding_weight,
    single_batch_loader,
)


class CodecOracleTest(unittest.TestCase):
    def test_collate_pads_variable_length_codec_sequences(self):
        codec = OmegaConf.create(
            {"view": "longcat", "objective": "flow", "frame_rate": 2.0}
        )
        data = OmegaConf.create({"max_seconds": 2.0})
        batch = collate(
            [_sample(3), _sample(1)],
            codec=codec,
            data=data,
        )

        self.assertEqual(tuple(batch["codes"].shape), (2, 3, 4))
        self.assertTrue(
            torch.equal(batch["mask"], torch.tensor([[1, 1, 1], [1, 0, 0]]).bool())
        )
        self.assertTrue((batch["codes"][1, 1:] == -1).all())

    def test_single_batch_loader_keeps_discrete_training_inputs(self):
        codes = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        flow = next(iter(single_batch_loader(codes, objective=Objective.FLOW)))
        token = next(iter(single_batch_loader(codes[:, :1], objective=Objective.TOKEN)))

        self.assertEqual(tuple(flow["codes"].shape), (1, 2, 4))
        self.assertEqual(tuple(flow["mask"].shape), (1, 2))
        self.assertEqual(tuple(token["codes"].shape), (1, 2))
        self.assertTrue(flow["mask"].all())
        self.assertFalse(flow["codes"].is_floating_point())

    def test_token_objective_rejects_multiple_codebooks(self):
        with self.assertRaisesRegex(ValueError, "exactly one codebook"):
            Objective.TOKEN.select_codes(torch.ones(2, 3, dtype=torch.long))

    def test_random_embedding_is_deterministic_without_changing_global_rng(self):
        codebook = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        torch.manual_seed(17)
        state = torch.random.get_rng_state()

        first = embedding_weight(codebook, Initialization.RANDOM, seed=3)
        after = torch.random.get_rng_state()
        second = embedding_weight(codebook, Initialization.RANDOM, seed=3)

        self.assertTrue(torch.equal(state, after))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, codebook))

    def test_initialization_only_changes_unified_audio_embedding(self):
        codebook = torch.arange(32, dtype=torch.float32).reshape(8, 4)

        torch.manual_seed(11)
        codec = _token_oracle(codebook, initialization=Initialization.CODEC)
        torch.manual_seed(11)
        random = _token_oracle(codebook, initialization=Initialization.RANDOM)

        self.assertFalse(
            torch.equal(codec.embedding.weight[:8], random.embedding.weight[:8])
        )
        self.assertTrue(torch.equal(codec.position.weight, random.position.weight))
        self.assertTrue(torch.equal(codec.head.weight, random.head.weight))
        codec_backbone = dict(codec.backbone.named_parameters())
        random_backbone = dict(random.backbone.named_parameters())
        self.assertEqual(codec_backbone.keys(), random_backbone.keys())
        for name in codec_backbone:
            self.assertTrue(
                torch.equal(codec_backbone[name], random_backbone[name]),
                name,
            )

    def test_token_oracle_forward_backward(self):
        module = _token_oracle(torch.randn(8, 4), initialization=Initialization.CODEC)
        codes = torch.tensor([[1, 2, 3, 4]])

        logits = module(codes)
        loss = module.training_step(
            {"codes": codes, "mask": torch.ones_like(codes, dtype=torch.bool)},
            0,
        )
        loss.backward()

        self.assertEqual(tuple(logits.shape), (1, 4, 8))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(module.embedding.weight.grad)
        self.assertEqual(module.teacher_forced_ids(codes).shape, codes.shape)

    def test_flow_oracle_dequantizes_codes_in_training_module(self):
        calls: list[torch.Tensor] = []

        def dequantize(codes: torch.Tensor) -> torch.Tensor:
            calls.append(codes.clone())
            return codes.sum(dim=-1, keepdim=True).float()

        module = FlowOracle(
            torch.randn(8, 4),
            1,
            initialization=Initialization.CODEC,
            seed=0,
            dequantize=dequantize,
            flow_runtime=_Flow(),
            learning_rate=1e-3,
            weight_decay=0.0,
            target_mean=torch.zeros(1, 1, 1),
            target_std=torch.ones(1, 1, 1),
        )
        batch = {
            "codes": torch.tensor([[[1, 3, 4], [2, 5, 6]]]),
            "mask": torch.ones((1, 2), dtype=torch.bool),
        }

        output = module.training_step(batch, 0)
        output["loss"].backward()

        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(calls[0], batch["codes"][..., 1:]))
        self.assertIsNotNone(module.embedding.weight.grad)

    def test_flow_oracle_replaces_padding_before_dequantize(self):
        calls: list[torch.Tensor] = []

        def dequantize(codes: torch.Tensor) -> torch.Tensor:
            calls.append(codes.clone())
            return codes.sum(dim=-1, keepdim=True).float()

        module = FlowOracle(
            torch.randn(8, 4),
            1,
            initialization=Initialization.CODEC,
            seed=0,
            dequantize=dequantize,
            flow_runtime=_Flow(),
            learning_rate=1e-3,
            weight_decay=0.0,
            target_mean=torch.zeros(1, 1, 1),
            target_std=torch.ones(1, 1, 1),
        )
        batch = {
            "codes": torch.tensor([[[1, 3, 4], [-1, -1, -1]]]),
            "mask": torch.tensor([[True, False]]),
        }

        output = module.training_step(batch, 0)

        self.assertTrue(torch.equal(calls[0][0, 1], torch.tensor([0, 0])))
        self.assertTrue(torch.isfinite(output["loss"]))


class _Flow:
    def training_sample(self, target: torch.Tensor, *, x_0=None):
        del x_0
        return SimpleNamespace(
            x_t=torch.zeros_like(target),
            velocity=torch.ones_like(target),
            t=torch.zeros(target.size(0)),
        )


def _token_oracle(
    codebook: torch.Tensor,
    *,
    initialization: Initialization,
) -> TokenOracle:
    return TokenOracle(
        codebook,
        4,
        initialization=initialization,
        seed=5,
        layers=1,
        heads=2,
        feedforward_dim=8,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
    )


def _sample(frames: int):
    return {
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={AudioView.LONGCAT: torch.arange(frames * 4).reshape(frames, 4)}
        )
    }


if __name__ == "__main__":
    unittest.main()
