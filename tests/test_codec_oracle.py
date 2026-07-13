from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from anydataset.types import AudioItem, AudioView, Modality, Role
from anytrain.idspace import Layout
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf
from torch import nn

from speech_to_speech.codec_oracle import (
    AcousticFlowScreening,
    Initialization,
    collate,
    single_batch_loader,
)
from speech_to_speech.codec_oracle import SamplerEpochSetter
from speech_to_speech.model import AcousticFlow
from scripts.codec_oracle import training_callbacks


class CodecOracleTest(unittest.TestCase):
    def test_experiment_precision_matches_bfloat16_runtime(self):
        root = Path(__file__).parents[1]
        config = OmegaConf.load(root / "configs/experiment/acoustic_oracle.yaml")

        self.assertEqual(config.trainer.precision, "bf16-mixed")

    def test_sampler_epoch_callback_is_only_enabled_for_lba(self):
        config = OmegaConf.create(
            {
                "data": {"lba": {"enabled": False}},
                "trainer": {"expected_world_size": 1},
                "callbacks": {
                    "grad_norm": {"enabled": False},
                    "checkpoint": {"enabled": False},
                    "nonfinite": {"enabled": False},
                },
            }
        )

        callbacks = training_callbacks(config, Callback(), Path(self.id()))

        self.assertFalse(any(isinstance(x, SamplerEpochSetter) for x in callbacks))
        config.data.lba.enabled = True
        callbacks = training_callbacks(config, Callback(), Path(self.id()))
        self.assertTrue(any(isinstance(x, SamplerEpochSetter) for x in callbacks))

    def test_collate_pads_variable_length_codec_sequences(self):
        codec = OmegaConf.create({"view": "longcat", "objective": "flow"})
        data = OmegaConf.create({"max_seconds": 2.0})
        batch = collate(
            [_sample(3), _sample(1)],
            codec=codec,
            data=data,
            frame_rate=2.0,
        )

        self.assertEqual(tuple(batch["codes"].shape), (2, 3, 4))
        self.assertTrue(
            torch.equal(batch["mask"], torch.tensor([[1, 1, 1], [1, 0, 0]]).bool())
        )
        self.assertTrue((batch["codes"][1, 1:] == -1).all())

    def test_single_batch_loader_keeps_discrete_training_inputs(self):
        codes = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        flow = next(iter(single_batch_loader(codes)))
        self.assertEqual(tuple(flow["codes"].shape), (1, 2, 4))
        self.assertEqual(tuple(flow["mask"].shape), (1, 2))
        self.assertTrue(flow["mask"].all())
        self.assertFalse(flow["codes"].is_floating_point())

    def test_random_embedding_is_deterministic_without_changing_global_rng(self):
        codebook = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        torch.manual_seed(17)
        state = torch.random.get_rng_state()

        first = Initialization.RANDOM.weight(codebook, seed=3)
        after = torch.random.get_rng_state()
        second = Initialization.RANDOM.weight(codebook, seed=3)

        self.assertTrue(torch.equal(state, after))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, codebook))


    def test_acoustic_flow_screening_uses_formal_model_target_latent(self):
        calls: list[torch.Tensor] = []

        def dequantize(codes: torch.Tensor) -> torch.Tensor:
            calls.append(codes.clone())
            return codes.sum(dim=-1, keepdim=True).float()

        model = _OracleModel(dequantize)
        module = AcousticFlowScreening(
            model,
            initialization=Initialization.CODEC,
            seed=0,
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

        self.assertIs(module.model, model)
        parameter_names = dict(module.named_parameters())
        self.assertIn("model.acoustic_flow.decoder.input.weight", parameter_names)
        self.assertNotIn("decoder.input.weight", parameter_names)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(calls[0], batch["codes"][..., 1:]))
        self.assertIsNotNone(module.model.semantic_audio_embedding.weight.grad)

    def test_acoustic_flow_screening_replaces_padding_before_target_latent(self):
        calls: list[torch.Tensor] = []

        def dequantize(codes: torch.Tensor) -> torch.Tensor:
            calls.append(codes.clone())
            return codes.sum(dim=-1, keepdim=True).float()

        module = AcousticFlowScreening(
            _OracleModel(dequantize),
            initialization=Initialization.CODEC,
            seed=0,
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

    def sample(self, model, noise, **kwargs):
        del model, kwargs
        return SimpleNamespace(final=torch.zeros_like(noise))


class _OracleModel(nn.Module):
    def __init__(self, dequantize) -> None:
        super().__init__()
        self.layout = Layout(text=(0, 4), audio=(4, 14))
        self.semantic_audio_embedding = nn.Embedding.from_pretrained(
            torch.randn(8, 4),
            freeze=False,
        )
        self.semantic_audio_adapter = nn.Identity()
        self.acoustic_flow = AcousticFlow(4, 1, _Flow())
        self.dequantize = dequantize

    @property
    def acoustic_decoder(self):
        return self.acoustic_flow.decoder

    def target_frame_label_condition(self, labels, positions):
        return self.semantic_audio_embedding(labels - 4)

    def acoustic_target_latent(self, labels):
        return self.dequantize(labels)


def _sample(frames: int):
    return {
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={AudioView.LONGCAT: torch.arange(frames * 4).reshape(frames, 4)}
        )
    }


if __name__ == "__main__":
    unittest.main()
