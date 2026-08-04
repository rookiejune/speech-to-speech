from __future__ import annotations

import unittest

import torch
from lightning.pytorch import Trainer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from speech_to_speech.datamodule.mimo import collate_mimo
from speech_to_speech.mimo import MimoBatch, MimoSample
from speech_to_speech.pl_module.optim import Config as OptimConfig
from speech_to_speech.pl_module.mimo import MimoModule
from speech_to_speech.runtime.backbone.mimo import (
    DualStreamHiddenStates,
    DualStreamLogits,
)


class _TinyMimoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(8, 6)
        self.audio_embedding = nn.Embedding(8, 6)
        self.encoder = nn.Linear(6, 6)
        self.text_head = nn.Linear(6, 8)
        self.audio_head = nn.Linear(6, 8)

    def dual_hidden_states(self, batch: MimoBatch) -> DualStreamHiddenStates:
        shared = torch.tanh(
            self.encoder(
                self.text_embedding(batch.text_input_ids)
                + self.audio_embedding(batch.audio_input_ids)
            )
        )
        return DualStreamHiddenStates(text=shared, audio=shared, shared=shared)

    def dual_logits(self, hidden: DualStreamHiddenStates) -> DualStreamLogits:
        return DualStreamLogits(
            text=self.text_head(hidden.text),
            audio=self.audio_head(hidden.audio),
        )


def _sample() -> MimoSample:
    return MimoSample(
        text_input_ids=torch.tensor([0, 1, 2, 3]),
        audio_input_ids=torch.tensor([0, 3, 2, 1]),
        text_labels=torch.tensor([-100, 1, 2, 3]),
        audio_labels=torch.tensor([-100, 3, 2, 1]),
        text_loss_mask=torch.tensor([False, True, True, True]),
        audio_loss_mask=torch.tensor([False, True, True, True]),
        task_id="joint",
    )


def _collate(samples: list[MimoSample]) -> MimoBatch:
    return collate_mimo(
        samples,
        text_pad_token_id=7,
        audio_pad_token_id=7,
    )


class _Samples(Dataset[MimoSample]):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> MimoSample:
        if index < 0 or index >= self.size:
            raise IndexError(index)
        return _sample()


class MimoModuleSmokeTest(unittest.TestCase):
    def test_lightning_trainer_runs_one_optimizer_step(self) -> None:
        torch.manual_seed(5)
        model = _TinyMimoModel()
        module = MimoModule(
            model=model,
            optim=OptimConfig(
                name="adamw",
                learning_rate=1e-2,
                weight_decay=0.0,
            ),
        )
        loader = DataLoader(_Samples(2), batch_size=2, collate_fn=_collate)
        before = model.text_head.weight.detach().clone()
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )

        trainer.fit(module, train_dataloaders=loader)

        self.assertEqual(trainer.global_step, 1)
        self.assertFalse(torch.equal(before, model.text_head.weight.detach()))

    def test_optimizer_uses_pretraining_module_parameters(self) -> None:
        model = _TinyMimoModel()
        module = MimoModule(
            model=model,
            optim=OptimConfig(learning_rate=3e-4, weight_decay=0.02),
        )

        optimizer = module.configure_optimizers()

        self.assertIsInstance(optimizer, torch.optim.Optimizer)
        self.assertEqual(optimizer.param_groups[0]["lr"], 3e-4)
        optimized = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        self.assertTrue(all(id(parameter) in optimized for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
