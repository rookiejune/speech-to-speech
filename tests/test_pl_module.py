from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

import torch
from anytrain.idspace import Modality
from anytrain.optim import CompositeOptimizer
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from speech_to_speech.config import ModelConfig, TrainConfig
from speech_to_speech.datamodule.batch_builder import CausalLMBatchBuilder
from speech_to_speech.model.orchestrator import Orchestrator
from speech_to_speech.pl_module import SpeechToSpeechModule
from speech_to_speech.types import AutoregressionExample, CausalLMBatch
from helpers import MockQwen, MockTokenizer


class SpeechToSpeechModuleTest(unittest.TestCase):
    def test_configure_optimizers_uses_muon_llm_optimizer(self) -> None:
        module, _ = _module_and_batch(train=TrainConfig(learning_rate=1e-4))

        configured = module.configure_optimizers()

        optimizer = configured["optimizer"]
        self.assertIsInstance(optimizer, CompositeOptimizer)
        self.assertEqual(set(optimizer.optimizers), {"muon", "adamw"})
        self.assertTrue(
            all(group["lr"] == module.train_config.learning_rate for group in optimizer.param_groups)
        )

    def test_configure_optimizers_keeps_audio_embedding_trainable(self) -> None:
        module, _ = _module_and_batch(train=TrainConfig(learning_rate=1e-4))

        configured = module.configure_optimizers()

        optimizer = configured["optimizer"]
        params = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        audio_weight = module.model.embed_tokens.modality_embeddings[Modality.AUDIO.value].weight
        self.assertTrue(audio_weight.requires_grad)
        self.assertIn(id(audio_weight), params)

    def test_trainer_runs_one_step(self) -> None:
        module, batch = _module_and_batch()

        with TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                default_root_dir=tmpdir,
                max_steps=1,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                enable_progress_bar=False,
                accelerator="cpu",
                devices=1,
            )
            trainer.fit(module, train_dataloaders=DataLoader([batch], batch_size=None))

        self.assertEqual(trainer.global_step, 1)


def _module_and_batch(
    *,
    train: TrainConfig | None = None,
) -> tuple[SpeechToSpeechModule, CausalLMBatch]:
    tokenizer = MockTokenizer()
    model = Orchestrator(
        qwen3=MockQwen(),
        tokenizer=tokenizer,
        bpe_vocab_size=5,
        model_config=ModelConfig(train_backbone=True),
        pretrained=False,
    )
    builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
    batch = builder.autoregression(
        AutoregressionExample(audio_ids=torch.tensor([0, 1, 2]))
    )
    return SpeechToSpeechModule(
        model,
        train
        or TrainConfig(
            max_steps=4,
            learning_rate=1e-4,
            schedule="warmup_cosine",
            warmup_steps=1,
        ),
    ), batch


if __name__ == "__main__":
    unittest.main()
