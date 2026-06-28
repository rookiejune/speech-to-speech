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
from speech_to_speech.types import AutoregressionExample, CausalLMBatch, LongCatBatchSide
from helpers import MockQwen, MockTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast


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

    def test_transfer_batch_to_device_preserves_longcat_sides(self) -> None:
        module, batch = _module_and_batch()
        batch.target_audio = LongCatBatchSide(
            semantic_ids=torch.tensor([[1, 2]]),
            semantic_mask=torch.tensor([[True, True]]),
            acoustic_ids=torch.zeros((1, 4, 2), dtype=torch.long),
            acoustic_mask=torch.tensor([[True, True]]),
        )

        moved = module.transfer_batch_to_device(batch, torch.device("cpu"), 0)

        self.assertIsNotNone(moved.target_audio)
        assert moved.target_audio is not None
        self.assertTrue(torch.equal(moved.target_audio.semantic_ids, torch.tensor([[1, 2]])))
        self.assertTrue(torch.equal(moved.target_audio.acoustic_mask, torch.tensor([[True, True]])))

    def test_loss_combines_semantic_and_acoustic_when_configured(self) -> None:
        model = JointLossModel()
        extractor = FakeFeatureExtractor(torch.full((1, 2, 4), 3.0))
        module = SpeechToSpeechModule(
            model,
            TrainConfig(acoustic_loss_weight=0.5),
            bpe=object(),
            acoustic_feature_extractor=extractor,
        )
        batch = CausalLMBatch(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.tensor([[1, 1]]),
            labels=torch.tensor([[1, 2]]),
            logits_to_keep=1,
            target_audio=LongCatBatchSide(
                semantic_ids=torch.tensor([[1, 2]]),
                semantic_mask=torch.tensor([[True, True]]),
                acoustic_ids=torch.zeros((1, 4, 2), dtype=torch.long),
                acoustic_mask=torch.tensor([[True, True]]),
            ),
        )

        loss = module._loss(batch)

        self.assertTrue(torch.equal(loss, torch.tensor(3.5)))
        self.assertTrue(torch.equal(model.target_features, torch.full((1, 2, 4), 3.0)))
        self.assertTrue(torch.equal(model.target_mask, torch.tensor([[True, True]])))


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


class JointLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.target_features: torch.Tensor | None = None
        self.target_mask: torch.Tensor | None = None

    def forward(self, batch: CausalLMBatch) -> CausalLMOutputWithPast:
        del batch
        return CausalLMOutputWithPast(loss=torch.tensor(2.0), logits=torch.empty(0))

    def acoustic_flow_loss(
        self,
        batch: CausalLMBatch,
        bpe: object,
        target_features: torch.Tensor,
        *,
        target_mask: torch.Tensor,
        source_feature_extractor: object,
    ) -> torch.Tensor:
        del batch, bpe, source_feature_extractor
        self.target_features = target_features
        self.target_mask = target_mask
        return torch.tensor(3.0)


class FakeFeatureExtractor:
    def __init__(self, features: torch.Tensor) -> None:
        self.features = features

    def acoustic_codes_to_features(self, acoustic_ids: torch.Tensor) -> torch.Tensor:
        del acoustic_ids
        return self.features


if __name__ == "__main__":
    unittest.main()
