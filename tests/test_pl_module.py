from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

import torch
from anytrain.idspace import Modality
from anytrain.optim import CompositeOptimizer
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from speech_to_speech.config import ModelConfig, QwenBackboneConfig, TrainConfig
from speech_to_speech.datamodule.batch_builder import CausalLMBatchBuilder
from speech_to_speech.model.acoustic import AcousticFlowLossStats
from speech_to_speech.model.orchestrator import (
    AcousticFlowInputs,
    DiTConditionTensors,
    Orchestrator,
)
from speech_to_speech.pl_module import SpeechToSpeechModule
from speech_to_speech.types import (
    AutoregressionExample,
    CausalLMBatch,
    IGNORE_INDEX,
    LongCatBatchSide,
    TaskFamily,
)
from helpers import MockQwen, MockTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast


class SpeechToSpeechModuleTest(unittest.TestCase):
    def test_configure_optimizers_uses_muon_llm_optimizer(self) -> None:
        module, _ = _module_and_batch(
            train=TrainConfig(max_steps=100, learning_rate=1e-4, warmup_ratio=0.01)
        )

        configured = module.configure_optimizers()

        optimizer = configured["optimizer"]
        self.assertIsInstance(optimizer, CompositeOptimizer)
        self.assertEqual(set(optimizer.optimizers), {"muon", "adamw"})
        self.assertTrue(
            all(lr == module.train_config.learning_rate for lr in configured["lr_scheduler"]["scheduler"].base_lrs)
        )

    def test_configure_optimizers_keeps_audio_embedding_trainable(self) -> None:
        module, _ = _module_and_batch(
            train=TrainConfig(max_steps=100, learning_rate=1e-4, warmup_ratio=0.01)
        )

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

    def test_configure_optimizers_supports_muon_and_adamw_lrs(self) -> None:
        module, _ = _module_and_batch(
            train=TrainConfig(
                learning_rate=1e-4,
                adamw_learning_rate=2e-4,
                muon_learning_rate=3e-4,
                max_steps=100,
                warmup_ratio=0.01,
            )
        )

        configured = module.configure_optimizers()

        optimizer = configured["optimizer"]
        scheduler = configured["lr_scheduler"]["scheduler"]
        self.assertIsInstance(optimizer, CompositeOptimizer)
        self.assertEqual(scheduler.base_lrs[0], 3e-4)
        self.assertTrue(
            all(lr == 2e-4 for lr in scheduler.base_lrs[1:])
        )

    def test_configure_optimizers_rejects_non_warmup_cosine_schedule(self) -> None:
        module, _ = _module_and_batch(
            train=TrainConfig(max_steps=100, learning_rate=1e-4, schedule="constant")
        )

        with self.assertRaisesRegex(ValueError, "warmup_cosine"):
            module.configure_optimizers()

    def test_configure_optimizers_derives_warmup_steps_from_ratio(self) -> None:
        module, _ = _module_and_batch(
            train=TrainConfig(
                max_steps=100,
                learning_rate=1e-4,
                warmup_ratio=0.1,
            )
        )

        configured = module.configure_optimizers()

        scheduler = configured["lr_scheduler"]["scheduler"]
        self.assertEqual(scheduler.lr_lambdas[0](0), 0.0)
        self.assertEqual(scheduler.lr_lambdas[0](10), 1.0)
        self.assertLess(scheduler.lr_lambdas[0](20), 1.0)

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

    def test_training_step_uses_last_forward_hidden_state_for_acoustic_loss(self) -> None:
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

        loss = module.training_step(batch, 0)

        self.assertTrue(torch.equal(loss, torch.tensor(3.5)))
        self.assertTrue(model.forward_requested_hidden_states)
        self.assertIs(model.hidden_states, model.forward_hidden_states[-1])

    def test_loss_logs_acoustic_t_bin_losses(self) -> None:
        model = JointLossStatsModel()
        module = SpeechToSpeechModule(
            model,
            TrainConfig(acoustic_loss_weight=0.5),
            bpe=object(),
            acoustic_feature_extractor=FakeFeatureExtractor(torch.ones((4, 1, 1))),
        )
        batch = CausalLMBatch(
            input_ids=torch.ones((4, 2), dtype=torch.long),
            attention_mask=torch.ones((4, 2), dtype=torch.long),
            labels=torch.ones((4, 2), dtype=torch.long),
            logits_to_keep=1,
            target_audio=LongCatBatchSide(
                semantic_ids=torch.ones((4, 1), dtype=torch.long),
                semantic_mask=torch.ones((4, 1), dtype=torch.bool),
                acoustic_ids=torch.zeros((4, 1, 1), dtype=torch.long),
                acoustic_mask=torch.ones((4, 1), dtype=torch.bool),
            ),
        )
        logged: dict[str, torch.Tensor] = {}
        batch_sizes: dict[str, int] = {}

        def log(name: str, value: torch.Tensor, **kwargs: object) -> None:
            logged[name] = value.detach()
            batch_size = kwargs["batch_size"]
            if isinstance(batch_size, int):
                batch_sizes[name] = batch_size

        module.log = log  # type: ignore[method-assign]

        loss = module._loss(batch)

        self.assertTrue(torch.equal(loss, torch.tensor(3.625)))
        self.assertTrue(torch.equal(logged["loss/acoustic"], torch.tensor(3.25)))
        self.assertTrue(torch.equal(logged["loss/acoustic_t/0_025"], torch.tensor(1.0)))
        self.assertTrue(torch.equal(logged["loss/acoustic_t/025_050"], torch.tensor(2.0)))
        self.assertTrue(torch.equal(logged["loss/acoustic_t/050_075"], torch.tensor(3.0)))
        self.assertTrue(torch.equal(logged["loss/acoustic_t/075_100"], torch.tensor(4.0)))
        self.assertEqual(batch_sizes["loss/acoustic_t/075_100"], 5)

    def test_loss_logs_acoustic_condition_mean_and_std(self) -> None:
        model = ConditionStatsModel()
        module = SpeechToSpeechModule(
            model,
            TrainConfig(acoustic_loss_weight=0.5),
            bpe=object(),
            acoustic_feature_extractor=FakeFeatureExtractor(torch.ones((1, 2, 4))),
        )
        batch = CausalLMBatch(
            input_ids=torch.ones((1, 2), dtype=torch.long),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            labels=torch.ones((1, 2), dtype=torch.long),
            logits_to_keep=1,
            target_audio=LongCatBatchSide(
                semantic_ids=torch.ones((1, 2), dtype=torch.long),
                semantic_mask=torch.ones((1, 2), dtype=torch.bool),
                acoustic_ids=torch.zeros((1, 1, 2), dtype=torch.long),
                acoustic_mask=torch.ones((1, 2), dtype=torch.bool),
            ),
        )
        logged: dict[str, torch.Tensor] = {}

        def log(name: str, value: torch.Tensor, **kwargs: object) -> None:
            del kwargs
            logged[name] = value.detach()

        module.log = log  # type: ignore[method-assign]

        module._loss(batch)

        self.assertTrue(torch.equal(logged["condition/hidden_mean"], torch.tensor(3.5)))
        self.assertTrue(torch.allclose(logged["condition/hidden_std"], torch.tensor(2.2913)))
        self.assertTrue(torch.equal(logged["condition/time_mean"], torch.tensor(2.0)))
        self.assertTrue(torch.equal(logged["condition/time_std"], torch.tensor(1.0)))
        self.assertTrue(torch.equal(logged["condition/acoustic_mean"], torch.tensor(6.0)))
        self.assertTrue(torch.equal(logged["condition/acoustic_std"], torch.tensor(1.0)))

    def test_loss_uses_supervised_token_weighted_row_loss(self) -> None:
        module = SpeechToSpeechModule(RowLossModel(torch.tensor([2.0, 8.0])))
        batch = CausalLMBatch(
            input_ids=torch.tensor([[1, 2, 3], [4, 5, 0]]),
            attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
            labels=torch.tensor([[1, 2, 3], [4, IGNORE_INDEX, IGNORE_INDEX]]),
            logits_to_keep=3,
        )

        loss = module._loss(batch)

        self.assertTrue(torch.equal(loss, torch.tensor(3.5)))

    def test_task_loss_logs_reduce_ready_metrics_without_sync_dist(self) -> None:
        module, _ = _module_and_batch()
        batch = CausalLMBatch(
            input_ids=torch.ones((2, 3), dtype=torch.long),
            attention_mask=torch.ones((2, 3), dtype=torch.long),
            labels=torch.ones((2, 3), dtype=torch.long),
            logits_to_keep=1,
            task_family=torch.tensor(
                [TaskFamily.SOURCE_AR.id, TaskFamily.SOURCE_TO_TARGET.id],
                dtype=torch.long,
            ),
        )
        logged: dict[str, torch.Tensor] = {}
        sync_dist: dict[str, object] = {}

        def log(name: str, value: torch.Tensor, **kwargs: object) -> None:
            logged[name] = value.detach()
            sync_dist[name] = kwargs["sync_dist"]

        module.log = log  # type: ignore[method-assign]

        module._log_task_losses(
            batch,
            torch.tensor([2.0, 8.0]),
            torch.tensor([3.0, 1.0]),
            stage=None,
        )
        module._log_family_group_losses(
            batch,
            torch.tensor([2.0, 8.0]),
            torch.tensor([3.0, 1.0]),
            stage=None,
        )

        self.assertEqual(
            set(logged),
            {
                "loss/source_ar",
                "tokens/source_ar",
                "loss/source_to_target",
                "tokens/source_to_target",
                "loss/semantic_ar",
                "loss/translation",
            },
        )
        self.assertTrue(torch.equal(logged["loss/source_ar"], torch.tensor(2.0)))
        self.assertTrue(torch.equal(logged["tokens/source_ar"], torch.tensor(3.0)))
        self.assertTrue(torch.equal(logged["loss/source_to_target"], torch.tensor(8.0)))
        self.assertTrue(torch.equal(logged["tokens/source_to_target"], torch.tensor(1.0)))
        self.assertTrue(all(value is False for value in sync_dist.values()))

    def test_acoustic_feature_extractor_is_not_registered_as_submodule(self) -> None:
        model = JointLossModel()
        extractor = FakeModuleFeatureExtractor(torch.full((1, 2, 4), 3.0))
        module = SpeechToSpeechModule(
            model,
            TrainConfig(acoustic_loss_weight=0.5),
            bpe=object(),
            acoustic_feature_extractor=extractor,
        )

        self.assertIs(module.acoustic_feature_extractor, extractor)
        self.assertNotIn("acoustic_feature_extractor", module._modules)
        self.assertNotIn("_acoustic_feature_extractor", module._modules)
        parameter_ids = {id(parameter) for parameter in module.parameters()}
        self.assertNotIn(id(extractor.weight), parameter_ids)


def _module_and_batch(
    *,
    train: TrainConfig | None = None,
) -> tuple[SpeechToSpeechModule, CausalLMBatch]:
    tokenizer = MockTokenizer()
    model = Orchestrator(
        qwen3=MockQwen(),
        tokenizer=tokenizer,
        bpe_vocab_size=5,
        model_config=ModelConfig(backbone=QwenBackboneConfig(train=True)),
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
            warmup_ratio=0.25,
        ),
    ), batch


class JointLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.target_features: torch.Tensor | None = None
        self.target_mask: torch.Tensor | None = None
        self.forward_hidden_states = (
            torch.zeros((1, 2, 4)),
            torch.ones((1, 2, 4)),
        )
        self.forward_requested_hidden_states = False
        self.hidden_states: torch.Tensor | None = None

    def forward(
        self,
        batch: CausalLMBatch,
        *,
        return_hidden_states: bool = False,
    ) -> CausalLMOutputWithPast:
        del batch
        self.forward_requested_hidden_states = return_hidden_states
        hidden_states = self.forward_hidden_states if return_hidden_states else None
        return CausalLMOutputWithPast(
            loss=torch.tensor(2.0),
            logits=torch.empty(0),
            hidden_states=hidden_states,
        )

    def acoustic_flow_loss(
        self,
        batch: CausalLMBatch,
        bpe: object,
        target_features: torch.Tensor,
        *,
        hidden_states: torch.Tensor | None = None,
        target_mask: torch.Tensor,
        source_feature_extractor: object,
    ) -> torch.Tensor:
        del batch, bpe, source_feature_extractor
        self.target_features = target_features
        self.target_mask = target_mask
        self.hidden_states = hidden_states
        return torch.tensor(3.0)

    def semantic_accuracy(
        self,
        batch: CausalLMBatch,
        output: CausalLMOutputWithPast | None = None,
    ) -> torch.Tensor:
        del batch, output
        return torch.tensor(1.0)


class JointLossStatsModel(torch.nn.Module):
    def forward(
        self,
        batch: CausalLMBatch,
        *,
        return_hidden_states: bool = False,
    ) -> CausalLMOutputWithPast:
        del batch
        hidden_states = (torch.ones((4, 2, 1)),) if return_hidden_states else None
        return CausalLMOutputWithPast(
            loss=torch.tensor(2.0),
            logits=torch.empty(0),
            hidden_states=hidden_states,
        )

    def acoustic_flow_loss_stats(
        self,
        batch: CausalLMBatch,
        bpe: object,
        target_features: torch.Tensor,
        *,
        hidden_states: torch.Tensor | None = None,
        target_mask: torch.Tensor,
        source_feature_extractor: object,
    ) -> AcousticFlowLossStats:
        del batch, bpe, target_features, hidden_states, target_mask, source_feature_extractor
        row_loss = torch.tensor([1.0, 2.0, 3.0, 4.0])
        row_weight = torch.tensor([1.0, 1.0, 1.0, 5.0])
        return AcousticFlowLossStats(
            loss=(row_loss * row_weight).sum() / row_weight.sum(),
            timesteps=torch.tensor([0.1, 0.3, 0.7, 1.0]),
            row_loss=row_loss,
            row_weight=row_weight,
        )


class ConditionStatsModel(torch.nn.Module):
    def forward(
        self,
        batch: CausalLMBatch,
        *,
        return_hidden_states: bool = False,
    ) -> CausalLMOutputWithPast:
        del batch
        hidden_states = (torch.ones((1, 2, 4)),) if return_hidden_states else None
        return CausalLMOutputWithPast(
            loss=torch.tensor(2.0),
            logits=torch.empty(0),
            hidden_states=hidden_states,
        )

    def acoustic_flow_inputs(
        self,
        batch: CausalLMBatch,
        bpe: object,
        target_features: torch.Tensor,
        *,
        hidden_states: torch.Tensor | None = None,
        target_mask: torch.Tensor,
        noise: torch.Tensor | None = None,
        acoustic_condition: torch.Tensor | None = None,
        source_feature_extractor: object,
    ) -> AcousticFlowInputs:
        del batch, bpe, target_features, hidden_states, target_mask
        del noise, acoustic_condition, source_feature_extractor
        return AcousticFlowInputs(
            target_features=torch.zeros((1, 2, 4)),
            noise=torch.zeros((1, 2, 4)),
            last_hidden_state=torch.zeros((1, 2, 4)),
            acoustic_condition=torch.zeros((1, 4)),
            mask=torch.tensor([[True, True]]),
        )

    def acoustic_flow_loss_stats_from_inputs(
        self,
        inputs: AcousticFlowInputs,
        *,
        timesteps: torch.Tensor | None = None,
    ) -> AcousticFlowLossStats:
        del inputs, timesteps
        return AcousticFlowLossStats(
            loss=torch.tensor(3.0),
            timesteps=torch.tensor([0.25]),
            row_loss=torch.tensor([3.0]),
            row_weight=torch.tensor([2.0]),
        )

    def acoustic_condition_tensors(
        self,
        inputs: AcousticFlowInputs,
        *,
        timesteps: torch.Tensor,
    ) -> DiTConditionTensors:
        del inputs, timesteps
        return DiTConditionTensors(
            time=torch.tensor([[[1.0, 1.0, 3.0, 3.0]]]),
            hidden=torch.tensor([[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]]),
            acoustic=torch.tensor([[[5.0, 5.0, 7.0, 7.0]]]),
        )

    def semantic_accuracy(
        self,
        batch: CausalLMBatch,
        output: CausalLMOutputWithPast | None = None,
    ) -> torch.Tensor:
        del batch, output
        return torch.tensor(1.0)


class RowLossModel(torch.nn.Module):
    def __init__(self, loss: torch.Tensor) -> None:
        super().__init__()
        self.loss = loss

    def forward(
        self,
        batch: CausalLMBatch,
        *,
        return_hidden_states: bool = False,
    ) -> CausalLMOutputWithPast:
        del batch, return_hidden_states
        return CausalLMOutputWithPast(loss=self.loss, logits=torch.empty(0))


class FakeFeatureExtractor:
    def __init__(self, features: torch.Tensor) -> None:
        self.features = features

    def acoustic_codes_to_features(self, acoustic_ids: torch.Tensor) -> torch.Tensor:
        del acoustic_ids
        return self.features


class FakeModuleFeatureExtractor(torch.nn.Module):
    def __init__(self, features: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.features = features
        self.device = torch.device("cpu")

    def acoustic_codes_to_features(self, acoustic_ids: torch.Tensor) -> torch.Tensor:
        del acoustic_ids
        return self.features.to(device=self.device)


if __name__ == "__main__":
    unittest.main()
