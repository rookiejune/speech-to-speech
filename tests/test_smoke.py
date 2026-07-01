from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from speech_to_speech.config import (
    AcousticAttentionMode,
    AcousticConditionSource,
    AdapterType,
    AudioEmbeddingType,
    ModelConfig,
    ModelTrainMode,
)
from speech_to_speech import smoke
from helpers import (
    MockFrameBPE,
    MockTokenizer,
    isolated_anydataset_home,
    patched_wmt19_longcat,
    toy_embedding,
    write_toy_longcat_store,
)


class SmokeRunnerTest(unittest.TestCase):
    def test_load_config_reads_wmt19_ar_smoke_experiment(self) -> None:
        config = smoke.load_config(overrides=("experiment=wmt19_ar_smoke",))

        self.assertEqual(config.datamodule.dataset_factory.name, "wmt19_tts_longcat")
        self.assertEqual(config.datamodule.dataloader.batch_size, 1)
        self.assertEqual(config.datamodule.dataloader.num_workers, 8)
        self.assertEqual(config.tasks.enabled, ("autoregression",))
        self.assertEqual(config.bpe.vocab_size, 10000)
        self.assertEqual(config.train.max_steps, 5_000_000)
        self.assertEqual(config.train.schedule, "warmup_cosine")
        self.assertEqual(config.train.warmup_ratio, 0.01)
        self.assertEqual(config.train.device, "auto")
        self.assertEqual(config.trainer.callbacks.checkpoint.every_n_steps, 10_000)
        self.assertEqual(config.trainer.callbacks.sample.every_n_steps, 0)
        generation = config.trainer.callbacks.generation
        self.assertEqual(generation.every_n_steps, 5_000)
        self.assertEqual(generation.flow_steps, 32)
        self.assertIsNone(generation.chunk_size)
        self.assertIsNone(generation.left_context_chunks)
        self.assertEqual(generation.acoustic_sampler, "serial")

    def test_load_config_reads_wmt19_mixed_smoke_experiment(self) -> None:
        config = smoke.load_config(overrides=("experiment=wmt19_mixed_smoke",))

        self.assertEqual(config.datamodule.dataloader.batch_size, 4)
        self.assertEqual(config.tasks.enabled, ("autoregression", "translation"))
        self.assertEqual(config.tasks.weights.source_ar, 1.0)
        self.assertEqual(config.tasks.weights.source_to_target, 1.0)
        self.assertTrue(config.model.backbone.load_in_4bit)
        self.assertTrue(config.model.backbone.lora.enabled)
        self.assertEqual(config.model.acoustic.condition_dropout, 0.1)
        self.assertFalse(config.model.acoustic.enabled)
        self.assertFalse(config.model.acoustic.dit.norm_time)
        self.assertFalse(config.model.acoustic.dit.norm_hidden)
        self.assertFalse(config.model.acoustic.dit.norm_acoustic)
        self.assertEqual(config.train.acoustic_loss_weight, 0.0)

    def test_load_config_reads_stage_task_weights(self) -> None:
        config = smoke.load_config(
            overrides=(
                "experiment=wmt19_mixed_smoke",
                "tasks=s2_translation_weighted",
            )
        )

        self.assertEqual(config.tasks.weights.source_ar, 1.0)
        self.assertEqual(config.tasks.weights.target_ar, 2.0)
        self.assertEqual(config.tasks.weights.source_to_target, 1.0)
        self.assertEqual(config.tasks.weights.target_to_source, 0.25)

    def test_load_config_reads_wmt19_acoustic_smoke_experiment(self) -> None:
        config = smoke.load_config(overrides=("experiment=wmt19_acoustic_smoke",))

        self.assertEqual(config.tasks.enabled, ("translation",))
        self.assertEqual(config.datamodule.dataloader.batch_size, 4)
        self.assertEqual(config.train.acoustic_loss_weight, 0.01)
        self.assertEqual(config.model.acoustic.condition_dropout, 0.1)
        self.assertTrue(config.model.acoustic.enabled)
        self.assertIs(config.model.acoustic.attention_mode, AcousticAttentionMode.CAUSAL)

    def test_load_config_reads_target_embedding_acoustic_ablation(self) -> None:
        config = smoke.load_config(
            overrides=("experiment=wmt19_acoustic_target_embed_100k_muon",)
        )

        self.assertEqual(config.tasks.enabled, ("autoregression", "translation"))
        self.assertIs(config.model.train_mode, ModelTrainMode.ACOUSTIC_ONLY)
        self.assertIs(
            config.model.acoustic.condition_source,
            AcousticConditionSource.TARGET_AUDIO_EMBEDDING,
        )
        self.assertIs(
            config.model.token_space.audio_embedding_type,
            AudioEmbeddingType.SEMANTIC_COMPOSITION,
        )
        self.assertEqual(config.train.semantic_loss_weight, 0.0)
        self.assertEqual(config.train.acoustic_loss_weight, 1.0)
        self.assertTrue(config.model.acoustic.enabled)

    def test_load_config_reads_dit_overrides(self) -> None:
        config = smoke.load_config(
            overrides=(
                "experiment=wmt19_acoustic_smoke",
                "model.acoustic.condition_adapter.type=linear",
                "model.acoustic.condition_encoder.enabled=true",
                "model.acoustic.condition_encoder.num_hidden_layers=1",
                "model.acoustic.attention_mode=bidirectional",
                "model.acoustic.dit.norm_hidden=true",
                "model.acoustic.dit.norm_acoustic=true",
            )
        )

        self.assertIs(config.model.acoustic.condition_adapter.type, AdapterType.LINEAR)
        self.assertTrue(config.model.acoustic.condition_encoder.enabled)
        self.assertEqual(config.model.acoustic.condition_encoder.num_hidden_layers, 1)
        self.assertIs(config.model.acoustic.attention_mode, AcousticAttentionMode.BIDIRECTIONAL)
        self.assertFalse(config.model.acoustic.dit.norm_time)
        self.assertTrue(config.model.acoustic.dit.norm_hidden)
        self.assertTrue(config.model.acoustic.dit.norm_acoustic)

    def test_load_config_rejects_unknown_acoustic_attention_mode(self) -> None:
        with self.assertRaises(ValueError):
            smoke.load_config(
                overrides=(
                    "experiment=wmt19_acoustic_smoke",
                    "model.acoustic.attention_mode=offline",
                )
            )

    def test_acoustic_training_requires_acoustic_decoder(self) -> None:
        config = smoke.load_config(
            overrides=(
                "experiment=wmt19_acoustic_smoke",
                "model.acoustic.enabled=false",
            )
        )

        with self.assertRaisesRegex(ValueError, "model.acoustic.enabled"):
            smoke._validate_acoustic_training_model(config.model, config.train)

    def test_unknown_model_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown ModelConfig field"):
            smoke.load_config(
                overrides=(
                    "experiment=wmt19_acoustic_smoke",
                    "+model.unknown=false",
                )
            )

    def test_unknown_trainer_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown TrainerConfig field"):
            smoke.load_config(
                overrides=(
                    "experiment=wmt19_acoustic_smoke",
                    "+trainer.unknown=2",
                )
            )

    def test_unknown_trainer_callback_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown TrainerCallbacksConfig field"):
            smoke.load_config(
                overrides=(
                    "experiment=wmt19_acoustic_smoke",
                    "+trainer.callbacks.unknown.enabled=true",
                )
            )

    def test_run_smoke_wires_real_dataset_to_trainer(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = write_toy_longcat_store(
                root / "store",
                pairs=(([0, 1, 2], [2, 1]),),
            )
            overrides = (
                "experiment=wmt19_ar_smoke",
                "bpe.vocab_size=16",
                "bpe.max_token_length=4",
                "model.acoustic.train=false",
                "model.acoustic.condition_dropout=0.25",
                "model.backbone.lora.enabled=false",
                "train.optimizer=adamw",
                "train.device=cpu",
                "train.precision=32-true",
                "datamodule.dataloader.num_workers=0",
            )
            with isolated_anydataset_home(root):
                with patched_wmt19_longcat(store):
                    with patch.object(smoke, "qwen3_tokenizer", return_value=MockTokenizer()) as qwen:
                        with patch.object(
                            smoke,
                            "prepare_longcat_tokenizer",
                            return_value=MockFrameBPE(),
                        ):
                            with patch.object(smoke, "Orchestrator", FakeOrchestrator):
                                with patch.object(smoke, "SpeechToSpeechModule", FakeModule):
                                    with patch.object(smoke, "Trainer", FakeTrainer):
                                        with patch.object(smoke, "seed_everything") as seed:
                                            trainer = smoke.run_smoke(
                                                overrides=overrides,
                                                max_steps=1,
                                                default_root_dir=root / "outputs",
                                                enable_progress_bar=False,
                                            )

            self.assertIsInstance(trainer, FakeTrainer)
            seed.assert_called_once_with(0, workers=True)
            qwen.assert_called_once()
            self.assertIsInstance(qwen.call_args.args[0], ModelConfig)
            self.assertEqual(FakeTrainer.last_kwargs["max_steps"], 1)
            self.assertEqual(FakeTrainer.last_kwargs["accelerator"], "cpu")
            self.assertFalse(FakeTrainer.last_kwargs["enable_progress_bar"])
            self.assertEqual(FakeTrainer.last_fit_module.train.max_steps, 1)
            self.assertEqual(
                FakeTrainer.last_fit_module.model.model_config.acoustic.condition_dropout,
                0.25,
            )
            with isolated_anydataset_home(root):
                with patched_wmt19_longcat(store):
                    batch = next(iter(FakeTrainer.last_fit_datamodule.train_dataloader()))
            self.assertEqual(batch.input_ids.size(0), 1)


class FakeOrchestrator:
    def __init__(
        self,
        *,
        model_config: ModelConfig,
        bpe_config: object,
        tokenizer: object,
        bpe_vocab_size: int,
        space: object,
        bpe: object | None = None,
    ) -> None:
        del bpe_config, tokenizer, bpe
        self.model_config = model_config
        self.embed_tokens = toy_embedding(audio_vocab_size=bpe_vocab_size)
        self.idspace = space


class FakeModule:
    def __init__(
        self,
        model: FakeOrchestrator,
        train: object,
        *,
        bpe: object | None = None,
        acoustic_feature_extractor: object | None = None,
    ) -> None:
        del bpe, acoustic_feature_extractor
        self.model = model
        self.train = train


class FakeTrainer:
    last_kwargs: dict[str, object] = {}
    last_fit_module: FakeModule
    last_fit_datamodule: object

    def __init__(self, **kwargs: object) -> None:
        self.global_step = 0
        FakeTrainer.last_kwargs = kwargs

    def fit(self, module: FakeModule, *, datamodule: object) -> None:
        FakeTrainer.last_fit_module = module
        FakeTrainer.last_fit_datamodule = datamodule
        self.global_step = 1


if __name__ == "__main__":
    unittest.main()
