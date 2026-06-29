from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from speech_to_speech.config import ModelConfig
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
        self.assertEqual(config.tasks.enabled, ("autoregression",))
        self.assertEqual(config.bpe.vocab_size, 10000)
        self.assertEqual(config.train.max_steps, 100)

    def test_load_config_reads_wmt19_mixed_smoke_experiment(self) -> None:
        config = smoke.load_config(overrides=("experiment=wmt19_mixed_smoke",))

        self.assertEqual(config.datamodule.dataloader.batch_size, 4)
        self.assertEqual(config.tasks.enabled, ("autoregression", "translation"))
        self.assertEqual(config.tasks.weights.source_ar, 1.0)
        self.assertEqual(config.tasks.weights.source_to_target, 1.0)
        self.assertTrue(config.model.load_in_4bit)
        self.assertTrue(config.model.lora.enabled)
        self.assertEqual(config.model.acoustic_condition_dropout, 0.1)
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
        self.assertEqual(config.model.acoustic_condition_dropout, 0.1)

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
                "model.train_dit=false",
                "model.acoustic_condition_dropout=0.25",
                "model.lora.enabled=false",
                "train.optimizer=adamw",
                "train.device=cpu",
                "train.precision=32-true",
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
                FakeTrainer.last_fit_module.model.model_config.acoustic_condition_dropout,
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
        dit: object | None = None,
        model_config: ModelConfig,
        bpe_config: object,
        tokenizer: object,
        bpe_vocab_size: int,
    ) -> None:
        del dit, bpe_config, tokenizer
        self.model_config = model_config
        self.embed_tokens = toy_embedding(audio_vocab_size=bpe_vocab_size)


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
