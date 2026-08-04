from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from scripts._config.overfit import overfit
from scripts._config.train import train


@patch.dict(
    "os.environ",
    {
        "DYNAMIC_HOME": "/tmp/dynamic",
        "SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer",
    },
)
class ConfigSafetyTest(unittest.TestCase):
    def test_train_hydra_metadata_uses_the_training_output(self) -> None:
        config = _compose("train", return_hydra_config=True)

        self.assertFalse(config.hydra.job.chdir)
        self.assertEqual(config.hydra.run.dir, f"{config.output_dir}/hydra")
        self.assertEqual(config.hydra.output_subdir, "config")

    def test_train_trainer_presets_are_not_overridden_by_the_entry(self) -> None:
        cases = {
            "staged_static_ddp": ("ddp_find_unused_parameters_false", False),
            "staged_ddp": ("ddp_find_unused_parameters_true", False),
            "default": ("auto", True),
            "ddp": ("ddp_find_unused_parameters_true", True),
            "static_ddp": ("ddp_find_unused_parameters_false", True),
        }

        for preset, (strategy, sampler) in cases.items():
            with self.subTest(preset=preset):
                config = _compose("train", f"trainer={preset}")
                self.assertEqual(config.trainer.strategy, strategy)
                self.assertIs(config.trainer.use_distributed_sampler, sampler)

    def test_train_rejects_invalid_execution_values(self) -> None:
        cases = {
            "train.max_steps=0": "train.max_steps",
            "train.seed=-1": "train.seed",
            "trainer.log_every_n_steps=0": "trainer.log_every_n_steps",
            "callbacks.task_sample.seed=-1": "callbacks.task_sample.seed",
            "callbacks.task_sample.every_n_steps=0": (
                "callbacks.task_sample.every_n_steps"
            ),
            "callbacks.task_sample.max_new_tokens=0": (
                "callbacks.task_sample.max_new_tokens"
            ),
            "callbacks.checkpoint.every_n_train_steps=0": (
                "callbacks.checkpoint.every_n_train_steps"
            ),
            "callbacks.performance.log_every_n_steps=0": (
                "callbacks.performance.log_every_n_steps"
            ),
            "callbacks.performance.warmup_steps=-1": (
                "callbacks.performance.warmup_steps"
            ),
            "callbacks.performance.measure_window_steps=0": (
                "callbacks.performance.measure_window_steps"
            ),
            "optim.name=sgd": "optim.name",
            "optim.learning_rate=0": "optim.learning_rate",
            "optim.weight_decay=-1": "optim.weight_decay",
            "optim.schedule.unit=bad-unit": "optim.schedule.unit",
            "optim.schedule.measure_window_batches=0": (
                "optim.schedule.measure_window_batches"
            ),
            "optim.schedule.phases=[]": "optim.schedule requires phases",
        }

        for override, message in cases.items():
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    train(_compose("train", override))

    def test_overfit_rejects_invalid_execution_values(self) -> None:
        cases = {
            "train.max_steps=0": "train.max_steps",
            "train.seed=-1": "train.seed",
            "sample_index=-1": "sample_index",
            "callbacks.task_sample.every_n_steps=0": (
                "callbacks.task_sample.every_n_steps"
            ),
        }

        for override, message in cases.items():
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    overfit(_compose("overfit", override))

    def test_train_performance_rejects_task_sample_generation(self) -> None:
        with self.assertRaisesRegex(ValueError, "train performance requires"):
            train(
                _compose(
                    "train",
                    "callback/parameter_policy@callbacks.parameter_policy=speech_interface",
                    "model.lora=null",
                    "callbacks.performance.enabled=true",
                    "callbacks.task_sample.enabled=true",
                )
            )


def _compose(
    config_name: str,
    *overrides: str,
    return_hydra_config: bool = False,
) -> DictConfig:
    root = Path(__file__).parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        return compose(
            config_name=config_name,
            overrides=list(overrides),
            return_hydra_config=return_hydra_config,
        )


if __name__ == "__main__":
    unittest.main()
