from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from hydra import compose, initialize_config_dir

from scripts._mimo_train_config import MimoTrainConfig, parse
from scripts.mimo_train import mimo_callbacks, run


class MimoTrainEntryTest(unittest.TestCase):
    def _compose(self, *overrides: str):
        root = Path(__file__).parents[1]
        with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
            return compose(config_name="mimo_train", overrides=list(overrides))

    def test_structured_cpu_smoke_config_parses(self) -> None:
        config = parse(
            self._compose(
                "model.toy=true",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "trainer.precision=32-true",
                "train.max_steps=1",
            )
        )

        self.assertIsInstance(config, MimoTrainConfig)
        self.assertTrue(config.model.toy)
        self.assertEqual(config.data.kind, "segments")
        self.assertEqual(config.trainer.accelerator, "cpu")

    def test_cpu_one_step_uses_only_mimo_safe_callbacks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = parse(
                self._compose(
                    "model.toy=true",
                    "trainer.accelerator=cpu",
                    "trainer.devices=1",
                    "trainer.precision=32-true",
                    "trainer.log_every_n_steps=1",
                    "trainer.enable_checkpointing=false",
                    "callbacks.checkpoint.enabled=false",
                    "train.max_steps=1",
                    "data.samples_per_epoch=8",
                    "dataloader.batch_size=2",
                    f"repo_output_root={directory}",
                    "output_subdir=mimo-smoke",
                    f"output_dir={directory}/mimo-smoke",
                )
            )
            callbacks = mimo_callbacks(config)
            self.assertEqual(callbacks, [])
            result = run(config)

        self.assertEqual(result["mode"], "mimo")
        self.assertEqual(result["global_step"], 1)


if __name__ == "__main__":
    unittest.main()
