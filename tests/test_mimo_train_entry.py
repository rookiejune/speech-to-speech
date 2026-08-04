from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from hydra import compose, initialize_config_dir

from scripts._config.mimo import MimoTrainConfig, parse
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

    def test_formal_kimi_preset_declares_runtime_and_data_contract(self) -> None:
        config = parse(self._compose())

        self.assertFalse(config.model.toy)
        self.assertEqual(
            config.data.factory,
            "speech_to_speech.datamodule.mimo.dataset:JsonlMimoSegmentDataset",
        )
        self.assertIsNone(config.model.audio_embedding_dim)
        self.assertEqual(config.data.max_sequence_length, 4096)
        self.assertEqual(config.model.text_readout, "last_hidden_state[1]")
        self.assertEqual(config.model.audio_readout, "last_hidden_state[0]")
        self.assertEqual(config.data.audio_delay_tokens, 5)
        self.assertTrue(config.data.derive_special_tokens)

    def test_toy_override_switches_prepared_factory_for_cpu_smoke(self) -> None:
        config = parse(
            self._compose(
                "model.toy=true",
                "trainer.accelerator=cpu",
                "trainer.devices=1",
                "trainer.precision=32-true",
                "train.max_steps=1",
            )
        )

        # The production config remains JSONL, but the entry resolves a toy
        # source automatically when the explicit model.toy override is used.
        from scripts.mimo_train import _data_for_model

        data = _data_for_model(config, None)
        self.assertEqual(
            data.factory,
            "speech_to_speech.datamodule.mimo.dataset:ToyMimoSegmentDataset",
        )
        self.assertFalse(data.derive_special_tokens)

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
                    "dataloader.num_workers=0",
                    "dataloader.pin_memory=false",
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
