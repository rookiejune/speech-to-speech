from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _contracts_helpers import *


class OverfitDataModuleContractTest(unittest.TestCase):
    @patch("speech_to_speech.datamodule.module.load_dataset")
    def test_datamodule_rejects_enabled_costs_for_fixed_sample_loader(
        self,
        load_dataset,
    ):
        load_dataset.return_value = [_raw_sample(), _raw_sample()]
        runtime = _data_runtime()
        config = SpeechConfig(
            codec="longcat",
            dataloader=DataLoaderConfig(
                batch_size=1,
                num_workers=0,
                costs=DataLoaderCostsConfig(
                    enabled=True,
                    max_batch_frames=8,
                ),
            ),
        )
        datamodule = DataModule(
            runtime,
            {
                "train": LoaderSpec.speech(
                    config,
                    {Task.TTS: 1.0},
                    sample_index=0,
                )
            },
        )
        datamodule.setup()
        with self.assertRaisesRegex(ValueError, "fixed-sample"):
            datamodule.train_dataloader()
    def test_overfit_datamodule_repeats_only_the_selected_sample(self):
        samples = [object(), object()]
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(),
        )
        collator = Mock(side_effect=lambda batch: batch)
        with (
            patch(
                "speech_to_speech.datamodule.module.load_dataset",
                return_value=samples,
            ) as load_dataset,
            patch("speech_to_speech.datamodule.module._collator", return_value=collator),
        ):
            datamodule = DataModule(
                _data_runtime(),
                {
                    "train": LoaderSpec.speech(
                        config,
                        {Task.TTS: 1.0},
                        sample_index=1,
                    )
                },
            )

            datamodule.setup()
            first_epoch = list(datamodule.train_dataloader())
            second_epoch = list(datamodule.train_dataloader())

        load_dataset.assert_called_once()
        self.assertEqual(first_epoch, [[samples[1]]])
        self.assertEqual(second_epoch, [[samples[1]]])
    def test_overfit_datamodule_rejects_runtime_codec_mismatch(self):
        config = SpeechConfig(
            codec="unicodec",
            dataloader=_loader(),
        )
        datamodule = DataModule(
            _data_runtime(),
            {
                "train": LoaderSpec.speech(
                    config,
                    {Task.TTS: 1.0},
                    sample_index=0,
                )
            },
        )

        with self.assertRaisesRegex(ValueError, "same codec"):
            datamodule.setup()


if __name__ == "__main__":
    unittest.main()
