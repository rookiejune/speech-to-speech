from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _contracts_helpers import *


class TextDataContractTest(unittest.TestCase):
    def test_text_parser_ignores_audio_fields(self):
        tokenizer = _Tokenizer(10)
        runtime = SimpleNamespace(text_tokenizer=tokenizer)

        pair = parse_text_sample(_raw_text_sample(), runtime)

        self.assertTrue(torch.equal(pair.source.text_token_ids, torch.tensor([1, 2])))
        self.assertIs(pair.source.language, Language.ZH)
        self.assertIs(pair.target.language, Language.EN)
        self.assertEqual(tokenizer.encoded, ("target text", False))

    def test_text_collator_builds_mt_batches_without_audio_runtime(self):
        runtime = SimpleNamespace(
            text_tokenizer=_ChatTokenizer(32),
            layout=Layout(text=(0, 32), audio=(32, 36)),
            pad_token_id=0,
            eos_token_id=31,
        )

        batch = TextCollator(runtime, {Task.MT: 1.0})([_raw_text_sample()])

        self.assertEqual(batch.tasks, [Task.MT])
        self.assertIsNone(batch.acoustic_target)
        self.assertTrue(batch.token_labels.ne(-100).any())
        labels = batch.token_labels[batch.token_labels.ne(-100)]
        self.assertTrue((labels >= 0).all())
        self.assertTrue((labels < 32).all())

    def test_text_collator_rejects_audio_tasks(self):
        runtime = SimpleNamespace(
            text_tokenizer=_ChatTokenizer(32),
            layout=Layout(text=(0, 32), audio=(32, 36)),
            pad_token_id=0,
            eos_token_id=31,
        )

        with self.assertRaisesRegex(ValueError, "text-only"):
            TextCollator(runtime, {Task.TTS: 1.0})

    @patch("anydataset.presets.WMT19")
    def test_text_dataset_config_loads_anydataset_wmt19(self, wmt19):
        config = TextDatasetConfig(
            name=TextDatasetName.WMT19,
            split="validation",
            source_lang="de",
            target_lang="en",
        )

        loaded = load_text_dataset(config)

        self.assertIs(loaded, wmt19.return_value)
        wmt19.assert_called_once_with(
            split="validation",
            source_lang="de",
            target_lang="en",
        )

    def test_text_datamodule_reads_toy_text_without_codec_runtime(self):
        runtime = SimpleNamespace(
            text_tokenizer=_ChatTokenizer(32),
            layout=Layout(text=(0, 32), audio=(32, 36)),
            pad_token_id=0,
            eos_token_id=31,
        )
        datamodule = DataModule(
            runtime,
            {
                "mt": LoaderSpec.text(
                    TextConfig(
                        dataloader=_loader(2),
                        dataset=TextDatasetConfig(
                            name=TextDatasetName.TOY,
                            toy_samples=2,
                        ),
                    ),
                    {Task.MT: 1.0},
                )
            },
        )

        datamodule.setup()
        batch = next(iter(datamodule.train_dataloader()))

        self.assertEqual(batch.input_ids.size(0), 2)
        self.assertEqual(batch.tasks, [Task.MT, Task.MT])
        self.assertIsNone(batch.acoustic_target)

    def test_text_validation_dataloader_limits_samples(self):
        runtime = SimpleNamespace(
            text_tokenizer=_ChatTokenizer(32),
            layout=Layout(text=(0, 32), audio=(32, 36)),
            pad_token_id=0,
            eos_token_id=31,
        )
        text_config = TextConfig(
            dataloader=_loader(4),
            dataset=TextDatasetConfig(
                name=TextDatasetName.TOY,
                toy_samples=5,
            ),
        )
        datamodule = DataModule(
            runtime,
            {"mt": LoaderSpec.text(text_config, {Task.MT: 1.0})},
            validation=LoaderSpec.text(
                text_config,
                {Task.MT: 1.0},
                max_samples=2,
            ),
        )

        datamodule.setup()
        batches = list(datamodule.val_dataloader())

        self.assertEqual(sum(batch.input_ids.size(0) for batch in batches), 2)
        self.assertTrue(
            all(task is Task.MT for batch in batches for task in batch.tasks)
        )



if __name__ == "__main__":
    unittest.main()
