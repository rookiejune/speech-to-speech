from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _contracts_helpers import *


class LoaderScheduleContractTest(unittest.TestCase):
    def test_scheduled_dataloader_rotates_homogeneous_loaders_by_weight(self):
        speech = ModelBatch.from_samples([_sample(Task.TTS)], pad_token_id=99)
        mt = ModelBatch.from_samples([_sample(Task.MT)], pad_token_id=99)
        loader = ScheduledDataLoader(
            {"speech": [speech], "mt": [mt]},
            LoaderSchedule({"speech": 2.0, "mt": 1.0}),
        )

        iterator = iter(loader)
        tasks = [next(iterator).tasks[0] for _ in range(6)]

        self.assertEqual(
            tasks,
            [Task.TTS, Task.MT, Task.TTS, Task.TTS, Task.MT, Task.TTS],
        )
    def test_scheduled_dataloader_interleaves_one_accumulation_window(self):
        speech = ModelBatch.from_samples([_sample(Task.TTS)], pad_token_id=99)
        mt = ModelBatch.from_samples([_sample(Task.MT)], pad_token_id=99)
        with self.assertRaisesRegex(ValueError, "too small"):
            LoaderSchedule(
                {"speech": 9.0, "mt": 1.0},
                accumulate_grad_batches=8,
            )
        loader = ScheduledDataLoader(
            {"speech": [speech], "mt": [mt]},
            LoaderSchedule(
                {"speech": 2.0, "mt": 1.0},
                accumulate_grad_batches=3,
            ),
        )

        batches = list(islice(loader, 3))

        self.assertTrue(all(isinstance(batch, ModelBatch) for batch in batches))
        self.assertEqual(
            [batch.tasks[0] for batch in batches],
            [Task.TTS, Task.MT, Task.TTS],
        )
    def test_datamodule_sets_up_loaders_and_returns_scheduled_loader(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(32)
        speech = LoaderSpec.speech(
            SpeechConfig(
                codec="longcat",
                dataloader=_loader(),
                dataset=DatasetConfig(
                    name=DatasetName.TOY,
                    toy_samples=1,
                    toy_frames=2,
                ),
            ),
            {Task.TTS: 1.0},
        )
        mt = LoaderSpec.text(
            TextConfig(
                dataloader=_loader(),
                dataset=TextDatasetConfig(
                    name=TextDatasetName.TOY,
                    toy_samples=1,
                ),
            ),
            {Task.MT: 1.0},
        )
        datamodule = DataModule(
            runtime,
            {"speech": speech, "mt": mt},
            LoaderSchedule(
                {"speech": 1.0, "mt": 1.0},
                accumulate_grad_batches=2,
            ),
        )

        datamodule.setup("fit")
        loader = datamodule.train_dataloader()
        iterator = iter(loader)

        self.assertEqual(datamodule.schedule.accumulate_grad_batches, 2)
        batches = [next(iterator), next(iterator)]
        self.assertEqual([batch.tasks[0] for batch in batches], [Task.TTS, Task.MT])
    def test_datamodule_validates_loader_names(self):
        runtime = _data_runtime()
        speech = LoaderSpec.speech(
            SpeechConfig(
                codec="longcat",
                dataloader=_loader(),
                dataset=DatasetConfig(name=DatasetName.TOY),
            ),
            {Task.TTS: 1.0},
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            DataModule(
                runtime,
                {"speech": speech},
                LoaderSchedule({"speech": 1.0, "mt": 1.0}),
            )
        with self.assertRaisesRegex(ValueError, "finite positive"):
            LoaderSchedule({"speech": 0.0, "mt": 0.0})


if __name__ == "__main__":
    unittest.main()
