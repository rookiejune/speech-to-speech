from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

import torch

from _contracts_helpers import *
from speech_to_speech.datamodule.loader import (
    LoaderConfig,
    LoaderPlanConfig,
    LoaderStepMode,
    count_supervised_tokens,
)
from speech_to_speech.datamodule.mimo import MimoBatch


class LoaderScheduleContractTest(unittest.TestCase):
    def test_token_weighted_tracks_supervised_tokens_not_microbatches(self):
        text = _token_batch(Task.MT, 5)
        audio_route = _token_batch(Task.ASR, 1)
        loader = ScheduledDataLoader(
            {"long": [text], "short": [audio_route]},
            LoaderSchedule(
                {"long": 1.0, "short": 1.0},
                accumulate_grad_batches=8,
                step_mode="token_weighted",
            ),
        )

        totals = {"long": 0, "short": 0}
        selected_names: list[str] = []
        batches = iter(loader)
        for _ in range(100):
            batch = next(batches)
            name = "long" if batch.tasks[0] is Task.MT else "short"
            selected_names.append(name)
            totals[name] += count_supervised_tokens(batch)

        self.assertLessEqual(abs(totals["long"] - totals["short"]), 5)
        self.assertNotEqual(selected_names[:20].count("long"), 10)
        self.assertEqual(text.supervised_token_count, 5)
        self.assertEqual(audio_route.supervised_token_count, 1)

    def test_token_weighted_supports_mimo_shifted_target_masks(self):
        batch = MimoBatch(
            text_input_ids=torch.tensor([[1, 2, 3]]),
            audio_input_ids=torch.tensor([[4, 5, 6]]),
            text_labels=torch.tensor([[-100, 7, -100]]),
            audio_labels=torch.tensor([[-100, -100, 8]]),
        )

        self.assertEqual(batch.supervised_token_count, 2)
        self.assertEqual(count_supervised_tokens(batch), 2)

        single = _token_batch(Task.MT, 1)
        single.token_labels[0, 0] = -7
        self.assertEqual(count_supervised_tokens(single, ignore_index=-7), 1)

    def test_token_weighted_accepts_custom_counter_and_rejects_fused(self):
        class Batch:
            def __init__(self, tokens: int) -> None:
                self.tokens = tokens

            @property
            def supervised_token_count(self) -> int:
                return self.tokens

        loader = ScheduledDataLoader(
            {"a": [Batch(2)], "b": [Batch(1)]},
            LoaderSchedule({"a": 1.0, "b": 1.0}, step_mode="token_weighted"),
        )
        self.assertEqual(next(iter(loader)).tokens, 2)

        with self.assertRaisesRegex(ValueError, "requires fuse_loaders_per_step=false"):
            LoaderSchedule(
                {"a": 1.0, "b": 1.0},
                step_mode="token_weighted",
                fuse_loaders_per_step=True,
            )

    def test_token_weighted_plan_allows_multiple_tasks_per_loader(self):
        plan = LoaderPlanConfig(
            loaders={
                "speech": LoaderConfig(
                    weight=1.0,
                    task_weights={"asr": 1.0, "s2tt": 1.0},
                )
            },
            step_mode="token_weighted",
        )

        self.assertIs(plan.mode, LoaderStepMode.TOKEN_WEIGHTED)
        self.assertFalse(plan.fuse_loaders_per_step)

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


def _token_batch(task: Task, count: int) -> ModelBatch:
    if count < 1:
        raise ValueError(count)
    return ModelBatch(
        input_ids=torch.arange(count + 1, dtype=torch.long).unsqueeze(0),
        token_labels=torch.cat(
            [
                torch.tensor([[-100]], dtype=torch.long),
                torch.ones((1, count), dtype=torch.long),
            ],
            dim=1,
        ),
        acoustic_target=None,
        tasks=[task],
        predictions=[task.prediction_modality],
        pad_token_id=0,
    )


if __name__ == "__main__":
    unittest.main()
