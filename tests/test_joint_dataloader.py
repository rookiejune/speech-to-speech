from __future__ import annotations

import unittest
from collections.abc import Iterator
from itertools import islice
from unittest.mock import Mock, patch

import torch
from anydataset import IterableAnyDataset

from speech_to_speech.datamodule.dataset import DatasetConfig
from speech_to_speech.datamodule.joint import LoaderSchedule, ScheduledDataLoader
from speech_to_speech.datamodule.config import DataLoaderConfig, SpeechConfig
from speech_to_speech.datamodule.module import DataModule, LoaderSpec
from speech_to_speech.datamodule.diagnostic import SampleSplit
from speech_to_speech.datamodule.text import TextConfig
from speech_to_speech.datamodule.types import ModelBatch
from speech_to_speech.task import Task


class _BatchSampler:
    def __init__(self) -> None:
        self.epoch = 0
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.epochs.append(epoch)


class _DirectEpochLoader:
    def __init__(self, batch: ModelBatch, *, batches: int = 1) -> None:
        self.batch = batch
        self.batches = batches
        self.epoch = 0
        self.epochs: list[int] = []
        self.iteration_epochs: list[int] = []
        self.batch_sampler = _BatchSampler()

    def __iter__(self) -> Iterator[ModelBatch]:
        self.iteration_epochs.append(self.epoch)
        yield from (self.batch for _ in range(self.batches))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.epochs.append(epoch)


class _BatchSamplerEpochLoader:
    def __init__(self, batch: ModelBatch, *, batches: int = 1) -> None:
        self.batch = batch
        self.batches = batches
        self.batch_sampler = _BatchSampler()
        self.iteration_epochs: list[int] = []

    def __iter__(self) -> Iterator[ModelBatch]:
        self.iteration_epochs.append(self.batch_sampler.epoch)
        yield from (self.batch for _ in range(self.batches))


class _IterableSamples(IterableAnyDataset):
    def __init__(self, samples: list[object]) -> None:
        self.samples = samples
        self.shards: list[tuple[int, int]] = []

    def iter_shard(self, num_shards: int, shard_id: int) -> Iterator[object]:
        self.shards.append((num_shards, shard_id))
        yield from self.samples


class ScheduledDataLoaderTest(unittest.TestCase):
    def test_datamodule_keeps_validation_loader_separate_from_training(self) -> None:
        train_config = SpeechConfig(
            codec="longcat",
            dataloader=DataLoaderConfig(batch_size=2, num_workers=0),
            dataset=DatasetConfig(split_label="train"),
        )
        validation_config = SpeechConfig(
            codec="longcat",
            dataloader=DataLoaderConfig(batch_size=2, num_workers=0),
            dataset=DatasetConfig(split_label="dev"),
        )
        train_spec = LoaderSpec.speech(train_config, {Task.TTS: 1.0})
        validation_spec = LoaderSpec.speech(
            validation_config,
            {Task.TTS: 1.0},
        )
        train_loader = Mock()
        validation_loader = Mock()
        train_batches = object()
        validation_batches = object()
        train_loader.train_dataloader.return_value = train_batches
        validation_loader.validation_dataloader.return_value = validation_batches
        runtime = Mock()

        with (
            patch(
                "speech_to_speech.datamodule.module._build_loader",
                return_value=train_loader,
            ) as build_train,
            patch(
                "speech_to_speech.datamodule.module._build_validation_loader",
                return_value=validation_loader,
            ) as build_validation,
        ):
            datamodule = DataModule(
                runtime,
                {"tts": train_spec},
                validation=validation_spec,
            )
            datamodule.setup("fit")
            training = datamodule.train_dataloader()
            validation = datamodule.val_dataloader()

        self.assertIs(training, train_batches)
        self.assertIs(validation, validation_batches)
        self.assertIs(datamodule.loader_specs["tts"], train_spec)
        self.assertIs(datamodule.validation_spec, validation_spec)
        self.assertIsNot(train_spec.speech_config, validation_spec.speech_config)
        build_train.assert_called_once_with(train_spec, runtime)
        build_validation.assert_called_once_with(validation_spec, runtime)
        train_loader.setup.assert_called_once_with("fit")
        validation_loader.setup.assert_called_once_with("fit")
        train_loader.train_dataloader.assert_called_once_with()
        validation_loader.validation_dataloader.assert_called_once_with()
        validation_loader.train_dataloader.assert_not_called()

    def test_diagnostic_panels_select_train_or_validation_data_and_one_task(self) -> None:
        train_spec = LoaderSpec.speech(
            SpeechConfig(
                codec="longcat",
                dataloader=DataLoaderConfig(batch_size=2, num_workers=0),
                dataset=DatasetConfig(split_label="train"),
            ),
            {Task.TTS: 0.5, Task.T2ST: 0.5},
        )
        validation_spec = LoaderSpec.speech(
            SpeechConfig(
                codec="longcat",
                dataloader=DataLoaderConfig(batch_size=2, num_workers=0),
                dataset=DatasetConfig(split_label="dev"),
            ),
            {Task.TTS: 0.5, Task.T2ST: 0.5},
        )
        train_samples = [object(), object()]
        validation_samples = [object(), object(), object()]
        runtime = Mock(codec_name="longcat")

        with patch(
            "speech_to_speech.datamodule.module.load_dataset",
            side_effect=[train_samples, validation_samples],
        ):
            datamodule = DataModule(
                runtime,
                {"speech": train_spec},
                validation=validation_spec,
            )
            datamodule.setup("fit")
            train = datamodule.diagnostic_samples(
                [1], split=SampleSplit.TRAIN, loader_name="speech"
            )
            validation = datamodule.diagnostic_samples(
                [2], split=SampleSplit.VALIDATION, loader_name="speech"
            )
            collator = datamodule.diagnostic_collator(
                Task.T2ST, split=SampleSplit.VALIDATION, loader_name="speech"
            )

        self.assertEqual(train, [train_samples[1]])
        self.assertEqual(validation, [validation_samples[2]])
        self.assertEqual(collator.tasks, [Task.T2ST])

    def test_text_diagnostic_panel_reads_global_iterable_indices(self) -> None:
        spec = LoaderSpec.text(
            TextConfig(dataloader=DataLoaderConfig(batch_size=2, num_workers=0)),
            {Task.MT: 1.0},
        )
        samples = [object(), object(), object()]
        dataset = _IterableSamples(samples)

        with patch(
            "speech_to_speech.datamodule._text.load_text_dataset",
            return_value=dataset,
        ):
            datamodule = DataModule(Mock(), {"mt": spec})
            datamodule.setup("fit")
            selected = datamodule.diagnostic_samples(
                [2, 0],
                split=SampleSplit.TRAIN,
                loader_name="mt",
            )
            collator = datamodule.diagnostic_collator(
                Task.MT,
                split=SampleSplit.TRAIN,
                loader_name="mt",
            )

        self.assertEqual(selected, [samples[2], samples[0]])
        self.assertEqual(dataset.shards, [(1, 0)])
        self.assertEqual(collator.tasks, [Task.MT])
        with self.assertRaisesRegex(ValueError, "validation.*speech loader"):
            datamodule.diagnostic_samples(
                [0],
                split=SampleSplit.VALIDATION,
                loader_name="mt",
            )

    def test_datamodule_without_validation_does_not_build_a_val_loader(self) -> None:
        spec = LoaderSpec.speech(
            SpeechConfig(
                codec="longcat",
                dataloader=DataLoaderConfig(batch_size=2, num_workers=0),
            ),
            {Task.TTS: 1.0},
        )

        with (
            patch(
                "speech_to_speech.datamodule.module._build_loader",
                return_value=Mock(),
            ),
            patch(
                "speech_to_speech.datamodule.module._build_validation_loader",
            ) as build_validation,
        ):
            datamodule = DataModule(Mock(), {"tts": spec})

        build_validation.assert_not_called()
        self.assertIsNone(datamodule.validation_spec)
        self.assertEqual(tuple(datamodule.val_dataloader()), ())

    def test_datamodule_rejects_text_validation(self) -> None:
        speech = LoaderSpec.speech(
            SpeechConfig(
                codec="longcat",
                dataloader=DataLoaderConfig(batch_size=2, num_workers=0),
            ),
            {Task.TTS: 1.0},
        )
        text = LoaderSpec.text(
            TextConfig(dataloader=DataLoaderConfig(batch_size=2, num_workers=0)),
            {Task.MT: 1.0},
        )

        with (
            patch(
                "speech_to_speech.datamodule.module._build_loader",
                return_value=Mock(),
            ),
            self.assertRaisesRegex(ValueError, "validation requires a speech loader"),
        ):
            DataModule(Mock(), {"tts": speech}, validation=text)

    def test_restart_advances_loader_or_batch_sampler_epoch(self) -> None:
        direct = _DirectEpochLoader(_batch(Task.TTS))
        fallback = _BatchSamplerEpochLoader(_batch(Task.MT))
        loader = ScheduledDataLoader(
            {"speech": direct, "text": fallback},
            LoaderSchedule(
                {"speech": 1.0, "text": 1.0},
                accumulate_grad_batches=2,
            ),
        )

        list(islice(loader, 6))

        self.assertEqual(direct.epochs, [1, 2])
        self.assertEqual(direct.iteration_epochs, [0, 1, 2])
        self.assertEqual(direct.batch_sampler.epochs, [])
        self.assertEqual(fallback.batch_sampler.epochs, [1, 2])
        self.assertEqual(fallback.iteration_epochs, [0, 1, 2])

    def test_epoch_cycles_are_deterministic_across_ranks(self) -> None:
        events = [_rank_events(), _rank_events()]

        self.assertEqual(events[0], events[1])


def _rank_events() -> tuple[list[int], list[int]]:
    speech = _DirectEpochLoader(_batch(Task.TTS), batches=2)
    text = _BatchSamplerEpochLoader(_batch(Task.MT), batches=3)
    loader = ScheduledDataLoader(
        {"speech": speech, "text": text},
        LoaderSchedule({"speech": 2.0, "text": 1.0}),
    )

    list(islice(loader, 18))

    return speech.iteration_epochs, text.iteration_epochs


def _batch(task: Task) -> ModelBatch:
    return ModelBatch(
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        token_labels=torch.tensor([[-100, 2]], dtype=torch.long),
        acoustic_target=None,
        tasks=[task],
        pad_token_id=0,
    )


if __name__ == "__main__":
    unittest.main()
