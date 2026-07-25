from __future__ import annotations

import unittest
from collections.abc import Iterator
from itertools import islice

import torch

from speech_to_speech.datamodule.joint import LoaderSchedule, ScheduledDataLoader
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


class ScheduledDataLoaderTest(unittest.TestCase):
    def test_restart_advances_loader_or_batch_sampler_epoch(self) -> None:
        direct = _DirectEpochLoader(_batch(Task.TTS))
        fallback = _BatchSamplerEpochLoader(_batch(Task.MT))
        loader = ScheduledDataLoader(
            {"speech": direct, "text": fallback},
            LoaderSchedule(
                {"speech": 1.0, "text": 1.0},
                batches_per_step=2,
            ),
        )

        list(islice(loader, 3))

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
        input_ids=torch.tensor([[1]], dtype=torch.long),
        token_labels=torch.tensor([[1]], dtype=torch.long),
        acoustic_target=None,
        tasks=[task],
        pad_token_id=0,
    )


if __name__ == "__main__":
    unittest.main()
