from typing import Any, Protocol, cast

import torch

from anydataset.types import Sample
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from ...datamodule import DataModule
from ...generation import Request, Result
from ...generation.batch import requests_from_batch
from .._lightning import attached_datamodule, audio_experiment, text_experiment


class _Module(Protocol):
    def generate(self, requests: list[Request]) -> list[Result]: ...


class SampleLogger(Callback):
    def __init__(
        self,
        indices: list[int],
        every_n_steps: int,
    ) -> None:
        super().__init__()
        if every_n_steps < 1:
            raise ValueError("every_n_steps must be positive.")
        self.indices = indices
        self.every_n_steps = every_n_steps
        self.samples: list[Sample] = []

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if not trainer.is_global_zero:
            return
        datamodule = cast(DataModule, attached_datamodule(trainer))
        self.samples = datamodule.train_samples(self.indices)

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        del batch, batch_idx
        if not trainer.is_global_zero:
            return
        if trainer.global_step % self.every_n_steps != 0:
            return
        audio_writer = audio_experiment(trainer)
        text_writer = text_experiment(trainer)
        if audio_writer is None and text_writer is None:
            return
        module = cast(_Module, cast(object, pl_module))
        datamodule = cast(DataModule, attached_datamodule(trainer))
        sample_batch = datamodule.collator(self.samples)
        cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=cuda_devices):
            results = module.generate(requests_from_batch(sample_batch))
        for index, result in enumerate(results):
            audio = result["audio"]
            if audio is not None and audio_writer is not None:
                audio_writer.add_audio(
                    f"sample/{index}",
                    audio["waveform"].detach().cpu(),
                    trainer.global_step,
                    sample_rate=audio["sample_rate"],
                )
            elif text_writer is not None:
                text_writer.add_text(
                    f"sample/{index}",
                    " ".join(str(value) for value in result["response_ids"].tolist()),
                    trainer.global_step,
                )
