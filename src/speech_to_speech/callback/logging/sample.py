from typing import Any, cast

from anydataset.types import Sample
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from ...datamodule import DataModule
from ...pl_module.generation import requests_from_batch


class SampleLogger(Callback):
    def __init__(
        self, indices: list[int], intervals: int, sample_rate: int = 24_000
    ) -> None:
        super().__init__()

        self.indices = indices
        self.intervals = intervals
        self.sample_rate = sample_rate
        self.samples: list[Sample] = []

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        datamodule = cast(DataModule, trainer.datamodule)
        self.samples = [datamodule.train_dataset[index] for index in self.indices]

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        del batch, batch_idx
        if self.intervals <= 0:
            raise ValueError("sample logging interval must be positive.")
        if trainer.global_step % self.intervals != 0:
            return
        datamodule = cast(DataModule, pl_module.datamodule)
        causal_batch = datamodule.collator(self.samples)
        results = cast(Any, pl_module).generate(requests_from_batch(causal_batch))
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return
        experiment = logger.experiment
        for index, result in enumerate(results):
            waveform = result["waveform"]
            if waveform is not None and hasattr(experiment, "add_audio"):
                experiment.add_audio(
                    f"sample/{index}",
                    waveform.detach().cpu(),
                    trainer.global_step,
                    sample_rate=self.sample_rate,
                )
            elif hasattr(experiment, "add_text"):
                experiment.add_text(
                    f"sample/{index}",
                    " ".join(str(value) for value in result["token_ids"].tolist()),
                    trainer.global_step,
                )
