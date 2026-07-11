from typing import Any, cast

from anydataset.types import Sample
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from ...datamodule import DataModule


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
        outputs = cast(Any, pl_module).generate_batch(causal_batch)
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return
        experiment = logger.experiment
        waveforms = None
        if all(
            task.value in {"audio_ar", "s2st", "t2st", "tts"}
            for task in causal_batch.tasks
        ):
            waveforms = cast(Any, pl_module).generate_waveforms(causal_batch)
        for index, output in enumerate(outputs):
            if waveforms is not None and hasattr(experiment, "add_audio"):
                experiment.add_audio(
                    f"sample/{index}",
                    waveforms[index].detach().cpu(),
                    trainer.global_step,
                    sample_rate=self.sample_rate,
                )
            elif hasattr(experiment, "add_text"):
                experiment.add_text(
                    f"sample/{index}",
                    " ".join(str(value) for value in output.tolist()),
                    trainer.global_step,
                )
