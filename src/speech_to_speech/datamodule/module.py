from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from anytrain.idspace import IdSpaceEmbedding
from lightning.pytorch import LightningDataModule
from torch import device as TorchDevice
from torch.utils.data import DataLoader

from ..config import BPEConfig, DataModuleConfig, TaskConfig
from ..runtime import longcat_tokenizer
from ..types import CausalLMBatch
from .batch_builder import CausalLMBatchBuilder
from .pipeline import (
    TaskBatchMapper,
    TaskSampleCollator,
    TaskSampleStream,
    TaskSample,
    task_sample_length,
)


class SpeechToSpeechDataModule(LightningDataModule):
    """Builds unified causal LM batches from expanded anydataset task samples."""

    def __init__(
        self,
        datamodule: DataModuleConfig,
        tasks: TaskConfig,
        embedding: IdSpaceEmbedding,
        *,
        tokenizer: object | None = None,
        bpe_tokenizer: object | None = None,
        bpe: BPEConfig | None = None,
    ) -> None:
        super().__init__()
        self.datamodule = datamodule
        self.tasks = tasks
        self.builder = CausalLMBatchBuilder(embedding, tokenizer=tokenizer)
        self.bpe_tokenizer = bpe_tokenizer
        self.bpe = bpe or BPEConfig()

    def setup(self, stage: str | None = None) -> None:
        return

    def train_dataloader(self) -> Iterable[CausalLMBatch]:
        dataloader = self.datamodule.dataloader
        loader = DataLoader(
            TaskSampleStream(
                self.datamodule.dataset_factory,
                tasks=self.tasks,
            ),
            batch_size=dataloader.batch_size,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            collate_fn=TaskSampleCollator(),
        )
        source: Iterable[Sequence[TaskSample]]
        if self.datamodule.lba.enabled:
            from lba import LBA

            source = LBA(
                loader,
                len_fn=task_sample_length,
                drop_last_flush=dataloader.drop_last,
                log_dir=_lba_log_dir(self.datamodule),
            )
        else:
            source = loader

        return TaskBatchMapper(
            source,
            builder=self.builder,
            bpe_tokenizer=self._bpe_tokenizer(),
            device=self._device(),
        )

    def _bpe_tokenizer(self) -> object:
        if self.bpe_tokenizer is None:
            self.bpe_tokenizer = longcat_tokenizer(self.bpe)
        return self.bpe_tokenizer

    def _device(self) -> TorchDevice:
        return TorchDevice("cpu")


def _lba_log_dir(datamodule: DataModuleConfig) -> Path:
    if datamodule.lba.log_dir is not None:
        return Path(datamodule.lba.log_dir)
    return Path(".lba") / "logs"
