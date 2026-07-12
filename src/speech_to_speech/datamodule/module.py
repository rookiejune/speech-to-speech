from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from typing import TypedDict

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .collator import Collator
from .types import ModelBatch, Task


class DataLoaderConfig(TypedDict):
    batch_size: int
    num_workers: int


@dataclass
class Config:
    codec: str
    dataloader: DataLoaderConfig

    def train_dataset(self):
        from zhuyin.datasets.wmt19_tts import wmt19_tts_codec

        return wmt19_tts_codec(codec=self.codec)


class DataModule(LightningDataModule):
    def __init__(self, config: Config, strategy: Mapping[Task, float]) -> None:
        super().__init__()

        self.config = config
        self.collator = Collator(strategy)
        self._train_dataset = None

    def setup(self, stage: str) -> None:
        self._train_dataset = self.config.train_dataset()

    def set_strategy(self, strategy: Mapping[Task, float]) -> None:
        self.collator.set_strategy(strategy)

    def train_dataloader(self) -> Iterable[ModelBatch]:
        if self._train_dataset is None:
            raise RuntimeError("DataModule.setup() must run before train_dataloader().")
        return DataLoader(
            self._train_dataset,
            **self.config.dataloader,
            persistent_workers=False,
            collate_fn=self.collator,
        )
