from dataclasses import dataclass
from typing import Any, Iterable, Mapping, TypedDict

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .collator import Batch, Collator, Task


class DataLoaderConfig(TypedDict):
    batch_size: int
    num_workers: int


@dataclass
class Config:
    dataset: str
    dataloader: DataLoaderConfig
    lba: bool = True

    def train_dataset(self):
        from zhuyin.datasets.wmt19_tts import wmt19_tts_longcat

        return wmt19_tts_longcat()


class DataModule(LightningDataModule):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config

        self._strategy: Mapping[Task, float]

    def setup(self, stage: str) -> None:
        self.train_dataset = self.config.train_dataset()

    def set_strategy(self, strategy: Mapping[Task, float]):
        self._strategy = strategy

    @property
    def collator(self):
        return Collator(self._strategy)

    def train_dataloader(self) -> Iterable[Batch]:
        # 通过外层的callback控制weight_dict
        return DataLoader(
            self.train_dataset,
            **self.config.dataloader,
            persistent_workers=False,  # 每个epoch需要重建
            collate_fn=self.collator,
        )
