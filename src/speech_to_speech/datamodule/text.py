from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from anydataset import IterableAnyDataset
from anydataset.types import Lang, Modality, Role, Sample, TextItem, TextMeta, TextView
from torch.utils.data import Dataset

from .._compat import StrEnum, auto
from .config import DataLoaderConfig


class TextDatasetName(StrEnum):
    WMT19 = auto()
    TOY = auto()


@dataclass
class TextDatasetConfig:
    name: TextDatasetName = TextDatasetName.WMT19
    split: str = "train"
    config_name: Optional[str] = None
    source_lang: Optional[str] = "zh"
    target_lang: Optional[str] = "en"
    toy_samples: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.name, TextDatasetName):
            raise TypeError("text dataset name must be a TextDatasetName.")
        if not isinstance(self.split, str):
            raise TypeError("text dataset split must be a string.")
        if not self.split:
            raise ValueError("text dataset split must not be empty.")
        for name in ("config_name", "source_lang", "target_lang"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None.")
            if value == "":
                raise ValueError(f"{name} must not be empty.")
        if isinstance(self.toy_samples, bool) or not isinstance(self.toy_samples, int):
            raise TypeError("toy_samples must be an integer.")
        if self.toy_samples <= 0:
            raise ValueError("toy_samples must be positive.")


class ToyTextDataset(Dataset[Sample]):
    def __init__(self, *, samples: int = 8) -> None:
        if isinstance(samples, bool) or not isinstance(samples, int):
            raise TypeError("toy text samples must be an integer.")
        if samples <= 0:
            raise ValueError("toy text samples must be positive.")
        self.samples = samples

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> Sample:
        if index < 0:
            index += self.samples
        if index < 0 or index >= self.samples:
            raise IndexError(index)
        return {
            (Role.SOURCE, Modality.TEXT): TextItem(
                views={TextView.TEXT: f"toy source {index}"},
                meta={TextMeta.LANG: Lang.ZH},
            ),
            (Role.TARGET, Modality.TEXT): TextItem(
                views={TextView.TEXT: f"toy target {index}"},
                meta={TextMeta.LANG: Lang.EN},
            ),
        }


def load_text_dataset(
    config: TextDatasetConfig,
) -> Dataset[Sample] | IterableAnyDataset:
    if config.name is TextDatasetName.TOY:
        return ToyTextDataset(samples=config.toy_samples)
    if config.name is TextDatasetName.WMT19:
        from anydataset.presets import WMT19

        kwargs = {}
        if config.config_name is not None:
            kwargs["config_name"] = config.config_name
        if config.source_lang is not None:
            kwargs["source_lang"] = config.source_lang
        if config.target_lang is not None:
            kwargs["target_lang"] = config.target_lang
        return WMT19(split=config.split, **kwargs)
    raise AssertionError(f"unsupported text dataset: {config.name}")


@dataclass
class TextConfig:
    dataloader: DataLoaderConfig
    dataset: TextDatasetConfig = field(default_factory=TextDatasetConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.dataloader, DataLoaderConfig):
            raise TypeError("text dataloader must be a DataLoaderConfig.")


__all__ = [
    "TextConfig",
    "TextDatasetConfig",
    "TextDatasetName",
    "ToyTextDataset",
    "load_text_dataset",
]
