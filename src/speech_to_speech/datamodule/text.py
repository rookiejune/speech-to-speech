from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, cast

from anydataset.types import Lang, Modality, Role, Sample, TextItem, TextMeta, TextView
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .._compat import StrEnum, auto
from ..task import Task
from .collator import TextCollator
from .lba import LBA, LBAConfig, PlannerMode, text_length
from .module import DataLoaderConfig
from .protocol import TextRuntime, TextRuntimeSnapshot
from .types import ModelBatch


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


def load_text_dataset(config: TextDatasetConfig):
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
        batch_size = self.dataloader["batch_size"]
        num_workers = self.dataloader["num_workers"]
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("text dataloader batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("text dataloader batch_size must be positive.")
        if isinstance(num_workers, bool) or not isinstance(num_workers, int):
            raise TypeError("text dataloader num_workers must be an integer.")
        if num_workers < 0:
            raise ValueError("text dataloader num_workers must be non-negative.")
        for name in ("pin_memory", "persistent_workers"):
            value = self.dataloader.get(name, False)
            if not isinstance(value, bool):
                raise TypeError(f"text dataloader {name} must be a boolean.")
        lba = self.dataloader.get("lba")
        if lba is not None and not isinstance(lba, LBAConfig):
            raise TypeError("text dataloader lba must be an LBAConfig.")


class TextDataModule(LightningDataModule):
    def __init__(
        self,
        config: TextConfig,
        runtime: TextRuntime,
        task_weights: Mapping[Task, float],
        *,
        output_dir: Path | None = None,
        loader_name: str = "text",
    ) -> None:
        super().__init__()
        self.config = config
        self.runtime = runtime
        self.collator = TextCollator(runtime, task_weights)
        self.output_dir = output_dir
        self.loader_name = loader_name
        self._train_dataset = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self._train_dataset is not None:
            return
        self._train_dataset = load_text_dataset(self.config.dataset)

    def set_task_weights(self, task_weights: Mapping[Task, float]) -> None:
        self.collator.set_task_weights(task_weights)

    def train_dataloader(self) -> Iterable[ModelBatch]:
        if self._train_dataset is None:
            raise RuntimeError(
                "TextDataModule.setup() must run before train_dataloader()."
            )
        loader = self.config.dataloader
        num_workers = loader["num_workers"]
        if not isinstance(self.collator.runtime, TextRuntimeSnapshot):
            self.collator.runtime = cast(
                TextRuntime,
                cast(object, TextRuntimeSnapshot.from_runtime(self.runtime)),
            )
        lba = loader.get("lba")
        if lba is not None and lba.enabled:
            return LBA(
                self._train_dataset,
                batch_size=loader["batch_size"],
                shuffle=True,
                num_workers=num_workers,
                pin_memory=loader.get("pin_memory", False),
                persistent_workers=(
                    loader.get("persistent_workers", False) and num_workers > 0
                ),
                collate_fn=self.collator,
                len_fn=partial(
                    text_length,
                    runtime=self.collator.runtime,
                    tasks=tuple(self.collator.tasks),
                    config=lba,
                ),
                max_padded_length=lba.max_batch_cost,
                max_padding_ratio=lba.max_padding_ratio,
                prefetch_batches=lba.prefetch_batches,
                planner_mode=cast(PlannerMode, lba.planner_mode),
                drop_last_flush=lba.drop_last_flush,
                log_dir=_lba_log_dir(self.output_dir, self.loader_name),
            )
        return DataLoader(
            self._train_dataset,
            batch_size=loader["batch_size"],
            num_workers=num_workers,
            pin_memory=loader.get("pin_memory", False),
            persistent_workers=(
                loader.get("persistent_workers", False) and num_workers > 0
            ),
            collate_fn=self.collator,
        )


__all__ = [
    "TextConfig",
    "TextDataModule",
    "TextDatasetConfig",
    "TextDatasetName",
    "ToyTextDataset",
    "load_text_dataset",
]


def _lba_log_dir(output_dir: Path | None, loader_name: str) -> Path | None:
    if output_dir is None:
        return None
    return output_dir / "lba" / loader_name
