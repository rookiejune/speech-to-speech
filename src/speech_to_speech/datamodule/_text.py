from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

from torch.utils.data import DataLoader

from ..task import Task
from .collator import TextCollator
from .protocol import TextRuntime, TextRuntimeSnapshot
from .text import TextConfig, load_text_dataset
from .types import ModelBatch


class TextLoader:
    def __init__(
        self,
        config: TextConfig,
        runtime: TextRuntime,
        task_weights: Mapping[Task, float],
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.collator = TextCollator(runtime, task_weights)
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
                "text loader setup() must run before train_dataloader()."
            )
        loader = self.config.dataloader
        num_workers = loader.num_workers
        if not isinstance(self.collator.runtime, TextRuntimeSnapshot):
            self.collator.runtime = cast(
                TextRuntime,
                cast(object, TextRuntimeSnapshot.from_runtime(self.runtime)),
            )
        return DataLoader(
            self._train_dataset,
            batch_size=loader.batch_size,
            num_workers=num_workers,
            pin_memory=loader.pin_memory,
            persistent_workers=(loader.persistent_workers and num_workers > 0),
            collate_fn=self.collator,
        )


__all__ = ["TextLoader"]
