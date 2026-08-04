"""Lightning data entry for already-prepared aligned MIMO samples."""

from __future__ import annotations

from functools import partial
from typing import cast

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from ..config import DataLoaderConfig
from .batch import MIMO_IGNORE_INDEX, MimoBatch, MimoSample, collate_mimo


class MimoDataModule(LightningDataModule):
    """Batch prepared MIMO samples without routing through single-stream tasks."""

    def __init__(
        self,
        train_dataset: Dataset[MimoSample],
        *,
        dataloader: DataLoaderConfig,
        text_pad_token_id: int,
        audio_pad_token_id: int,
        validation_dataset: Dataset[MimoSample] | None = None,
        ignore_index: int = MIMO_IGNORE_INDEX,
    ) -> None:
        super().__init__()
        if not isinstance(train_dataset, Dataset):
            raise TypeError("train_dataset must be a torch Dataset of MimoSample.")
        if validation_dataset is not None and not isinstance(validation_dataset, Dataset):
            raise TypeError("validation_dataset must be a torch Dataset or None.")
        if not isinstance(dataloader, DataLoaderConfig):
            raise TypeError("dataloader must be a DataLoaderConfig.")
        if dataloader.costs.enabled:
            raise ValueError("cost-planned batches are not implemented for MIMO data.")
        if (
            dataloader.persistent_workers
            and dataloader.num_workers > 0
            and (
                callable(getattr(train_dataset, "set_epoch", None))
                or callable(getattr(validation_dataset, "set_epoch", None))
            )
        ):
            raise ValueError(
                "persistent_workers is incompatible with epoch-aware MIMO datasets; "
                "set persistent_workers=false or use a stateless dataset."
            )
        for name, value in (
            ("text_pad_token_id", text_pad_token_id),
            ("audio_pad_token_id", audio_pad_token_id),
            ("ignore_index", ignore_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.config = dataloader
        self.text_pad_token_id = text_pad_token_id
        self.audio_pad_token_id = audio_pad_token_id
        self.ignore_index = ignore_index

    def train_dataloader(self) -> DataLoader[MimoBatch]:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader[MimoBatch] | None:
        if self.validation_dataset is None:
            return None
        return self._loader(self.validation_dataset, shuffle=False)

    def _loader(
        self,
        dataset: Dataset[MimoSample],
        *,
        shuffle: bool,
    ) -> DataLoader[MimoBatch]:
        workers = self.config.num_workers
        return cast(
            DataLoader[MimoBatch],
            DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=workers,
                pin_memory=self.config.pin_memory,
                persistent_workers=self.config.persistent_workers and workers > 0,
                collate_fn=partial(
                    collate_mimo,
                    text_pad_token_id=self.text_pad_token_id,
                    audio_pad_token_id=self.audio_pad_token_id,
                    ignore_index=self.ignore_index,
                ),
            ),
        )


__all__ = ["MimoDataModule"]
