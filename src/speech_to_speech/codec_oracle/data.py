from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional, cast

import torch
from anydataset.types import AudioItem, AudioView, Modality, Role
from lba import LBA
from lightning import pytorch as pl
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from zhuyin.datasets.wmt19_tts import wmt19_tts_codec


@dataclass(frozen=True)
class LBAConfig:
    enabled: bool = True
    max_batch_seconds: float = 8.0
    max_padding_ratio: float = 0.05
    prefetch_batches: int = 0
    planner_mode: str = "quality"
    drop_last_flush: bool = True


@dataclass(frozen=True)
class DataConfig:
    root: Optional[str] = None
    split: str = "train"
    sample_index: int = 0
    max_seconds: float = 4.0
    sample_limit: Optional[int] = None
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    lba: LBAConfig = field(default_factory=LBAConfig)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: DataConfig,
        codec: str,
        *,
        frame_rate: float,
        output_dir: Path,
    ) -> None:
        super().__init__()
        self.data = data
        self.codec = codec
        self.frame_rate = frame_rate
        self.output_dir = output_dir
        self.dataset: Any | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.dataset is not None:
            return
        dataset = wmt19_tts_codec(
            codec=self.codec,
            root=_path(self.data.root),
            split=self.data.split,
        )
        sample_limit = self.data.sample_limit
        if sample_limit is not None:
            dataset = Subset(dataset, range(min(sample_limit, len(dataset))))
        self.dataset = dataset

    def train_dataloader(self):
        if self.dataset is None:
            raise RuntimeError("codec oracle DataModule.setup() must run first.")
        collate_fn = partial(
            collate,
            codec=self.codec,
            data=self.data,
            frame_rate=self.frame_rate,
        )
        persistent_workers = (
            self.data.persistent_workers and self.data.num_workers > 0
        )
        if not self.data.lba.enabled:
            return DataLoader(
                self.dataset,
                batch_size=self.data.batch_size,
                shuffle=True,
                num_workers=self.data.num_workers,
                pin_memory=self.data.pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
            )
        return LBA(
            self.dataset,
            batch_size=self.data.batch_size,
            shuffle=True,
            num_workers=self.data.num_workers,
            pin_memory=self.data.pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
            len_fn=partial(
                length,
                codec=self.codec,
                data=self.data,
                frame_rate=self.frame_rate,
            ),
            max_padded_length=round(
                self.data.lba.max_batch_seconds * self.frame_rate
            ),
            max_padding_ratio=self.data.lba.max_padding_ratio,
            prefetch_batches=self.data.lba.prefetch_batches,
            planner_mode=cast(
                Literal["quality", "throughput"],
                self.data.lba.planner_mode,
            ),
            drop_last_flush=self.data.lba.drop_last_flush,
            log_dir=self.output_dir / "lba",
        )


def codes(
    sample: Mapping[Any, Any],
    *,
    codec: str,
    data: DataConfig,
    frame_rate: float,
) -> Tensor:
    item = sample[(Role.TARGET, Modality.AUDIO)]
    if not isinstance(item, AudioItem):
        raise TypeError("WMT19 target audio must be an AudioItem.")
    codes = item.views[AudioView(codec)]
    if not isinstance(codes, Tensor) or codes.dim() != 2:
        raise ValueError("prepared codec codes must have shape [frame, codebook].")
    frames = min(
        codes.size(0),
        round(data.max_seconds * frame_rate),
    )
    codes = codes[:frames].long().contiguous()
    if codes.size(0) == 0:
        raise ValueError("selected prepared codec sequence is empty.")
    return codes


def length(
    sample: Mapping[Any, Any],
    *,
    codec: str,
    data: DataConfig,
    frame_rate: float,
) -> int:
    return codes(sample, codec=codec, data=data, frame_rate=frame_rate).size(0)


def collate(
    samples: list[Mapping[Any, Any]],
    *,
    codec: str,
    data: DataConfig,
    frame_rate: float,
) -> dict[str, Tensor]:
    values = [
        codes(sample, codec=codec, data=data, frame_rate=frame_rate)
        for sample in samples
    ]
    padded = pad_sequence(values, batch_first=True, padding_value=-1)
    mask = (padded >= 0).all(dim=-1)
    return {"codes": padded, "mask": mask}


def single_batch_loader(
    codes: Tensor,
) -> DataLoader[dict[str, Tensor]]:
    sample = {
        "codes": codes,
        "mask": torch.ones(codes.size(0), dtype=torch.bool),
    }
    dataset = cast(Dataset[dict[str, Tensor]], cast(object, [sample]))
    return DataLoader(dataset, batch_size=1, num_workers=0)


def _path(value: Optional[str]) -> Path | None:
    return None if value is None else Path(value).expanduser()
