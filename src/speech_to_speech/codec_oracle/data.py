from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import torch
from anydataset.types import AudioItem, AudioView, Modality, Role
from lightning import pytorch as pl
from omegaconf import DictConfig
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from zhuyin.datasets.wmt19_tts import wmt19_tts_codec

from .types import Objective


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: DictConfig,
        codec: DictConfig,
        *,
        output_dir: Path,
        seed: int,
    ) -> None:
        super().__init__()
        self.data = data
        self.codec = codec
        self.output_dir = output_dir
        self.seed = seed
        self.dataset: Any | None = None
        self.sampler: DistributedSampler[Any] | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.dataset is not None:
            return
        dataset = wmt19_tts_codec(
            codec=str(self.codec.name),
            root=_path(self.data.root),
            split=str(self.data.split),
        )
        sample_limit = self.data.sample_limit
        if sample_limit is not None:
            dataset = Subset(dataset, range(min(int(sample_limit), len(dataset))))
        self.dataset = dataset

    def train_dataloader(self):
        if self.dataset is None:
            raise RuntimeError("codec oracle DataModule.setup() must run first.")
        trainer = self.trainer
        if trainer is None:
            raise RuntimeError("codec oracle DataModule must be attached to a Trainer.")
        sampler = None
        if trainer.world_size > 1:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=trainer.world_size,
                rank=trainer.global_rank,
                shuffle=True,
                seed=self.seed,
                drop_last=False,
            )
        self.sampler = sampler
        loader = DataLoader(
            self.dataset,
            batch_size=int(self.data.batch_size),
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=int(self.data.num_workers),
            pin_memory=bool(self.data.pin_memory),
            persistent_workers=(
                bool(self.data.persistent_workers) and int(self.data.num_workers) > 0
            ),
            collate_fn=partial(collate, codec=self.codec, data=self.data),
        )
        if not bool(self.data.lba.enabled):
            return loader
        from lba import LBA

        return LBA(
            loader,
            len_fn=partial(length, codec=self.codec, data=self.data),
            max_padded_length=round(
                float(self.data.lba.max_batch_seconds) * float(self.codec.frame_rate)
            ),
            max_padding_ratio=float(self.data.lba.max_padding_ratio),
            prefetch_batches=int(self.data.lba.prefetch_batches),
            planner_mode=cast(
                Literal["quality", "throughput"],
                str(self.data.lba.planner_mode),
            ),
            drop_last_flush=bool(self.data.lba.drop_last_flush),
            log_dir=self.output_dir / "lba",
        )


def codes(
    sample: Mapping[Any, Any],
    *,
    codec: DictConfig,
    data: DictConfig,
) -> Tensor:
    item = sample[(Role.TARGET, Modality.AUDIO)]
    if not isinstance(item, AudioItem):
        raise TypeError("WMT19 target audio must be an AudioItem.")
    codes = item.views[AudioView(str(codec.view))]
    if not isinstance(codes, Tensor) or codes.dim() != 2:
        raise ValueError("prepared codec codes must have shape [frame, codebook].")
    frames = min(
        codes.size(0),
        round(float(data.max_seconds) * float(codec.frame_rate)),
    )
    codes = codes[:frames].long().contiguous()
    if codes.size(0) == 0:
        raise ValueError("selected prepared codec sequence is empty.")
    return codes


def length(
    sample: Mapping[Any, Any],
    *,
    codec: DictConfig,
    data: DictConfig,
) -> int:
    return codes(sample, codec=codec, data=data).size(0)


def collate(
    samples: list[Mapping[Any, Any]],
    *,
    codec: DictConfig,
    data: DictConfig,
) -> dict[str, Tensor]:
    values = [codes(sample, codec=codec, data=data) for sample in samples]
    padded = pad_sequence(values, batch_first=True, padding_value=-1)
    mask = (padded >= 0).all(dim=-1)
    selected = Objective(str(codec.objective)).select_codes(padded)
    return {"codes": selected, "mask": mask}


def single_batch_loader(
    codes: Tensor,
    *,
    objective: Objective,
) -> DataLoader[dict[str, Tensor]]:
    value = objective.select_codes(codes)
    sample = {
        "codes": value,
        "mask": torch.ones(value.size(0), dtype=torch.bool),
    }
    dataset = cast(Dataset[dict[str, Tensor]], cast(object, [sample]))
    return DataLoader(dataset, batch_size=1, num_workers=0)


def _path(value: Any) -> Path | None:
    return None if value is None else Path(str(value)).expanduser()
