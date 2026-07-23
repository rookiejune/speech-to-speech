from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sized
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

from ..runtime.types import AudioTokenizer


@dataclass(frozen=True)
class LBAConfig:
    enabled: bool = True
    max_batch_seconds: float = 8.0
    max_padding_ratio: float = 0.05
    prefetch_batches: int = 4
    planner_mode: str = "quality"
    drop_last_flush: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("lba.enabled must be a boolean.")
        _positive_number(self.max_batch_seconds, name="max_batch_seconds")
        if isinstance(self.max_padding_ratio, bool) or not isinstance(
            self.max_padding_ratio, (int, float)
        ):
            raise TypeError("max_padding_ratio must be a number.")
        if not math.isfinite(self.max_padding_ratio) or not (
            0 <= self.max_padding_ratio <= 1
        ):
            raise ValueError("max_padding_ratio must be between 0 and 1.")
        if isinstance(self.prefetch_batches, bool) or not isinstance(
            self.prefetch_batches, int
        ):
            raise TypeError("prefetch_batches must be an integer.")
        if self.prefetch_batches < 0:
            raise ValueError("prefetch_batches must be non-negative.")
        if self.planner_mode not in {"quality", "throughput"}:
            raise ValueError("planner_mode must be 'quality' or 'throughput'.")
        if not isinstance(self.drop_last_flush, bool):
            raise TypeError("drop_last_flush must be a boolean.")


@dataclass(frozen=True)
class DataConfig:
    root: Optional[str] = None
    split: str = "train"
    sample_index: int = 0
    max_seconds: Optional[float] = None
    overlong: str = "error"
    sample_limit: Optional[int] = None
    batch_size: int = 8
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    lba: LBAConfig = field(default_factory=LBAConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.split, str):
            raise TypeError("split must be a string.")
        if not self.split:
            raise ValueError("split must not be empty.")
        if isinstance(self.sample_index, bool) or not isinstance(
            self.sample_index, int
        ):
            raise TypeError("sample_index must be an integer.")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative.")
        if self.max_seconds is not None:
            _positive_number(self.max_seconds, name="max_seconds")
        if self.overlong not in {"error", "filter", "truncate"}:
            raise ValueError("overlong must be 'error', 'filter', or 'truncate'.")
        if (
            self.max_seconds is not None
            and self.lba.enabled
            and self.max_seconds > self.lba.max_batch_seconds
        ):
            raise ValueError(
                "max_seconds must not exceed lba.max_batch_seconds when LBA is enabled."
            )
        for name, value, minimum in (
            ("batch_size", self.batch_size, 1),
            ("num_workers", self.num_workers, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < minimum:
                qualifier = "positive" if minimum == 1 else "non-negative"
                raise ValueError(f"{name} must be {qualifier}.")
        if self.sample_limit is not None:
            if isinstance(self.sample_limit, bool) or not isinstance(
                self.sample_limit, int
            ):
                raise TypeError("sample_limit must be an integer or None.")
            if self.sample_limit <= 0:
                raise ValueError("sample_limit must be positive.")
        for name, value in (
            ("pin_memory", self.pin_memory),
            ("persistent_workers", self.persistent_workers),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean.")


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: DataConfig,
        codec: str,
        *,
        audio_tokenizer: AudioTokenizer | None = None,
        frame_rate: float,
        output_dir: Path,
    ) -> None:
        super().__init__()
        self.data = data
        self.codec = codec
        self.audio_tokenizer = audio_tokenizer
        self.frame_rate = frame_rate
        self.output_dir = output_dir
        self.dataset: Any | None = None
        self.filtered_samples = 0

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
        if self.data.overlong == "filter":
            dataset, self.filtered_samples = _filter(
                dataset,
                codec=self.codec,
                data=self.data,
                frame_rate=self.frame_rate,
            )
            if self.filtered_samples:
                max_seconds = _max_seconds(self.data)
                if max_seconds is None:
                    raise RuntimeError("duration filtering requires a hard limit.")
                warnings.warn(
                    f"filtered {self.filtered_samples} codec oracle samples longer "
                    f"than {max_seconds:g} seconds.",
                    stacklevel=2,
                )
        self.dataset = dataset

    def train_dataloader(self):
        if self.dataset is None:
            raise RuntimeError("codec oracle DataModule.setup() must run first.")
        collate_fn = partial(
            collate,
            codec=self.codec,
            data=self.data,
            audio_tokenizer=self.audio_tokenizer,
            frame_rate=self.frame_rate,
        )
        persistent_workers = self.data.persistent_workers and self.data.num_workers > 0
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
            max_padded_length=_frames(
                self.data.lba.max_batch_seconds,
                self.frame_rate,
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
    codes = _prepared_codes(sample, codec=codec)
    max_seconds = _max_seconds(data)
    if max_seconds is not None:
        max_frames = _frames(max_seconds, frame_rate)
        if codes.size(0) > max_frames:
            if data.overlong == "truncate":
                codes = codes[:max_frames]
            else:
                raise ValueError(
                    f"prepared codec sequence has {codes.size(0)} frames, exceeding "
                    f"the {max_frames}-frame ({max_seconds:g}s) hard limit; "
                    f"overlong policy is {data.overlong!r}."
                )
    codes = codes.long().contiguous()
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
    audio_tokenizer: AudioTokenizer | None = None,
) -> dict[str, Tensor]:
    values = [
        training_item(
            codes(sample, codec=codec, data=data, frame_rate=frame_rate),
            audio_tokenizer=audio_tokenizer,
        )
        for sample in samples
    ]
    return {
        "codes": pad_sequence(
            [value["codes"] for value in values],
            batch_first=True,
            padding_value=-1,
        ),
        "mask": pad_sequence(
            [value["mask"] for value in values],
            batch_first=True,
            padding_value=False,
        ),
        "semantic_tokens": pad_sequence(
            [value["semantic_tokens"] for value in values],
            batch_first=True,
            padding_value=0,
        ),
        "semantic_token_spans": pad_sequence(
            [value["semantic_token_spans"] for value in values],
            batch_first=True,
            padding_value=0,
        ),
    }


def single_batch_loader(
    codes: Tensor,
    audio_tokenizer: AudioTokenizer | None = None,
) -> DataLoader[dict[str, Tensor]]:
    sample = training_item(codes, audio_tokenizer=audio_tokenizer)
    dataset = cast(Dataset[dict[str, Tensor]], cast(object, [sample]))
    return DataLoader(dataset, batch_size=1, num_workers=0)


def training_item(
    codes: Tensor,
    *,
    audio_tokenizer: AudioTokenizer | None = None,
) -> dict[str, Tensor]:
    tokens, spans = _audio_tokens(codes, audio_tokenizer)
    return {
        "codes": codes,
        "mask": torch.ones(codes.size(0), dtype=torch.bool),
        "semantic_tokens": tokens,
        "semantic_token_spans": spans,
    }


def _path(value: Optional[str]) -> Path | None:
    return None if value is None else Path(value).expanduser()


def _filter(
    dataset: Dataset[Any],
    *,
    codec: str,
    data: DataConfig,
    frame_rate: float,
) -> tuple[Dataset[Any], int]:
    max_seconds = _max_seconds(data)
    if max_seconds is None:
        return dataset, 0
    max_frames = _frames(max_seconds, frame_rate)
    size = len(cast(Sized, dataset))
    indices = [
        index
        for index in range(size)
        if _prepared_codes(dataset[index], codec=codec).size(0) <= max_frames
    ]
    dropped = size - len(indices)
    if not indices:
        raise ValueError("codec oracle duration filter removed every sample.")
    return Subset(dataset, indices), dropped


def _max_seconds(data: DataConfig) -> float | None:
    if not data.lba.enabled:
        return data.max_seconds
    if data.max_seconds is None:
        return data.lba.max_batch_seconds
    return min(data.max_seconds, data.lba.max_batch_seconds)


def _prepared_codes(sample: Mapping[Any, Any], *, codec: str) -> Tensor:
    item = sample[(Role.TARGET, Modality.AUDIO)]
    if not isinstance(item, AudioItem):
        raise TypeError("WMT19 target audio must be an AudioItem.")
    value = item.views[AudioView(codec)]
    if not isinstance(value, Tensor) or value.dim() != 2:
        raise ValueError("prepared codec codes must have shape [frame, codebook].")
    return value


def _audio_tokens(
    codes: Tensor,
    audio_tokenizer: AudioTokenizer | None,
) -> tuple[Tensor, Tensor]:
    if codes.dim() != 2 or codes.size(-1) < 1:
        raise ValueError("codec oracle codes must have shape [frames, codebooks].")
    if audio_tokenizer is None:
        tokens = codes[:, 0].to(dtype=torch.long).contiguous()
        spans = torch.ones(tokens.size(0), dtype=torch.long)
        return tokens, spans

    semantic_codes = codes[:, :1].to(dtype=torch.long).contiguous()
    tokens = _as_tensor(audio_tokenizer.encode(semantic_codes)).to(dtype=torch.long)
    spans = _as_tensor(audio_tokenizer.frame_spans(tokens)).to(dtype=torch.long)
    if tokens.dim() != 1:
        raise ValueError("audio tokenizer must encode oracle semantic codes to [tokens].")
    if spans.shape != tokens.shape:
        raise ValueError("audio tokenizer spans must align with encoded tokens.")
    if tokens.numel() == 0:
        raise ValueError("audio tokenizer encoded an empty oracle sequence.")
    if bool((spans <= 0).any()):
        raise ValueError("audio tokenizer spans must be positive.")
    if int(spans.sum()) != codes.size(0):
        raise ValueError("audio tokenizer spans must cover every oracle frame.")
    return tokens.contiguous(), spans.contiguous()


def _as_tensor(value: Tensor | list[int]) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.tensor(value, dtype=torch.long)


def _frames(seconds: float, frame_rate: float) -> int:
    _positive_number(frame_rate, name="frame_rate")
    frames = round(seconds * frame_rate)
    if frames < 1:
        raise ValueError("duration limit must contain at least one codec frame.")
    return frames


def _positive_number(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number.")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
