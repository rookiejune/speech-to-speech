"""Load speech-to-speech training datasets.

This module owns the project dataset source selection. The repository default
is the prepared WMT19 TTS LongCat dataset exposed by `workspace`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from anydataset import AnyDataset

from .config import DatasetFactoryConfig

SpeechDataset: TypeAlias = AnyDataset
DatasetMeta: TypeAlias = Mapping[str, object]


def training_dataset(config: DatasetFactoryConfig) -> SpeechDataset:
    if config.name == "wmt19_tts_longcat":
        return _wmt19_tts_longcat()
    raise ValueError(f"unsupported dataset factory: {config.name}")


def dataset_metadata(config: DatasetFactoryConfig) -> tuple[DatasetMeta, ...]:
    return _dataset_object_metadata(training_dataset(config))


def _wmt19_tts_longcat() -> AnyDataset:
    from zhuyin.datasets.wmt19_tts import wmt19_tts_longcat

    dataset = wmt19_tts_longcat()
    if dataset is Ellipsis:
        raise NotImplementedError(
            "zhuyin.datasets.wmt19_tts.wmt19_tts_longcat() is not implemented yet."
        )
    return dataset

def _dataset_object_metadata(dataset: SpeechDataset) -> tuple[DatasetMeta, ...]:
    spec = getattr(dataset, "spec", None)
    to_dict = getattr(spec, "to_dict", None)
    if callable(to_dict):
        return (to_dict(),)

    children = getattr(dataset, "datasets", None)
    if children is not None:
        return tuple(
            meta
            for child in children
            for meta in _dataset_object_metadata(child)
        )

    raise TypeError("default speech dataset must expose anydataset spec metadata.")
