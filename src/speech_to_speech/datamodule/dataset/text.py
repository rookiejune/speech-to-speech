from __future__ import annotations

import codecs
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Optional, Union, cast

from anydataset import IterableAnyDataset
from anydataset.types import Lang, Modality, Role, Sample, TextItem, TextMeta, TextView
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset

from ..._compat import StrEnum, auto
from ...task import PredictionModality, Task
from ..batch import ModelBatch
from ..collate import TextCollator
from ..config import DataLoaderConfig
from ..config import TaskConfig
from ..loader.contract import ARFraming
from ..contract import TextRuntime, TextRuntimeSnapshot


class TextDatasetName(StrEnum):
    WMT19 = auto()
    TOY = auto()
    GENERAL = auto()
    JSONL = auto()
    DOCUMENT = auto()

    @classmethod
    def _missing_(cls, value: object) -> TextDatasetName | None:
        if not isinstance(value, str):
            return None
        aliases = {
            "jsonlines": cls.JSONL,
            "jsonl": cls.JSONL,
            "text": cls.DOCUMENT,
            "txt": cls.DOCUMENT,
            "plain": cls.DOCUMENT,
        }
        return aliases.get(value.lower().lstrip("."))


@dataclass
class TextDatasetConfig:
    name: TextDatasetName = TextDatasetName.WMT19
    split: str = "train"
    config_name: Optional[str] = None
    source_lang: Optional[str] = "zh"
    target_lang: Optional[str] = "en"
    toy_samples: int = 8
    # ``path`` is used by the local GENERAL/JSONL/DOCUMENT readers.  Keeping
    # it optional preserves the existing WMT19 and TOY contracts.
    path: Optional[Union[str, Path]] = None
    format: Optional[str] = None
    encoding: str = "utf-8"
    language: str = "en"

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
        if self.path is not None:
            if not isinstance(self.path, (str, Path)):
                raise TypeError("text dataset path must be a string, Path, or None.")
            if isinstance(self.path, str) and not self.path.strip():
                raise ValueError("text dataset path must not be empty.")
            if isinstance(self.path, Path) and not str(self.path):
                raise ValueError("text dataset path must not be empty.")
        if self.format is not None:
            if not isinstance(self.format, str):
                raise TypeError("text dataset format must be a string or None.")
            if not self.format.strip():
                raise ValueError("text dataset format must not be empty.")
            normalized_format = self.format.lower().lstrip(".")
            if normalized_format not in {"jsonl", "jsonlines", "document", "text", "txt"}:
                raise ValueError(
                    "text dataset format must be jsonl or document/text."
                )
        if not isinstance(self.encoding, str):
            raise TypeError("text dataset encoding must be a string.")
        if not self.encoding.strip():
            raise ValueError("text dataset encoding must not be empty.")
        try:
            codecs.lookup(self.encoding)
        except LookupError as error:
            raise ValueError(f"unknown text dataset encoding: {self.encoding!r}.") from error
        if not isinstance(self.language, str):
            raise TypeError("text dataset language must be a string.")
        if not self.language.strip():
            raise ValueError("text dataset language must not be empty.")
        _lang(self.language)
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


class JsonlTextDataset(Dataset[Sample]):
    """Strict local JSONL text reader.

    Each non-empty line must be either a JSON string or an object containing a
    string ``text`` field and an optional string ``lang`` field.  Samples are
    exposed as text pairs so they can flow through the existing text parser;
    TEXT_AR pretraining consumes the target side and therefore does not need a
    parallel source corpus.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        encoding: str = "utf-8",
        language: str = "en",
    ) -> None:
        self.path = _file_path(path, name="text dataset path")
        self.encoding = _encoding(encoding)
        self.language = _lang(language)
        self._offsets = _jsonl_offsets(self.path, self.encoding, self.language)
        if not self._offsets:
            raise ValueError(f"text dataset JSONL file is empty: {self.path}.")

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> Sample:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("text dataset index must be an integer.")
        if index < 0:
            index += len(self._offsets)
        if index < 0 or index >= len(self._offsets):
            raise IndexError(index)
        try:
            with self.path.open(
                "r",
                encoding=self.encoding,
                errors="strict",
                newline="",
            ) as handle:
                handle.seek(self._offsets[index])
                line = handle.readline()
        except UnicodeError as error:
            raise ValueError(
                f"text JSONL is not valid {self.encoding!r}: {self.path}."
            ) from error
        text, lang = _parse_jsonl_line(line, index + 1, self.language)
        return _sample(text, lang)


class DocumentTextDataset(Dataset[Sample]):
    """Strict reader for one plain-text document per file.

    The whole file is represented as one training document.  Use JSONL when a
    corpus contains many documents or per-record language metadata.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        encoding: str = "utf-8",
        language: str = "en",
    ) -> None:
        self.path = _file_path(path, name="text document path")
        self.encoding = _encoding(encoding)
        lang = _lang(language)
        try:
            text = self.path.read_text(encoding=self.encoding, errors="strict")
        except UnicodeError as error:
            raise ValueError(
                f"text document is not valid {self.encoding!r}: {self.path}."
            ) from error
        if not text.strip():
            raise ValueError(f"text document must not be empty: {self.path}.")
        self._sample = _sample(text, lang)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> Sample:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("text dataset index must be an integer.")
        if index not in {0, -1}:
            raise IndexError(index)
        return self._sample


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
    if config.name in {
        TextDatasetName.GENERAL,
        TextDatasetName.JSONL,
        TextDatasetName.DOCUMENT,
    }:
        if config.path is None:
            raise ValueError(
                f"{config.name.value} text datasets require dataset.path."
            )
        kind = _dataset_format(config)
        if kind == "jsonl":
            return JsonlTextDataset(
                config.path,
                encoding=config.encoding,
                language=config.language,
            )
        return DocumentTextDataset(
            config.path,
            encoding=config.encoding,
            language=config.language,
        )
    raise AssertionError(f"unsupported text dataset: {config.name}")


@dataclass
class TextConfig:
    dataloader: DataLoaderConfig
    dataset: TextDatasetConfig = field(default_factory=TextDatasetConfig)
    # Optional token packing for instruction-free TEXT_AR batches.  These
    # fields live at the datamodule level because packing is a batching policy,
    # not a property of the source file.
    max_tokens: Optional[int] = None
    pack_documents: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.dataloader, DataLoaderConfig):
            raise TypeError("text dataloader must be a DataLoaderConfig.")
        if self.dataloader.costs.enabled:
            raise ValueError("dataloader costs are unsupported for text loaders.")
        if self.max_tokens is not None:
            if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
                raise TypeError("text max_tokens must be an integer or None.")
            if self.max_tokens < 2:
                raise ValueError("text max_tokens must be at least 2.")
        if not isinstance(self.pack_documents, bool):
            raise TypeError("text pack_documents must be a boolean.")
        # ``max_tokens`` is also accepted as the shorthand for enabling pack.
        if self.pack_documents and self.max_tokens is None:
            raise ValueError("text pack_documents requires max_tokens.")


def _dataset_format(config: TextDatasetConfig) -> str:
    if config.name is TextDatasetName.JSONL:
        return "jsonl"
    if config.name is TextDatasetName.DOCUMENT:
        return "document"
    if config.format is not None:
        value = config.format.lower().lstrip(".")
        return "jsonl" if value in {"jsonl", "jsonlines"} else "document"
    if config.path is None:
        raise ValueError("general text datasets require dataset.path.")
    path = Path(config.path)
    return "jsonl" if path.suffix.lower() in {".jsonl", ".jsonlines"} else "document"


def _file_path(value: str | Path, *, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{name} must be a string or Path.")
    path = Path(value).expanduser()
    if not str(path):
        raise ValueError(f"{name} must not be empty.")
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}.")
    if not path.is_file():
        raise ValueError(f"{name} must point to a regular file: {path}.")
    return path


def _encoding(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("text dataset encoding must be a string.")
    if not value.strip():
        raise ValueError("text dataset encoding must not be empty.")
    try:
        codecs.lookup(value)
    except LookupError as error:
        raise ValueError(f"unknown text dataset encoding: {value!r}.") from error
    return value


def _lang(value: str) -> Lang:
    if not isinstance(value, str):
        raise TypeError("text dataset language must be a string.")
    normalized = value.strip().lower().replace("_", "-")
    aliases = {"english": "en", "chinese": "zh", "zh-cn": "zh", "en-us": "en"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"en", "zh"}:
        raise ValueError(
            f"unsupported text dataset language {value!r}; "
            "the current text runtime supports 'en' and 'zh'."
        )
    return Lang(normalized)


def _jsonl_offsets(path: Path, encoding: str, default_lang: Lang) -> tuple[int, ...]:
    offsets: list[int] = []
    try:
        with path.open("r", encoding=encoding, errors="strict", newline="") as handle:
            line_number = 1
            while True:
                offset = handle.tell()
                line = handle.readline()
                if line == "":
                    break
                _parse_jsonl_line(line, line_number, default_lang)
                offsets.append(offset)
                line_number += 1
    except UnicodeError as error:
        raise ValueError(f"text JSONL is not valid {encoding!r}: {path}.") from error
    return tuple(offsets)


def _parse_jsonl_line(line: str, line_number: int, default_lang: Lang) -> tuple[str, Lang]:
    if not line.strip():
        raise ValueError(f"text JSONL line {line_number} is blank.")
    try:
        value: Any = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON on text JSONL line {line_number}.") from error
    lang = default_lang
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        unknown = set(value) - {"text", "lang"}
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(
                f"text JSONL line {line_number} has unknown fields: {names}."
            )
        text = value.get("text")
        if "lang" in value:
            raw_lang = value["lang"]
            if not isinstance(raw_lang, str):
                raise ValueError(
                    f"text JSONL line {line_number} field 'lang' must be a string."
                )
            try:
                lang = _lang(raw_lang)
            except ValueError as error:
                raise ValueError(
                    f"text JSONL line {line_number} has invalid field 'lang'."
                ) from error
    else:
        raise ValueError(
            f"text JSONL line {line_number} must be a string or object with text."
        )
    if not isinstance(text, str):
        raise ValueError(f"text JSONL line {line_number} field 'text' must be a string.")
    if not text.strip():
        raise ValueError(f"text JSONL line {line_number} field 'text' must not be empty.")
    return text, lang


def _sample(text: str, lang: Lang) -> Sample:
    def item() -> TextItem:
        return TextItem(
            views={TextView.TEXT: text},
            meta={TextMeta.LANG: lang},
        )

    return {
        (Role.SOURCE, Modality.TEXT): item(),
        (Role.TARGET, Modality.TEXT): item(),
    }


# Readable acronym aliases for callers that prefer the conventional JSONL
# spelling.  The lowercase class remains the canonical implementation name.
JSONLTextDataset = JsonlTextDataset
PlainTextDataset = DocumentTextDataset


class TextLoader:
    def __init__(
        self,
        config: TextConfig,
        runtime: TextRuntime,
        task_weights: Mapping[Task, float],
        *,
        prediction: PredictionModality | None = None,
        trace: str | None = None,
        ar_framing: ARFraming = ARFraming.INSTRUCTION,
        max_samples: int | None = None,
        tasks: Mapping[Task, TaskConfig] | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.task_configs = tasks
        self.collator = TextCollator(
            runtime,
            task_weights,
            prediction=prediction,
            trace=trace,
            ar_framing=ar_framing,
            tasks=tasks,
            max_tokens=config.max_tokens,
            pack_documents=config.pack_documents,
        )
        self.max_samples = max_samples
        self.num_workers = config.dataloader.num_workers
        self._train_dataset: Dataset[Sample] | IterableAnyDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self._train_dataset is not None:
            return
        self._train_dataset = load_text_dataset(self.config.dataset)

    def train_samples(self, indices: Sequence[int]) -> list[Sample]:
        if self._train_dataset is None:
            raise RuntimeError("TextLoader.setup() must run before reading samples.")
        return _samples(self._train_dataset, indices)

    def diagnostic_collator(self, task: Task) -> TextCollator:
        return TextCollator(
            self.runtime,
            {task: 1.0},
            prediction=self.collator.prediction,
            trace=self.collator.trace,
            ar_framing=self.collator.ar_framing,
            tasks=self.task_configs,
            max_tokens=self.config.max_tokens,
            pack_documents=self.config.pack_documents,
        )

    def train_dataloader(self) -> Iterable[ModelBatch]:
        return self._dataloader(shuffle=True)

    def validation_dataloader(self) -> Iterable[ModelBatch]:
        return self._dataloader(shuffle=False)

    def _dataloader(self, *, shuffle: bool) -> Iterable[ModelBatch]:
        if self._train_dataset is None:
            raise RuntimeError(
                "text loader setup() must run before building a dataloader."
            )
        loader = self.config.dataloader
        num_workers = self.num_workers
        if not isinstance(self.collator.runtime, TextRuntimeSnapshot):
            self.collator.runtime = cast(
                TextRuntime,
                cast(object, TextRuntimeSnapshot.from_runtime(self.runtime)),
            )
        dataset = _limit_dataset(self._train_dataset, self.max_samples)
        map_style = not isinstance(dataset, (IterableAnyDataset, IterableDataset))
        return DataLoader(
            dataset,
            batch_size=loader.batch_size,
            shuffle=(shuffle and map_style),
            num_workers=num_workers,
            pin_memory=loader.pin_memory,
            persistent_workers=(loader.persistent_workers and num_workers > 0),
            collate_fn=self.collator,
        )


def _samples(
    dataset: Dataset[Sample] | IterableAnyDataset,
    indices: Sequence[int],
) -> list[Sample]:
    if not indices:
        return []
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise TypeError("text sample indices must contain integers.")
    if any(index < 0 for index in indices):
        raise ValueError("text sample indices must be non-negative.")
    if not isinstance(dataset, IterableAnyDataset):
        return [dataset[index] for index in indices]

    selected: dict[int, Sample] = {}
    requested = set(indices)
    iterator: Iterator[tuple[int, Sample]] = dataset.iter_shard(1, 0)
    for index, sample in islice(iterator, max(requested) + 1):
        if index in requested:
            selected[index] = sample
    missing = requested - set(selected)
    if missing:
        raise IndexError(f"text sample index {min(missing)} is outside the dataset.")
    return [selected[index] for index in indices]


def _limit_dataset(
    dataset: Dataset[Sample] | IterableAnyDataset,
    max_samples: int | None,
) -> Dataset[Sample] | IterableAnyDataset | IterableDataset[Sample]:
    if max_samples is None:
        return dataset
    if isinstance(max_samples, bool) or not isinstance(max_samples, int):
        raise TypeError("text max_samples must be an integer or None.")
    if max_samples <= 0:
        raise ValueError("text max_samples must be positive.")
    if not isinstance(dataset, IterableAnyDataset):
        length = len(cast(Sized, dataset))
        return Subset(dataset, range(min(max_samples, length)))
    return _LimitedAnyDataset(dataset, max_samples)


class _LimitedAnyDataset(IterableDataset[Sample]):
    def __init__(self, dataset: IterableAnyDataset, max_samples: int) -> None:
        self.dataset = dataset
        self.max_samples = max_samples

    def __iter__(self) -> Iterator[Sample]:
        for index, sample in self.dataset.iter_shard(1, 0):
            if index >= self.max_samples:
                break
            yield sample


__all__ = [
    "DocumentTextDataset",
    "JSONLTextDataset",
    "JsonlTextDataset",
    "PlainTextDataset",
    "TextConfig",
    "TextDatasetConfig",
    "TextDatasetName",
    "TextLoader",
    "load_text_dataset",
    "ToyTextDataset",
]
