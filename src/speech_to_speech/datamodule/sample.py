from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Union

from anydataset.types import Item, Modality, Reference, Sample
from torch import Tensor

from .._compat import StrEnum, auto
from .._tensor import is_signed_integer_dtype
from ..task import PredictionModality, SourceLayout, Task, resolve_response
from .loader.contract import ARFraming, validate_ar_framing


def seconds(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


@dataclass(frozen=True)
class AudioContextSample(Mapping[Reference, Item]):
    """Keep an enrollment utterance separate from task source/target roles."""

    sample: Sample
    audio_context: Sample

    def __getitem__(self, key: Reference) -> Item:
        return self.sample[key]

    def __iter__(self) -> Iterator[Reference]:
        return iter(self.sample)

    def __len__(self) -> int:
        return len(self.sample)


@dataclass(frozen=True)
class AudioContextCostRow:
    """Keep target and enrollment metadata rows separate during cost planning."""

    sample: object
    audio_context: object


class DataShape(StrEnum):
    PAIR = auto()
    SINGLE = auto()


class Language(StrEnum):
    ZH = "Chinese"
    EN = "English"

    @property
    def code(self) -> str:
        return self.name.lower()

    @classmethod
    def _missing_(cls, value: object) -> Language | None:
        if not isinstance(value, str):
            return None
        normalized = value.lower()
        if normalized in {"zh", "zh-cn", "zh_cn", "chinese"}:
            return cls.ZH
        if normalized in {"en", "en-us", "en_us", "english"}:
            return cls.EN
        return None


@dataclass
class Speech:
    semantic_codes: Tensor
    acoustic_codes: Tensor | None
    text_token_ids: Tensor
    audio_token_ids: Tensor
    audio_token_spans: Tensor
    language: Language
    duration_seconds: float | None = None
    global_codes: Tensor | None = None

    def __post_init__(self) -> None:
        if self.semantic_codes.dim() != 2:
            raise ValueError("semantic_codes must have shape [frames, codebooks].")
        if self.acoustic_codes is not None:
            if self.acoustic_codes.dim() != 2:
                raise ValueError("acoustic_codes must have shape [frames, codebooks].")
            if self.acoustic_codes.size(0) != self.semantic_codes.size(0):
                raise ValueError(
                    "semantic_codes and acoustic_codes must share the frame axis."
                )
        if self.global_codes is not None:
            if self.global_codes.dim() != 2:
                raise ValueError("global_codes must have shape [slots, codebooks].")
            if self.acoustic_codes is not None:
                raise ValueError(
                    "Speech cannot contain global and aligned acoustic codes together."
                )
        if self.audio_token_ids.dim() != 1 or self.audio_token_spans.shape != (
            self.audio_token_ids.numel(),
        ):
            raise ValueError("audio token ids and spans must be aligned 1D tensors.")
        if int(self.audio_token_spans.sum().item()) != self.semantic_codes.size(0):
            raise ValueError("audio token spans must cover all semantic frames.")
        seconds(self.duration_seconds, name="speech duration_seconds")


@dataclass
class SpeechPair:
    source: Speech
    target: Speech


@dataclass
class Text:
    text_token_ids: Tensor
    language: Language

    def __post_init__(self) -> None:
        if self.text_token_ids.dim() != 1:
            raise ValueError("text_token_ids must have shape [tokens].")
        if not is_signed_integer_dtype(self.text_token_ids.dtype):
            raise TypeError("text_token_ids must use a signed integer dtype.")


@dataclass
class TextPair:
    source: Text
    target: Text


@dataclass
class RawSpeech:
    text_token_ids: Tensor
    waveform: Tensor
    sample_rate: int
    language: Language
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.text_token_ids.dim() != 1:
            raise ValueError("raw speech text_token_ids must have shape [tokens].")
        if not is_signed_integer_dtype(self.text_token_ids.dtype):
            raise TypeError("raw speech text_token_ids must use signed integer dtype.")
        if self.waveform.dim() not in {1, 2}:
            raise ValueError(
                "raw speech waveform must have shape [time] or [channel, time]."
            )
        if self.waveform.numel() == 0:
            raise ValueError("raw speech waveform must not be empty.")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("raw speech sample_rate must be an integer.")
        if self.sample_rate <= 0:
            raise ValueError("raw speech sample_rate must be positive.")
        if not isinstance(self.language, Language):
            raise TypeError("raw speech language must be a Language.")
        seconds(self.duration_seconds, name="raw speech duration_seconds")

    def pin_memory(self) -> RawSpeech:
        return RawSpeech(
            text_token_ids=self.text_token_ids.pin_memory(),
            waveform=self.waveform.pin_memory(),
            sample_rate=self.sample_rate,
            language=self.language,
            duration_seconds=self.duration_seconds,
        )


@dataclass
class SpeechTaskSample:
    source: Union[Speech, Text, RawSpeech, None]
    target: Union[Speech, Text, RawSpeech]
    task: Task
    trace: str | None = None
    audio_context: Union[Speech, RawSpeech, None] = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, Task):
            raise TypeError("speech task sample task must be a Task.")
        response = resolve_response(self.task, trace=self.trace)
        self.trace = response.name
        _validate_source_item(self.source, self.task.source_layout, name="source")
        _validate_target_item(
            self.target,
            self.task,
            prediction=response.prediction,
            name="target",
        )
        if self.audio_context is not None and not isinstance(
            self.audio_context,
            (Speech, RawSpeech),
        ):
            raise TypeError("speech task audio_context must be Speech or RawSpeech.")

    @property
    def prediction(self) -> PredictionModality:
        return resolve_response(self.task, trace=self.trace).prediction

    @property
    def needs_codec(self) -> bool:
        return (
            isinstance(self.source, RawSpeech)
            or isinstance(self.target, RawSpeech)
            or isinstance(self.audio_context, RawSpeech)
        )

    def pin_memory(self) -> SpeechTaskSample:
        target = _pin_task_item(self.target)
        if target is None:
            raise AssertionError("speech task target must not be None.")
        return SpeechTaskSample(
            source=_pin_task_item(self.source),
            target=target,
            task=self.task,
            trace=self.trace,
            audio_context=_audio_context(_pin_task_item(self.audio_context)),
        )


@dataclass
class RawSpeechBatch:
    samples: tuple[SpeechTaskSample, ...]
    pad_token_id: int
    interleave_audio_frames: int = 25
    mask_text_ratio: float = 0.5
    mask_audio_ratio: float = 0.5
    ar_framing: ARFraming = ARFraming.INSTRUCTION

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("RawSpeechBatch requires at least one sample.")
        if isinstance(self.pad_token_id, bool) or not isinstance(
            self.pad_token_id, int
        ):
            raise TypeError("RawSpeechBatch pad_token_id must be an integer.")
        if (
            isinstance(self.interleave_audio_frames, bool)
            or not isinstance(self.interleave_audio_frames, int)
            or self.interleave_audio_frames < 1
        ):
            raise ValueError("RawSpeechBatch interleave_audio_frames must be positive.")
        _validate_ratio(self.mask_text_ratio, name="mask_text_ratio")
        _validate_ratio(self.mask_audio_ratio, name="mask_audio_ratio")
        if not any(sample.needs_codec for sample in self.samples):
            raise ValueError("RawSpeechBatch requires at least one waveform to encode.")
        signatures = {
            (sample.task.source_layout, sample.prediction) for sample in self.samples
        }
        if len(signatures) != 1:
            raise ValueError(
                "all raw speech samples in a batch must use the same execution signature."
            )
        validate_ar_framing(self.ar_framing, self.tasks)

    @property
    def tasks(self) -> list[Task]:
        return [sample.task for sample in self.samples]

    def pin_memory(self) -> RawSpeechBatch:
        return RawSpeechBatch(
            samples=tuple(sample.pin_memory() for sample in self.samples),
            pad_token_id=self.pad_token_id,
            interleave_audio_frames=self.interleave_audio_frames,
            mask_text_ratio=self.mask_text_ratio,
            mask_audio_ratio=self.mask_audio_ratio,
            ar_framing=self.ar_framing,
        )


def _validate_ratio(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a float.")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")


def _validate_source_item(
    item: Speech | Text | RawSpeech | None,
    layout: SourceLayout,
    *,
    name: str,
) -> None:
    if layout is SourceLayout.NONE:
        if item is not None:
            raise ValueError(f"modality-free task {name} must be None.")
        return
    if layout is SourceLayout.TEXT_AUDIO:
        if not isinstance(item, (Speech, RawSpeech)):
            raise TypeError(f"{name} TEXT_AUDIO must be Speech or RawSpeech.")
        return
    modality = layout.as_modality()
    if modality is None:
        raise AssertionError(f"unexpected source layout: {layout.value}")
    expected = (Speech, RawSpeech) if modality is Modality.AUDIO else (Text,)
    if not isinstance(item, expected):
        expected_names = " or ".join(value.__name__ for value in expected)
        raise TypeError(f"{name} {modality.value} must be {expected_names}.")


def _validate_task_item(
    item: Speech | Text | RawSpeech | None,
    modality: Modality | None,
    *,
    name: str,
) -> None:
    if modality is None:
        if item is not None:
            raise ValueError(f"modality-free task {name} must be None.")
        return
    expected = (Speech, RawSpeech) if modality is Modality.AUDIO else (Text,)
    if not isinstance(item, expected):
        expected_names = " or ".join(value.__name__ for value in expected)
        raise TypeError(f"{name} {modality.value} must be {expected_names}.")


def _validate_target_item(
    item: Speech | Text | RawSpeech | None,
    task: Task,
    *,
    prediction: PredictionModality,
    name: str,
) -> None:
    del task
    if prediction.supervises_audio:
        if not isinstance(item, (Speech, RawSpeech)):
            raise TypeError(
                f"{name} for {prediction.value} must be Speech or RawSpeech."
            )
        return
    if prediction.supervises_text:
        if not isinstance(item, Text):
            raise TypeError(f"{name} for {prediction.value} must be Text.")
        return
    raise ValueError(f"unsupported prediction modality: {prediction.value}")


def _pin_task_item(
    item: Speech | Text | RawSpeech | None,
) -> Speech | Text | RawSpeech | None:
    if item is None:
        return None
    if isinstance(item, RawSpeech):
        return item.pin_memory()
    if isinstance(item, Text):
        return Text(
            text_token_ids=item.text_token_ids.pin_memory(),
            language=item.language,
        )
    return Speech(
        semantic_codes=item.semantic_codes.pin_memory(),
        acoustic_codes=(
            None if item.acoustic_codes is None else item.acoustic_codes.pin_memory()
        ),
        text_token_ids=item.text_token_ids.pin_memory(),
        audio_token_ids=item.audio_token_ids.pin_memory(),
        audio_token_spans=item.audio_token_spans.pin_memory(),
        language=item.language,
        duration_seconds=item.duration_seconds,
        global_codes=(
            None if item.global_codes is None else item.global_codes.pin_memory()
        ),
    )


def _audio_context(
    item: Speech | Text | RawSpeech | None,
) -> Speech | RawSpeech | None:
    if item is None or isinstance(item, (Speech, RawSpeech)):
        return item
    raise TypeError("speech task audio_context must not contain text.")

__all__ = [
    "AudioContextCostRow",
    "AudioContextSample",
    "DataShape",
    "Language",
    "RawSpeech",
    "RawSpeechBatch",
    "Speech",
    "SpeechPair",
    "SpeechTaskSample",
    "Text",
    "TextPair",
]
