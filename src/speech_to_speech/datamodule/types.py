from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from collections.abc import Iterator, Mapping
from typing import TypeVar, TypedDict, Union

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from anydataset.types import Item, Modality, Reference, Sample
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .._compat import StrEnum, auto
from .._tensor import is_signed_integer_dtype
from ..generation.types import Request
from ..prediction import PredictionModality
from ..source import SourceLayout
from ..task import Task
from ._helper.duration import seconds

ACOUSTIC_PAD_ID = -1


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


class DataShape(StrEnum):
    PAIR = auto()
    SINGLE = auto()


class Language(StrEnum):
    ZH = "Chinese"
    EN = "English"

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
    acoustic_layout: AcousticLayout
    acoustic_unit_length: int | None
    text_token_ids: Tensor
    audio_token_ids: Tensor
    audio_token_spans: Tensor
    language: Language
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.semantic_codes.dim() != 2:
            raise ValueError("semantic_codes must have shape [frames, codebooks].")
        if self.acoustic_codes is not None:
            if self.acoustic_codes.dim() != 2:
                raise ValueError("acoustic_codes must have shape [frames, codebooks].")
            if (
                self.acoustic_layout is AcousticLayout.FRAME_ALIGNED
                and self.acoustic_codes.size(0) != self.semantic_codes.size(0)
            ):
                raise ValueError(
                    "semantic_codes and acoustic_codes must share the frame axis."
                )
            if (
                self.acoustic_unit_length is not None
                and self.acoustic_codes.size(0) != self.acoustic_unit_length
            ):
                raise ValueError(
                    "acoustic_codes must match the codec acoustic unit length."
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
            raise ValueError("raw speech waveform must have shape [time] or [channel, time].")
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
    prediction: PredictionModality
    audio_context: Union[Speech, RawSpeech, None] = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, Task):
            raise TypeError("speech task sample task must be a Task.")
        if not isinstance(self.prediction, PredictionModality):
            raise TypeError("speech task sample prediction must be a PredictionModality.")
        if self.prediction not in self.task.allowed_predictions:
            raise ValueError(
                f"{self.task.value} does not allow prediction={self.prediction.value}."
            )
        _validate_source_item(self.source, self.task.source_layout, name="source")
        _validate_target_item(
            self.target,
            self.task,
            prediction=self.prediction,
            name="target",
        )
        if self.audio_context is not None and not isinstance(
            self.audio_context,
            (Speech, RawSpeech),
        ):
            raise TypeError("speech task audio_context must be Speech or RawSpeech.")

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
            prediction=self.prediction,
            audio_context=_audio_context(_pin_task_item(self.audio_context)),
        )


@dataclass
class RawSpeechBatch:
    samples: tuple[SpeechTaskSample, ...]
    pad_token_id: int

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("RawSpeechBatch requires at least one sample.")
        if isinstance(self.pad_token_id, bool) or not isinstance(self.pad_token_id, int):
            raise TypeError("RawSpeechBatch pad_token_id must be an integer.")
        if not any(sample.needs_codec for sample in self.samples):
            raise ValueError("RawSpeechBatch requires at least one waveform to encode.")
        signatures = {
            (sample.task.source_layout, sample.prediction) for sample in self.samples
        }
        if len(signatures) != 1:
            raise ValueError(
                "all raw speech samples in a batch must use the same execution signature."
            )

    @property
    def tasks(self) -> list[Task]:
        return [sample.task for sample in self.samples]

    def pin_memory(self) -> RawSpeechBatch:
        return RawSpeechBatch(
            samples=tuple(sample.pin_memory() for sample in self.samples),
            pad_token_id=self.pad_token_id,
        )


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
        acoustic_layout=item.acoustic_layout,
        acoustic_unit_length=item.acoustic_unit_length,
        text_token_ids=item.text_token_ids.pin_memory(),
        audio_token_ids=item.audio_token_ids.pin_memory(),
        audio_token_spans=item.audio_token_spans.pin_memory(),
        language=item.language,
        duration_seconds=item.duration_seconds,
    )


def _audio_context(
    item: Speech | Text | RawSpeech | None,
) -> Speech | RawSpeech | None:
    if item is None or isinstance(item, (Speech, RawSpeech)):
        return item
    raise TypeError("speech task audio_context must not contain text.")


class AcousticTarget(TypedDict):
    semantic_codes: Tensor
    codes: Tensor
    token_positions: Tensor


@dataclass(frozen=True)
class Labels:
    """Training-only supervision for the response side of a sample."""

    response_ids: Tensor
    token_labels: Tensor
    token_groups: Tensor | None = None
    acoustic_target: AcousticTarget | None = None
    audio_seconds: float = 0.0


@dataclass
class ModelSample:
    """Training sample: shared generation Request plus Labels."""

    request: Request
    labels: Labels

    def __post_init__(self) -> None:
        if not isinstance(self.request["task"], Task):
            raise TypeError("ModelSample task must be a Task.")
        prediction = self.request.get("prediction")
        if not isinstance(prediction, PredictionModality):
            raise TypeError("ModelSample prediction must be a PredictionModality.")
        if prediction not in self.request["task"].allowed_predictions:
            raise ValueError(
                f"{self.request['task'].value} does not allow prediction={prediction.value}."
            )
        full = torch.cat([self.request["prompt_ids"], self.labels.response_ids])
        if self.labels.token_labels.shape != full.shape:
            raise ValueError(
                "token_labels must align with cat(prompt_ids, response_ids)."
            )
        if self.labels.token_groups is not None and (
            self.labels.token_groups.shape != full.shape
        ):
            raise ValueError(
                "token_groups must align with cat(prompt_ids, response_ids)."
            )

    @classmethod
    def pack(
        cls,
        *,
        prompt_ids: Tensor,
        response_ids: Tensor,
        token_labels: Tensor,
        task: Task,
        prediction: PredictionModality,
        token_groups: Tensor | None = None,
        acoustic_target: AcousticTarget | None = None,
        audio_seconds: float = 0.0,
        audio_input_positions: Tensor | None = None,
        audio_context: SemanticAcousticCodes | None = None,
    ) -> ModelSample:
        return cls(
            request=Request(
                prompt_ids=prompt_ids,
                task=task,
                prediction=prediction,
                audio_input_positions=audio_input_positions,
                audio_context=audio_context,
            ),
            labels=Labels(
                response_ids=response_ids,
                token_labels=token_labels,
                token_groups=token_groups,
                acoustic_target=acoustic_target,
                audio_seconds=audio_seconds,
            ),
        )

    @classmethod
    def from_sequence(
        cls,
        input_ids: Tensor,
        token_labels: Tensor,
        *,
        task: Task,
        prediction: PredictionModality,
        generation_prompt_length: int | None = None,
        token_groups: Tensor | None = None,
        acoustic_target: AcousticTarget | None = None,
        audio_seconds: float = 0.0,
        audio_input_positions: Tensor | None = None,
        audio_context: SemanticAcousticCodes | None = None,
    ) -> ModelSample:
        """Split a teacher-forcing sequence into Request prompt and Labels response."""
        if generation_prompt_length is None:
            positions = token_labels.ne(-100).nonzero(as_tuple=False)
            if positions.numel() == 0:
                raise ValueError("model sample must contain at least one target token.")
            generation_prompt_length = int(positions[0].item())
        if (
            isinstance(generation_prompt_length, bool)
            or not isinstance(generation_prompt_length, int)
        ):
            raise TypeError("generation_prompt_length must be an integer or None.")
        if generation_prompt_length < 1 or generation_prompt_length >= input_ids.numel():
            raise ValueError(
                "generation_prompt_length must leave a non-empty generated response."
            )
        return cls.pack(
            prompt_ids=input_ids[:generation_prompt_length],
            response_ids=input_ids[generation_prompt_length:],
            token_labels=token_labels,
            task=task,
            prediction=prediction,
            token_groups=token_groups,
            acoustic_target=acoustic_target,
            audio_seconds=audio_seconds,
            audio_input_positions=audio_input_positions,
            audio_context=audio_context,
        )

    @property
    def input_ids(self) -> Tensor:
        return torch.cat([self.request["prompt_ids"], self.labels.response_ids])

    @property
    def token_labels(self) -> Tensor:
        return self.labels.token_labels

    @property
    def token_groups(self) -> Tensor | None:
        return self.labels.token_groups

    @property
    def acoustic_target(self) -> AcousticTarget | None:
        return self.labels.acoustic_target

    @property
    def task(self) -> Task:
        return self.request["task"]

    @property
    def prediction(self) -> PredictionModality:
        prediction = self.request.get("prediction")
        if not isinstance(prediction, PredictionModality):
            raise TypeError("ModelSample prediction must be a PredictionModality.")
        return prediction

    @property
    def audio_seconds(self) -> float:
        return self.labels.audio_seconds

    @property
    def generation_prompt_length(self) -> int:
        return int(self.request["prompt_ids"].numel())

    @property
    def audio_input_positions(self) -> Tensor | None:
        return self.request["audio_input_positions"]

    @property
    def audio_context(self) -> SemanticAcousticCodes | None:
        return self.request["audio_context"]


@dataclass
class ModelBatch:
    input_ids: Tensor
    token_labels: Tensor
    acoustic_target: AcousticTarget | None
    tasks: list[Task]
    predictions: list[PredictionModality]
    pad_token_id: int
    token_groups: Tensor | None = None
    audio_seconds: Tensor | None = None
    generation_prompt_lengths: Tensor | None = None
    audio_input_positions: Tensor | None = None
    audio_contexts: tuple[SemanticAcousticCodes | None, ...] | None = None

    def __post_init__(self) -> None:
        if self.input_ids.dim() != 2 or self.token_labels.shape != self.input_ids.shape:
            raise ValueError(
                "batch input ids and token labels must be aligned 2D tensors."
            )
        if not is_signed_integer_dtype(
            self.input_ids.dtype
        ) or not is_signed_integer_dtype(self.token_labels.dtype):
            raise TypeError(
                "batch input ids and token labels must use signed integer dtypes."
            )
        if self.token_groups is not None:
            if self.token_groups.shape != self.input_ids.shape:
                raise ValueError(
                    "batch token groups must align with input ids and token labels."
                )
            if not is_signed_integer_dtype(self.token_groups.dtype):
                raise TypeError("batch token groups must use a signed integer dtype.")
            if bool((self.token_labels.eq(-100) & self.token_groups.ne(-1)).any()):
                raise ValueError("ignored token labels must use the forced token group.")
            if bool((self.token_labels.ne(-100) & self.token_groups.lt(0)).any()):
                raise ValueError("supervised token labels require prediction groups.")
        batch_size = self.input_ids.size(0)
        if batch_size < 1:
            raise ValueError("ModelBatch requires at least one row.")
        if len(self.tasks) != batch_size:
            raise ValueError("ModelBatch tasks must provide one Task per row.")
        if len(self.predictions) != batch_size:
            raise ValueError("ModelBatch predictions must provide one value per row.")
        if any(not isinstance(task, Task) for task in self.tasks):
            raise TypeError("ModelBatch tasks must contain Task values.")
        if any(
            not isinstance(prediction, PredictionModality)
            for prediction in self.predictions
        ):
            raise TypeError(
                "ModelBatch predictions must contain PredictionModality values."
            )
        for task, prediction in zip(self.tasks, self.predictions):
            if prediction not in task.allowed_predictions:
                raise ValueError(
                    f"{task.value} does not allow prediction={prediction.value}."
                )
        if self.audio_seconds is None:
            self.audio_seconds = self.input_ids.new_zeros(
                batch_size,
                dtype=torch.float32,
            )
        _validate_audio_seconds(self.audio_seconds, batch_size)
        if self.generation_prompt_lengths is None:
            self.generation_prompt_lengths = _generation_prompt_lengths(
                self.token_labels
            )
        _validate_generation_prompt_lengths(
            self.generation_prompt_lengths,
            self.input_ids,
        )
        _validate_batch_audio_input_positions(
            self.audio_input_positions,
            self.input_ids,
        )
        if self.audio_contexts is None:
            self.audio_contexts = (None,) * batch_size
        if len(self.audio_contexts) != batch_size:
            raise ValueError("ModelBatch audio_contexts must provide one value per row.")
        for context in self.audio_contexts:
            _validate_audio_context(context)
        signatures = {
            (task.source_layout, prediction)
            for task, prediction in zip(self.tasks, self.predictions)
        }
        if len(signatures) != 1:
            raise ValueError(
                "all samples in a batch must use the same execution signature."
            )
        _, prediction = next(iter(signatures))
        if not prediction.supervises_audio and self.acoustic_target is not None:
            raise ValueError(
                "text-only prediction batches must not provide acoustic target fields."
            )
        _validate_batch_acoustic(
            self.input_ids,
            self.acoustic_target,
            name="acoustic target",
            minimum_position=1,
        )
        _validate_batch_target_labels(self.token_labels, self.acoustic_target)

    @property
    def prediction_modality(self) -> PredictionModality:
        return self.predictions[0]

    @classmethod
    def from_samples(
        cls,
        samples: list[ModelSample],
        *,
        pad_token_id: int,
    ) -> ModelBatch:
        if not samples:
            raise ValueError("ModelBatch requires at least one sample.")
        for sample in samples:
            _validate_sample(sample, pad_token_id)
        input_ids = [
            torch.cat([sample.request["prompt_ids"], sample.labels.response_ids])
            for sample in samples
        ]
        return cls(
            input_ids=_pad(input_ids, pad_token_id),
            token_labels=_pad(
                [sample.labels.token_labels for sample in samples],
                -100,
            ),
            token_groups=_optional_tensor(
                [sample.labels.token_groups for sample in samples],
                padding_value=-1,
            ),
            acoustic_target=_target(
                [sample.labels.acoustic_target for sample in samples]
            ),
            tasks=[sample.request["task"] for sample in samples],
            predictions=[sample.prediction for sample in samples],
            pad_token_id=pad_token_id,
            audio_seconds=_audio_seconds(samples),
            generation_prompt_lengths=_sample_prompt_lengths(samples),
            audio_input_positions=_optional_tensor(
                [sample.request["audio_input_positions"] for sample in samples],
                padding_value=-1,
            ),
            audio_contexts=tuple(
                sample.request["audio_context"] for sample in samples
            ),
        )

    @cached_property
    def attention_mask(self) -> Tensor:
        return self.input_ids != self.pad_token_id

    @cached_property
    def acoustic_target_mask(self) -> Tensor | None:
        if self.acoustic_target is None:
            return None
        code_mask = (self.acoustic_target["codes"] != ACOUSTIC_PAD_ID).all(dim=-1)
        return (self.acoustic_target["token_positions"] >= 0) & code_mask

    def pin_memory(self) -> ModelBatch:
        audio_seconds = self.audio_seconds
        prompt_lengths = self.generation_prompt_lengths
        audio_input_positions = self.audio_input_positions
        audio_contexts = self.audio_contexts
        if audio_seconds is None:
            raise RuntimeError("ModelBatch audio_seconds is unavailable after validation.")
        if prompt_lengths is None or audio_contexts is None:
            raise RuntimeError("ModelBatch generation fields are unavailable after validation.")
        return ModelBatch(
            input_ids=self.input_ids.pin_memory(),
            token_labels=self.token_labels.pin_memory(),
            token_groups=(
                None if self.token_groups is None else self.token_groups.pin_memory()
            ),
            acoustic_target=_pin_target(self.acoustic_target),
            tasks=list(self.tasks),
            predictions=list(self.predictions),
            pad_token_id=self.pad_token_id,
            audio_seconds=audio_seconds.pin_memory(),
            generation_prompt_lengths=prompt_lengths.pin_memory(),
            audio_input_positions=(
                None
                if audio_input_positions is None
                else audio_input_positions.pin_memory()
            ),
            audio_contexts=tuple(
                _pin_audio_context(value) for value in audio_contexts
            ),
        )

    def to(self, device: torch.device) -> ModelBatch:
        audio_seconds = self.audio_seconds
        prompt_lengths = self.generation_prompt_lengths
        audio_input_positions = self.audio_input_positions
        audio_contexts = self.audio_contexts
        if audio_seconds is None:
            raise RuntimeError("ModelBatch audio_seconds is unavailable after validation.")
        if prompt_lengths is None or audio_contexts is None:
            raise RuntimeError("ModelBatch generation fields are unavailable after validation.")
        return ModelBatch(
            input_ids=self.input_ids.to(device=device),
            token_labels=self.token_labels.to(device=device),
            token_groups=(
                None if self.token_groups is None else self.token_groups.to(device=device)
            ),
            acoustic_target=_to_target(self.acoustic_target, device),
            tasks=list(self.tasks),
            predictions=list(self.predictions),
            pad_token_id=self.pad_token_id,
            audio_seconds=audio_seconds.to(device=device),
            generation_prompt_lengths=prompt_lengths.to(device=device),
            audio_input_positions=(
                None
                if audio_input_positions is None
                else audio_input_positions.to(device=device)
            ),
            audio_contexts=tuple(
                _to_audio_context(value, device) for value in audio_contexts
            ),
        )

    def row(self, index: int) -> ModelBatch:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("ModelBatch row index must be an integer.")
        if index < 0 or index >= self.input_ids.size(0):
            raise IndexError(f"ModelBatch row index is out of range: {index}.")
        audio_seconds = self.audio_seconds
        prompt_lengths = self.generation_prompt_lengths
        audio_input_positions = self.audio_input_positions
        audio_contexts = self.audio_contexts
        if audio_seconds is None:
            raise RuntimeError("ModelBatch audio_seconds is unavailable after validation.")
        if prompt_lengths is None or audio_contexts is None:
            raise RuntimeError("ModelBatch generation fields are unavailable after validation.")
        return ModelBatch(
            input_ids=self.input_ids[index : index + 1],
            token_labels=self.token_labels[index : index + 1],
            token_groups=(
                None
                if self.token_groups is None
                else self.token_groups[index : index + 1]
            ),
            acoustic_target=_target_row(self.acoustic_target, index),
            tasks=[self.tasks[index]],
            predictions=[self.predictions[index]],
            pad_token_id=self.pad_token_id,
            audio_seconds=audio_seconds[index : index + 1],
            generation_prompt_lengths=prompt_lengths[index : index + 1],
            audio_input_positions=(
                None
                if audio_input_positions is None
                else audio_input_positions[index : index + 1]
            ),
            audio_contexts=(audio_contexts[index],),
        )


TrainInput = Union[ModelBatch, RawSpeechBatch]


@dataclass(frozen=True)
class FusedBatch:
    batches: tuple[TrainInput, ...]

    def __post_init__(self) -> None:
        if not self.batches:
            raise ValueError("FusedBatch requires at least one microbatch.")
        if any(
            not isinstance(batch, (ModelBatch, RawSpeechBatch))
            for batch in self.batches
        ):
            raise TypeError(
                "FusedBatch microbatches must be ModelBatch or RawSpeechBatch."
            )

    def pin_memory(self) -> FusedBatch:
        return FusedBatch(tuple(batch.pin_memory() for batch in self.batches))


TrainBatch = Union[TrainInput, FusedBatch]


def _pad(values: list[Tensor], padding_value: int) -> Tensor:
    return pad_sequence(
        values,
        batch_first=True,
        padding_value=padding_value,
    )


def _target(values: list[AcousticTarget | None]) -> AcousticTarget | None:
    targets = _present(values)
    if targets is None:
        return None
    return AcousticTarget(
        semantic_codes=_pad(
            [value["semantic_codes"] for value in targets], ACOUSTIC_PAD_ID
        ),
        codes=_pad([value["codes"] for value in targets], ACOUSTIC_PAD_ID),
        token_positions=_pad(
            [value["token_positions"] for value in targets], ACOUSTIC_PAD_ID
        ),
    )


def _optional_tensor(
    values: list[Tensor | None],
    *,
    padding_value: int,
) -> Tensor | None:
    present = _present(values)
    if present is None:
        return None
    return _pad(present, padding_value)


def _pin_target(value: AcousticTarget | None) -> AcousticTarget | None:
    if value is None:
        return None
    return AcousticTarget(
        semantic_codes=value["semantic_codes"].pin_memory(),
        codes=value["codes"].pin_memory(),
        token_positions=value["token_positions"].pin_memory(),
    )


def _to_target(
    value: AcousticTarget | None,
    device: torch.device,
) -> AcousticTarget | None:
    if value is None:
        return None
    return AcousticTarget(
        semantic_codes=value["semantic_codes"].to(device=device),
        codes=value["codes"].to(device=device),
        token_positions=value["token_positions"].to(device=device),
    )


def _target_row(value: AcousticTarget | None, index: int) -> AcousticTarget | None:
    if value is None:
        return None
    return AcousticTarget(
        semantic_codes=value["semantic_codes"][index : index + 1],
        codes=value["codes"][index : index + 1],
        token_positions=value["token_positions"][index : index + 1],
    )


def _audio_seconds(samples: list[ModelSample]) -> Tensor:
    return samples[0].request["prompt_ids"].new_tensor(
        [sample.labels.audio_seconds for sample in samples],
        dtype=torch.float32,
    )


def _sample_prompt_lengths(samples: list[ModelSample]) -> Tensor:
    return samples[0].request["prompt_ids"].new_tensor(
        [sample.request["prompt_ids"].numel() for sample in samples],
        dtype=torch.long,
    )


def _generation_prompt_lengths(token_labels: Tensor) -> Tensor:
    values = []
    for labels in token_labels:
        positions = labels.ne(-100).nonzero(as_tuple=False)
        if positions.numel() == 0:
            raise ValueError("each token label row must contain at least one target token.")
        values.append(int(positions[0].item()))
    return token_labels.new_tensor(values, dtype=torch.long)


def _validate_generation_prompt_lengths(value: Tensor, input_ids: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError("ModelBatch generation_prompt_lengths must be a Tensor.")
    if value.shape != (input_ids.size(0),):
        raise ValueError(
            "ModelBatch generation_prompt_lengths must have shape [batch]."
        )
    if not is_signed_integer_dtype(value.dtype):
        raise TypeError(
            "ModelBatch generation_prompt_lengths must use a signed integer dtype."
        )
    if bool((value < 1).any()) or bool((value >= input_ids.size(1)).any()):
        raise ValueError(
            "generation prompt lengths must leave a non-empty generated response."
        )


def _validate_batch_audio_input_positions(
    value: Tensor | None,
    input_ids: Tensor,
) -> None:
    if value is None:
        return
    if value.dim() != 2 or value.size(0) != input_ids.size(0):
        raise ValueError(
            "ModelBatch audio_input_positions must have shape [batch, frames]."
        )
    if not is_signed_integer_dtype(value.dtype):
        raise TypeError(
            "ModelBatch audio_input_positions must use a signed integer dtype."
        )
    if bool((value < -1).any()) or bool((value >= input_ids.size(1)).any()):
        raise ValueError(
            "ModelBatch audio_input_positions must use -1 padding or valid sequence positions."
        )
    for row in value:
        valid = row[row.ge(0)]
        if valid.numel() != torch.unique(valid).numel():
            raise ValueError("ModelBatch audio_input_positions must not repeat positions.")


def _validate_audio_context(value: SemanticAcousticCodes | None) -> None:
    if value is None:
        return
    for name, codes in (
        ("semantic", value.semantic),
        ("acoustic", value.acoustic),
    ):
        if codes.dim() != 2:
            raise ValueError(
                f"audio context {name} codes must have shape [units, codebooks]."
            )
        if not is_signed_integer_dtype(codes.dtype):
            raise TypeError(f"audio context {name} codes must use a signed integer dtype.")


def _pin_audio_context(
    value: SemanticAcousticCodes | None,
) -> SemanticAcousticCodes | None:
    if value is None:
        return None
    return SemanticAcousticCodes(
        semantic=value.semantic.pin_memory(),
        acoustic=value.acoustic.pin_memory(),
    )


def _to_audio_context(
    value: SemanticAcousticCodes | None,
    device: torch.device,
) -> SemanticAcousticCodes | None:
    if value is None:
        return None
    return SemanticAcousticCodes(
        semantic=value.semantic.to(device=device),
        acoustic=value.acoustic.to(device=device),
    )


def _validate_audio_seconds(value: Tensor, batch_size: int) -> None:
    if not isinstance(value, Tensor):
        raise TypeError("ModelBatch audio_seconds must be a Tensor.")
    if value.shape != (batch_size,):
        raise ValueError("ModelBatch audio_seconds must have shape [batch].")
    if value.dtype == torch.bool or value.is_complex():
        raise TypeError("ModelBatch audio_seconds must use a real numeric dtype.")
    if not bool(torch.isfinite(value).all().item()) or bool(value.lt(0).any().item()):
        raise ValueError("ModelBatch audio_seconds must be finite and non-negative.")


T = TypeVar("T")


def _present(values: list[T | None]) -> list[T] | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values):
        raise ValueError("a batch field must be present for every sample or none.")
    return present


def _validate_sample(sample: ModelSample, pad_token_id: int) -> None:
    prompt_ids = sample.request["prompt_ids"]
    response_ids = sample.labels.response_ids
    input_ids = torch.cat([prompt_ids, response_ids])
    token_labels = sample.labels.token_labels
    if prompt_ids.dim() != 1 or response_ids.dim() != 1:
        raise ValueError("sample prompt_ids and response_ids must be 1D tensors.")
    if prompt_ids.numel() < 1 or response_ids.numel() < 1:
        raise ValueError(
            "generation prompt lengths must leave a non-empty generated response."
        )
    if token_labels.shape != input_ids.shape:
        raise ValueError(
            "sample input ids and token labels must be aligned 1D tensors."
        )
    token_groups = sample.labels.token_groups
    if token_groups is not None:
        if token_groups.shape != input_ids.shape:
            raise ValueError("sample token groups must align with input ids.")
        if not is_signed_integer_dtype(token_groups.dtype):
            raise TypeError("sample token groups must use a signed integer dtype.")
        if bool((token_labels.eq(-100) & token_groups.ne(-1)).any()):
            raise ValueError("ignored sample labels must use the forced token group.")
        if bool((token_labels.ne(-100) & token_groups.lt(0)).any()):
            raise ValueError("supervised sample labels require prediction groups.")

    positions = sample.request["audio_input_positions"]
    if positions is not None:
        if positions.dim() != 1:
            raise ValueError("sample audio_input_positions must have shape [frames].")
        if not is_signed_integer_dtype(positions.dtype):
            raise TypeError("sample audio_input_positions must use a signed integer dtype.")
        if bool((positions < 0).any()) or bool(
            (positions >= input_ids.numel()).any()
        ):
            raise ValueError("sample audio_input_positions must be valid sequence positions.")
        if positions.numel() != torch.unique(positions).numel():
            raise ValueError("sample audio_input_positions must not repeat positions.")
    _validate_audio_context(sample.request["audio_context"])

    target = sample.labels.acoustic_target
    if target is not None:
        _validate_acoustic_pair(
            input_ids,
            target["codes"],
            target["token_positions"],
            name="acoustic target",
            pad_token_id=pad_token_id,
        )
        if bool((target["token_positions"] < 1).any()):
            raise ValueError(
                "acoustic target positions must be at least 1 so every frame has "
                "a causal predecessor."
            )
        semantic_codes = target["semantic_codes"]
        _validate_codes(semantic_codes, name="target semantic codes")
        if semantic_codes.size(0) != target["codes"].size(0):
            raise ValueError(
                "target semantic and acoustic codes must share the frame axis."
            )
    if not sample.prediction.supervises_audio and target is not None:
        raise ValueError(
            "text-only prediction samples must not provide acoustic target fields."
        )
    if target is not None:
        positions = target["token_positions"].to(
            device=token_labels.device,
            dtype=torch.long,
        )
        labels = token_labels[positions]
        if bool(labels.eq(-100).any()):
            raise ValueError("acoustic target positions must point to semantic labels.")


def _validate_batch_acoustic(
    input_ids: Tensor,
    value: AcousticTarget | None,
    *,
    name: str,
    minimum_position: int,
) -> None:
    if value is None:
        return
    codes = value["codes"]
    positions = value["token_positions"]
    if codes.dim() != 3 or positions.dim() != 2:
        raise ValueError(f"{name} batch fields must have shapes [B, F, Q] and [B, F].")
    if not is_signed_integer_dtype(codes.dtype) or not is_signed_integer_dtype(
        positions.dtype
    ):
        raise TypeError(f"{name} batch fields must use signed integer dtypes.")
    if codes.size(-1) < 1:
        raise ValueError(f"{name} codes must contain at least one codebook.")
    if codes.shape[:2] != positions.shape:
        raise ValueError(f"{name} batch fields must align on batch and frame.")
    if positions.size(0) != input_ids.size(0):
        raise ValueError(f"{name} batch must align with input batch size.")
    if codes.device != input_ids.device or positions.device != input_ids.device:
        raise ValueError(f"{name} batch fields must use the input tensor device.")
    active = positions.ge(0)
    if bool((positions < -1).any()):
        raise ValueError(f"{name} positions may only use -1 as padding.")
    if bool((active & positions.lt(minimum_position)).any()):
        raise ValueError(
            f"{name} positions must be at least {minimum_position} for active frames."
        )
    if bool((active & positions.ge(input_ids.size(1))).any()):
        raise ValueError(f"{name} position exceeds the token sequence length.")
    code_mask = codes.ge(0).all(dim=-1)
    code_padding = codes.eq(ACOUSTIC_PAD_ID).all(dim=-1)
    if not bool((code_mask | code_padding).all()):
        raise ValueError(
            f"{name} codes must be non-negative or use -1 for a whole padded frame."
        )
    if not torch.equal(active, code_mask):
        raise ValueError(
            f"{name} positions and codes must share the same padding mask."
        )
    if positions.size(1) < 1 or not bool(active.any(dim=1).all()):
        raise ValueError(f"each {name} batch row must contain an active frame.")
    semantic = value.get("semantic_codes")
    if semantic is not None:
        if semantic.dim() != 3 or semantic.shape[:2] != positions.shape:
            raise ValueError(
                "acoustic target semantic codes must align on batch and frame."
            )
        if not is_signed_integer_dtype(semantic.dtype):
            raise TypeError("acoustic target semantic codes must use a signed dtype.")
        if semantic.size(-1) < 1:
            raise ValueError("acoustic target semantic codes must contain a codebook.")
        if semantic.device != input_ids.device:
            raise ValueError(
                "acoustic target semantic codes must use the input tensor device."
            )
        semantic_mask = semantic.ge(0).all(dim=-1)
        semantic_padding = semantic.eq(ACOUSTIC_PAD_ID).all(dim=-1)
        if not bool((semantic_mask | semantic_padding).all()) or not torch.equal(
            semantic_mask, active
        ):
            raise ValueError(
                "acoustic target semantic codes must share the frame padding mask."
            )


def _validate_batch_target_labels(
    labels: Tensor,
    target: AcousticTarget | None,
) -> None:
    if target is None:
        return
    positions = target["token_positions"]
    if positions.device != labels.device:
        raise ValueError("acoustic target labels and positions must use one device.")
    active = positions.ge(0)
    rows = torch.arange(labels.size(0), device=positions.device)[:, None]
    selected = labels[rows.expand_as(positions)[active], positions[active]]
    if bool(selected.eq(-100).any()):
        raise ValueError("acoustic target positions must point to semantic labels.")


def _validate_acoustic_pair(
    input_ids: Tensor,
    codes: Tensor | None,
    token_positions: Tensor | None,
    *,
    name: str,
    pad_token_id: int,
) -> None:
    if (codes is None) != (token_positions is None):
        raise ValueError(f"{name} codes and token positions must be provided together.")
    if codes is None or token_positions is None:
        return
    _validate_codes(codes, name=f"{name} codes")
    if token_positions.dim() != 1:
        raise ValueError(f"{name} token positions must have shape [frames].")
    if not is_signed_integer_dtype(token_positions.dtype):
        raise TypeError(
            f"{name} token positions must contain integer indices using a signed dtype."
        )
    if codes.size(0) != token_positions.numel():
        raise ValueError(f"{name} codes and token positions must share the frame axis.")
    if bool((token_positions < 0).any()) or bool(
        (token_positions >= input_ids.numel()).any()
    ):
        raise ValueError(f"{name} positions must point inside the token sequence.")
    positions = token_positions.to(device=input_ids.device, dtype=torch.long)
    if bool(input_ids[positions].eq(pad_token_id).any()):
        raise ValueError(f"{name} positions must not point to padding tokens.")


def _validate_codes(codes: Tensor, *, name: str) -> None:
    if codes.dim() != 2:
        raise ValueError(f"{name} must have shape [frames, codebooks].")
    if codes.size(0) == 0 or codes.size(1) == 0:
        raise ValueError(f"{name} must contain at least one frame and codebook.")
    if not is_signed_integer_dtype(codes.dtype):
        raise TypeError(f"{name} must contain integer codec IDs using a signed dtype.")
    if bool((codes < 0).any()):
        raise ValueError(f"{name} must contain non-negative codec IDs.")
