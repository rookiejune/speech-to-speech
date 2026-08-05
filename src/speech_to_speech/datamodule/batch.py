from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Union

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout
from torch import Tensor

from ..task import (
    PredictionModality,
    Request,
    Task,
    normalize_language_code,
    resolve_response,
    uses_source_ctc,
    uses_target_ctc,
)


from ._batch_ops import (
    _BatchGenerationFields,
    _BatchUnitCounts,
    _acoustic_target_mask,
    _batch_unit_counts,
    _checked_batch_generation_fields,
    _complete_batch_generation_fields,
    _ctc_target_row,
    _optional_row,
    _padded_samples,
    _pin_ctc_target,
    _pin_optional,
    _pin_target,
    _target_row,
    _to_ctc_target,
    _to_optional,
    _to_target,
    _validate_batch_acoustic,
    _validate_batch_ctc,
    _validate_batch_target_labels,
    _validate_batch_tasks,
    _validate_batch_tensors,
)
from .contract import AcousticTarget, CTCTarget, Labels
from .sample import RawSpeechBatch


@dataclass
class ModelSample:
    """Training sample: shared generation Request plus Labels."""

    request: Request
    labels: Labels

    def __post_init__(self) -> None:
        if not isinstance(self.request["task"], Task):
            raise TypeError("ModelSample task must be a Task.")
        if "prediction" in self.request:
            raise ValueError(
                "ModelSample request prediction override is not supported; "
                "select a task response with trace."
            )
        response = resolve_response(
            self.request["task"],
            trace=self.request.get("trace"),
        )
        self.request["trace"] = response.name
        target_language = self.request.get("target_language")
        if target_language is not None:
            self.request["target_language"] = normalize_language_code(target_language)
        full = torch.cat([self.request["prompt_ids"], self.labels.response_ids])
        if self.labels.token_labels.shape != full.shape:
            raise ValueError(
                "token_labels must align with cat(prompt_ids, response_ids)."
            )

    @classmethod
    def pack(
        cls,
        *,
        prompt_ids: Tensor,
        response_ids: Tensor,
        token_labels: Tensor,
        task: Task,
        trace: str | None = None,
        target_language: str | None = None,
        acoustic_target: AcousticTarget | None = None,
        source_ctc: CTCTarget | None = None,
        target_ctc: CTCTarget | None = None,
        audio_seconds: float = 0.0,
        audio_input_positions: Tensor | None = None,
    ) -> ModelSample:
        request = Request(
            prompt_ids=prompt_ids,
            task=task,
            trace=resolve_response(task, trace=trace).name,
            audio_input_positions=audio_input_positions,
        )
        if target_language is not None:
            request["target_language"] = normalize_language_code(target_language)
        return cls(
            request=request,
            labels=Labels(
                response_ids=response_ids,
                token_labels=token_labels,
                acoustic_target=acoustic_target,
                source_ctc=source_ctc,
                target_ctc=target_ctc,
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
        trace: str | None = None,
        target_language: str | None = None,
        generation_prompt_length: int | None = None,
        acoustic_target: AcousticTarget | None = None,
        source_ctc: CTCTarget | None = None,
        target_ctc: CTCTarget | None = None,
        audio_seconds: float = 0.0,
        audio_input_positions: Tensor | None = None,
    ) -> ModelSample:
        """Split a teacher-forcing sequence into Request prompt and Labels response."""
        if generation_prompt_length is None:
            positions = token_labels.ne(-100).nonzero(as_tuple=False)
            if positions.numel() == 0:
                raise ValueError("model sample must contain at least one target token.")
            generation_prompt_length = int(positions[0].item())
        if isinstance(generation_prompt_length, bool) or not isinstance(
            generation_prompt_length, int
        ):
            raise TypeError("generation_prompt_length must be an integer or None.")
        if (
            generation_prompt_length < 1
            or generation_prompt_length >= input_ids.numel()
        ):
            raise ValueError(
                "generation_prompt_length must leave a non-empty generated response."
            )
        return cls.pack(
            prompt_ids=input_ids[:generation_prompt_length],
            response_ids=input_ids[generation_prompt_length:],
            token_labels=token_labels,
            task=task,
            trace=trace,
            target_language=target_language,
            acoustic_target=acoustic_target,
            source_ctc=source_ctc,
            target_ctc=target_ctc,
            audio_seconds=audio_seconds,
            audio_input_positions=audio_input_positions,
        )

    @property
    def input_ids(self) -> Tensor:
        return torch.cat([self.request["prompt_ids"], self.labels.response_ids])

    @property
    def token_labels(self) -> Tensor:
        return self.labels.token_labels

    @property
    def acoustic_target(self) -> AcousticTarget | None:
        return self.labels.acoustic_target

    @property
    def source_ctc(self) -> CTCTarget | None:
        return self.labels.source_ctc

    @property
    def target_ctc(self) -> CTCTarget | None:
        return self.labels.target_ctc

    @property
    def task(self) -> Task:
        return self.request["task"]

    @property
    def prediction(self) -> PredictionModality:
        return resolve_response(self.task, trace=self.trace).prediction

    @property
    def trace(self) -> str:
        trace = self.request.get("trace")
        if not isinstance(trace, str) or not trace:
            raise TypeError("ModelSample trace must be a non-empty string.")
        return trace

    @property
    def target_language(self) -> str | None:
        return self.request.get("target_language")

    @property
    def audio_seconds(self) -> float:
        return self.labels.audio_seconds

    @property
    def generation_prompt_length(self) -> int:
        return int(self.request["prompt_ids"].numel())

    @property
    def audio_input_positions(self) -> Tensor | None:
        return self.request["audio_input_positions"]


@dataclass
class ModelBatch:
    input_ids: Tensor
    token_labels: Tensor
    acoustic_target: AcousticTarget | None
    tasks: list[Task]
    pad_token_id: int
    traces: list[str] | None = None
    target_languages: list[str | None] | None = None
    source_ctc: CTCTarget | None = None
    target_ctc: CTCTarget | None = None
    audio_seconds: Tensor | None = None
    generation_prompt_lengths: Tensor | None = None
    audio_input_positions: Tensor | None = None
    _unit_counts: _BatchUnitCounts = field(init=False, repr=False)
    _input_modalities: frozenset[Modality] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _audio_input_positions_validated: bool = field(
        init=False,
        default=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        batch_size = _validate_batch_tensors(
            self.input_ids,
            self.token_labels,
        )
        self.traces = _resolved_traces(
            self.tasks,
            self.traces,
            batch_size,
        )
        self.target_languages = _resolved_target_languages(
            self.target_languages,
            batch_size,
        )
        prediction = _validate_batch_tasks(
            self.tasks,
            self.predictions,
            batch_size,
        )
        fields = _complete_batch_generation_fields(
            self.input_ids,
            self.token_labels,
            audio_seconds=self.audio_seconds,
            generation_prompt_lengths=self.generation_prompt_lengths,
            audio_input_positions=self.audio_input_positions,
        )
        self.audio_seconds = fields.audio_seconds
        self.generation_prompt_lengths = fields.generation_prompt_lengths
        self.audio_input_positions = fields.audio_input_positions
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
        _validate_batch_ctc(
            self.input_ids,
            self.source_ctc,
            name="source CTC target",
            causal=False,
            allowed=[uses_source_ctc(task) for task in self.tasks],
        )
        _validate_batch_ctc(
            self.input_ids,
            self.target_ctc,
            name="target CTC target",
            causal=True,
            allowed=[
                uses_target_ctc(task, trace=trace)
                for task, trace in zip(self.tasks, self._traces())
            ],
        )
        self._unit_counts = _batch_unit_counts(
            self.input_ids,
            self.pad_token_id,
            self.acoustic_target,
            fields.audio_seconds,
        )

    @property
    def predictions(self) -> list[PredictionModality]:
        return [
            resolve_response(task, trace=trace).prediction
            for task, trace in zip(self.tasks, self._traces())
        ]

    @property
    def prediction_modality(self) -> PredictionModality:
        return self.predictions[0]

    @property
    def response_traces(self) -> list[str]:
        return list(self._traces())

    @classmethod
    def from_samples(
        cls,
        samples: list[ModelSample],
        *,
        pad_token_id: int,
        layout: Layout | None = None,
    ) -> ModelBatch:
        padded = _padded_samples(samples, pad_token_id, layout=layout)
        return cls(
            input_ids=padded.input_ids,
            token_labels=padded.token_labels,
            acoustic_target=padded.acoustic_target,
            tasks=padded.tasks,
            pad_token_id=pad_token_id,
            traces=padded.traces,
            target_languages=padded.target_languages,
            source_ctc=padded.source_ctc,
            target_ctc=padded.target_ctc,
            audio_seconds=padded.audio_seconds,
            generation_prompt_lengths=padded.generation_prompt_lengths,
            audio_input_positions=padded.audio_input_positions,
        )

    @cached_property
    def attention_mask(self) -> Tensor:
        return self.input_ids != self.pad_token_id

    @cached_property
    def acoustic_target_mask(self) -> Tensor | None:
        return _acoustic_target_mask(self.acoustic_target)

    def pin_memory(self) -> ModelBatch:
        fields = self._generation_fields()
        return self._replace(
            input_ids=self.input_ids.pin_memory(),
            token_labels=self.token_labels.pin_memory(),
            acoustic_target=_pin_target(self.acoustic_target),
            source_ctc=_pin_ctc_target(self.source_ctc),
            target_ctc=_pin_ctc_target(self.target_ctc),
            fields=_BatchGenerationFields(
                audio_seconds=fields.audio_seconds.pin_memory(),
                generation_prompt_lengths=fields.generation_prompt_lengths.pin_memory(),
                audio_input_positions=_pin_optional(fields.audio_input_positions),
            ),
        )

    def to(
        self,
        device: torch.device,
        *,
        non_blocking: bool = False,
    ) -> ModelBatch:
        fields = self._generation_fields()
        return self._replace(
            input_ids=self.input_ids.to(device=device, non_blocking=non_blocking),
            token_labels=self.token_labels.to(
                device=device,
                non_blocking=non_blocking,
            ),
            acoustic_target=_to_target(
                self.acoustic_target,
                device,
                non_blocking=non_blocking,
            ),
            source_ctc=_to_ctc_target(
                self.source_ctc,
                device,
                non_blocking=non_blocking,
            ),
            target_ctc=_to_ctc_target(
                self.target_ctc,
                device,
                non_blocking=non_blocking,
            ),
            fields=_BatchGenerationFields(
                audio_seconds=fields.audio_seconds.to(
                    device=device,
                    non_blocking=non_blocking,
                ),
                generation_prompt_lengths=fields.generation_prompt_lengths.to(
                    device=device,
                    non_blocking=non_blocking,
                ),
                audio_input_positions=_to_optional(
                    fields.audio_input_positions,
                    device,
                    non_blocking=non_blocking,
                ),
            ),
        )

    def training_units(self, unit: str) -> tuple[float, float | None]:
        counts = self._unit_counts
        if unit == "tokens":
            return counts.tokens, counts.padded_tokens
        if unit == "frames":
            if counts.frames is None:
                raise ValueError("frames require an acoustic target mask.")
            return counts.frames, counts.padded_frames
        if unit == "audio_seconds":
            return counts.audio_seconds, None
        raise ValueError(f"unsupported training unit: {unit!r}.")

    @property
    def supervised_token_count(self) -> int:
        """Number of non-ignore-index labels used by causal token loss."""
        return int(self.token_labels.ne(-100).sum().item())

    @property
    def input_modalities(self) -> frozenset[Modality] | None:
        """Exact input modalities validated before this batch moved to a device."""
        return self._input_modalities

    @property
    def audio_input_positions_validated(self) -> bool:
        return self._audio_input_positions_validated

    def set_input_hints(
        self,
        modalities: frozenset[Modality],
        *,
        audio_input_positions_validated: bool,
    ) -> None:
        if not modalities:
            raise ValueError("input modality hints must not be empty.")
        if not all(isinstance(modality, Modality) for modality in modalities):
            raise TypeError("input modality hints must contain Modality values.")
        self._input_modalities = frozenset(modalities)
        self._audio_input_positions_validated = audio_input_positions_validated

    def row(self, index: int) -> ModelBatch:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("ModelBatch row index must be an integer.")
        if index < 0 or index >= self.input_ids.size(0):
            raise IndexError(f"ModelBatch row index is out of range: {index}.")
        fields = self._generation_fields()
        return self._replace(
            input_ids=self.input_ids[index : index + 1],
            token_labels=self.token_labels[index : index + 1],
            acoustic_target=_target_row(self.acoustic_target, index),
            source_ctc=_ctc_target_row(self.source_ctc, index),
            target_ctc=_ctc_target_row(self.target_ctc, index),
            tasks=[self.tasks[index]],
            traces=[self._traces()[index]],
            target_languages=[self._target_languages()[index]],
            fields=_BatchGenerationFields(
                audio_seconds=fields.audio_seconds[index : index + 1],
                generation_prompt_lengths=fields.generation_prompt_lengths[
                    index : index + 1
                ],
                audio_input_positions=_optional_row(
                    fields.audio_input_positions,
                    index,
                ),
            ),
        )

    def _generation_fields(self) -> _BatchGenerationFields:
        return _checked_batch_generation_fields(
            self.audio_seconds,
            self.generation_prompt_lengths,
            self.audio_input_positions,
        )

    def _traces(self) -> list[str]:
        if self.traces is None:
            raise RuntimeError("ModelBatch traces are unavailable after validation.")
        return self.traces

    def _target_languages(self) -> list[str | None]:
        if self.target_languages is None:
            raise RuntimeError(
                "ModelBatch target languages are unavailable after validation."
            )
        return self.target_languages

    def _replace(
        self,
        *,
        input_ids: Tensor,
        token_labels: Tensor,
        acoustic_target: AcousticTarget | None,
        source_ctc: CTCTarget | None,
        target_ctc: CTCTarget | None,
        fields: _BatchGenerationFields,
        tasks: list[Task] | None = None,
        traces: list[str] | None = None,
        target_languages: list[str | None] | None = None,
    ) -> ModelBatch:
        result = ModelBatch.__new__(ModelBatch)
        result.input_ids = input_ids
        result.token_labels = token_labels
        result.acoustic_target = acoustic_target
        result.source_ctc = source_ctc
        result.target_ctc = target_ctc
        result.tasks = list(self.tasks) if tasks is None else tasks
        result.traces = list(self._traces()) if traces is None else traces
        result.target_languages = (
            list(self._target_languages())
            if target_languages is None
            else target_languages
        )
        result.pad_token_id = self.pad_token_id
        result.audio_seconds = fields.audio_seconds
        result.generation_prompt_lengths = fields.generation_prompt_lengths
        result.audio_input_positions = fields.audio_input_positions
        if tasks is None and traces is None and target_languages is None:
            result._unit_counts = self._unit_counts
            result._input_modalities = self._input_modalities
            result._audio_input_positions_validated = (
                self._audio_input_positions_validated
            )
        else:
            result._unit_counts = _batch_unit_counts(
                input_ids,
                self.pad_token_id,
                acoustic_target,
                fields.audio_seconds,
            )
            result._input_modalities = None
            result._audio_input_positions_validated = False
        return result


def _resolved_traces(
    tasks: list[Task],
    traces: list[str] | None,
    batch_size: int,
) -> list[str]:
    if len(tasks) != batch_size:
        raise ValueError("ModelBatch tasks must provide one Task per row.")
    if any(not isinstance(task, Task) for task in tasks):
        raise TypeError("ModelBatch tasks must contain Task values.")
    if traces is not None and len(traces) != batch_size:
        raise ValueError("ModelBatch traces must provide one value per row.")
    values = traces if traces is not None else [None] * batch_size
    resolved: list[str] = []
    for task, trace in zip(tasks, values):
        if trace is not None and (not isinstance(trace, str) or not trace):
            raise TypeError("ModelBatch traces must contain non-empty strings.")
        resolved.append(resolve_response(task, trace=trace).name)
    return resolved


def _resolved_target_languages(
    values: list[str | None] | None,
    batch_size: int,
) -> list[str | None]:
    if values is None:
        return [None] * batch_size
    if len(values) != batch_size:
        raise ValueError(
            "ModelBatch target_languages must provide one value per row."
        )
    return [
        None if value is None else normalize_language_code(value)
        for value in values
    ]


TrainInput = Union[ModelBatch, RawSpeechBatch]


@dataclass(frozen=True)
class LoaderBatch:
    batch: TrainInput
    loader_name: str
    loss_scale: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.batch, (ModelBatch, RawSpeechBatch)):
            raise TypeError("LoaderBatch batch must be ModelBatch or RawSpeechBatch.")
        if not isinstance(self.loader_name, str) or not self.loader_name:
            raise TypeError("LoaderBatch loader_name must be a non-empty string.")
        if self.loss_scale is not None:
            if isinstance(self.loss_scale, bool) or not isinstance(
                self.loss_scale,
                (float, int),
            ):
                raise TypeError("LoaderBatch loss_scale must be a number or None.")
            if not math.isfinite(self.loss_scale) or self.loss_scale < 0:
                raise ValueError(
                    "LoaderBatch loss_scale must be finite and non-negative."
                )

    def pin_memory(self) -> LoaderBatch:
        return LoaderBatch(
            self.batch.pin_memory(),
            self.loader_name,
            self.loss_scale,
        )


@dataclass(frozen=True)
class FusedBatch:
    batches: tuple[TrainInput, ...]
    loader_names: tuple[str, ...] | None = None
    loss_weights: tuple[float, ...] | None = None

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
        if self.loader_names is not None:
            if len(self.loader_names) != len(self.batches):
                raise ValueError(
                    "FusedBatch loader_names must align with microbatches."
                )
            if any(not isinstance(name, str) or not name for name in self.loader_names):
                raise TypeError("FusedBatch loader_names must be non-empty strings.")
        if self.loss_weights is not None:
            if len(self.loss_weights) != len(self.batches):
                raise ValueError(
                    "FusedBatch loss_weights must align with microbatches."
                )
            if any(
                isinstance(weight, bool) or not isinstance(weight, (float, int))
                for weight in self.loss_weights
            ):
                raise TypeError("FusedBatch loss_weights must contain numbers.")
            if any(
                not math.isfinite(weight) or weight < 0 for weight in self.loss_weights
            ):
                raise ValueError(
                    "FusedBatch loss_weights must be finite and non-negative."
                )

    def pin_memory(self) -> FusedBatch:
        return FusedBatch(
            tuple(batch.pin_memory() for batch in self.batches),
            self.loader_names,
            self.loss_weights,
        )


TrainBatch = Union[TrainInput, LoaderBatch, FusedBatch]

__all__ = [
    "FusedBatch",
    "LoaderBatch",
    "ModelBatch",
    "ModelSample",
    "TrainBatch",
    "TrainInput",
]
