from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict

from anydataset.types import Modality
from torch import Generator, Tensor
from typing_extensions import NotRequired

from .._compat import StrEnum, auto


class PredictionModality(StrEnum):
    """How a task supervises and generates token responses.

    TEXT / AUDIO: single-modality next-token prediction.
    PARALLEL: independent text and audio spans in one sequence (both heads);
        layout is block-wise, not time-aligned interleaving.
    INTERLEAVED: time-chunk text/audio alternation in one sequence.
    """

    TEXT = auto()
    AUDIO = auto()
    PARALLEL = auto()
    INTERLEAVED = auto()

    @property
    def supervises_text(self) -> bool:
        return self in {
            PredictionModality.TEXT,
            PredictionModality.PARALLEL,
            PredictionModality.INTERLEAVED,
        }

    @property
    def supervises_audio(self) -> bool:
        return self in {
            PredictionModality.AUDIO,
            PredictionModality.PARALLEL,
            PredictionModality.INTERLEAVED,
        }

    @property
    def is_mixed(self) -> bool:
        return self in {
            PredictionModality.PARALLEL,
            PredictionModality.INTERLEAVED,
        }

    def supervised_modalities(self) -> frozenset[Modality]:
        modalities: set[Modality] = set()
        if self.supervises_text:
            modalities.add(Modality.TEXT)
        if self.supervises_audio:
            modalities.add(Modality.AUDIO)
        return frozenset(modalities)


class SourceLayout(StrEnum):
    """What modalities appear in the visible source/content for a task."""

    NONE = auto()
    TEXT = auto()
    AUDIO = auto()
    TEXT_AUDIO = auto()

    @property
    def includes_text(self) -> bool:
        return self in {SourceLayout.TEXT, SourceLayout.TEXT_AUDIO}

    @property
    def includes_audio(self) -> bool:
        return self in {SourceLayout.AUDIO, SourceLayout.TEXT_AUDIO}

    def as_modality(self) -> Modality | None:
        if self is SourceLayout.TEXT:
            return Modality.TEXT
        if self is SourceLayout.AUDIO:
            return Modality.AUDIO
        return None


class FieldRole(StrEnum):
    """Dataset role bound to one model-visible or supervised task field."""

    SOURCE = auto()
    TARGET = auto()


class ResponseLayout(StrEnum):
    """How response fields are serialized into the causal token sequence."""

    SEQUENTIAL = auto()
    BLOCKWISE = auto()
    INTERLEAVED = auto()
    MASKED = auto()


class TaskObjective(StrEnum):
    """Training rule applied after a task program is compiled."""

    CAUSAL = auto()
    RECONSTRUCTION = auto()


@dataclass(frozen=True)
class TaskField:
    """One typed dataset field referenced by a task program."""

    role: FieldRole
    modality: Modality

    def __post_init__(self) -> None:
        if not isinstance(self.role, FieldRole):
            raise TypeError("task field role must be a FieldRole.")
        if not isinstance(self.modality, Modality):
            raise TypeError("task field modality must be a Modality.")


@dataclass(frozen=True)
class ResponseSpec:
    """One named response trace supported by a task program."""

    name: str
    fields: tuple[TaskField, ...]
    prediction: PredictionModality
    layout: ResponseLayout = ResponseLayout.SEQUENTIAL
    default_for_prediction: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("response spec name must be a non-empty string.")
        if not self.fields:
            raise ValueError("response spec must contain at least one field.")
        if any(not isinstance(field, TaskField) for field in self.fields):
            raise TypeError("response spec fields must contain TaskField values.")
        if not isinstance(self.prediction, PredictionModality):
            raise TypeError("response spec prediction must be a PredictionModality.")
        if not isinstance(self.layout, ResponseLayout):
            raise TypeError("response spec layout must be a ResponseLayout.")
        if not isinstance(self.default_for_prediction, bool):
            raise TypeError("response default_for_prediction must be a boolean.")
        modalities = frozenset(field.modality for field in self.fields)
        if modalities != self.prediction.supervised_modalities():
            raise ValueError(
                "response fields must exactly match the prediction modalities."
            )


@dataclass(frozen=True)
class TaskProgram:
    """Declarative input, response-trace, layout, and objective contract."""

    context: tuple[TaskField, ...]
    responses: tuple[ResponseSpec, ...]
    default_response: str = "direct"
    objective: TaskObjective = TaskObjective.CAUSAL
    supports_pretraining: bool = False

    def __post_init__(self) -> None:
        if any(not isinstance(field, TaskField) for field in self.context):
            raise TypeError("task program context must contain TaskField values.")
        if not self.responses:
            raise ValueError("task program must provide at least one response.")
        if any(not isinstance(response, ResponseSpec) for response in self.responses):
            raise TypeError("task program responses must contain ResponseSpec values.")
        names = [response.name for response in self.responses]
        if len(names) != len(set(names)):
            raise ValueError("task program response names must be unique.")
        if self.default_response not in names:
            raise ValueError("task program default response must name a response spec.")
        if not isinstance(self.objective, TaskObjective):
            raise TypeError("task program objective must be a TaskObjective.")
        if not isinstance(self.supports_pretraining, bool):
            raise TypeError("task program supports_pretraining must be a boolean.")
        for prediction in self.allowed_predictions:
            defaults = [
                response
                for response in self.responses
                if response.prediction is prediction
                and response.default_for_prediction
            ]
            if len(defaults) != 1:
                raise ValueError(
                    "task program must provide exactly one default response for "
                    f"prediction={prediction.value}."
                )

    @property
    def source_layout(self) -> SourceLayout:
        modalities = frozenset(field.modality for field in self.context)
        if not modalities:
            return SourceLayout.NONE
        if modalities == {Modality.TEXT}:
            return SourceLayout.TEXT
        if modalities == {Modality.AUDIO}:
            return SourceLayout.AUDIO
        if modalities == {Modality.TEXT, Modality.AUDIO}:
            return SourceLayout.TEXT_AUDIO
        raise ValueError("task program context contains unsupported modalities.")

    @property
    def default(self) -> ResponseSpec:
        return self.response(self.default_response)

    @property
    def allowed_predictions(self) -> frozenset[PredictionModality]:
        return frozenset(response.prediction for response in self.responses)

    def response(
        self,
        name: str | None = None,
        *,
        prediction: PredictionModality | None = None,
    ) -> ResponseSpec:
        if name is not None:
            if not isinstance(name, str) or not name:
                raise ValueError("response trace must be a non-empty string or None.")
            matches = [response for response in self.responses if response.name == name]
            if not matches:
                available = ", ".join(response.name for response in self.responses)
                raise ValueError(
                    f"unsupported response trace {name!r}; available: {available}."
                )
            response = matches[0]
            if prediction is not None and response.prediction is not prediction:
                raise ValueError(
                    f"response trace {name!r} requires "
                    f"prediction={response.prediction.value}, got {prediction.value}."
                )
            return response
        if prediction is None:
            return next(
                response
                for response in self.responses
                if response.name == self.default_response
            )
        matches = [
            response
            for response in self.responses
            if response.prediction is prediction and response.default_for_prediction
        ]
        if not matches:
            allowed = ", ".join(
                sorted(value.value for value in self.allowed_predictions)
            )
            raise ValueError(
                f"task program does not allow prediction={prediction.value}; "
                f"allowed: {allowed}."
            )
        return matches[0]


class Request(TypedDict):
    """Task-level tensor request shared by data and generation services."""

    prompt_ids: Tensor
    task: Task
    audio_input_positions: Tensor | None
    prediction: NotRequired[PredictionModality | None]
    trace: NotRequired[str]
    semantic_reference_features: NotRequired[Tensor | None]
    semantic_reference_mask: NotRequired[Tensor | None]
    semantic_decode_generator: NotRequired[Generator | None]

class Task(StrEnum):
    AUDIO_AR = auto()
    ASR = auto()
    INTERLEAVED_AR = auto()
    MASKED_AR = auto()
    MT = auto()
    PARALLEL_AR = auto()
    S2ST = auto()
    S2TT = auto()
    TEXT_AR = auto()
    T2ST = auto()
    T2TT = auto()
    TTS = auto()

    @property
    def program(self) -> TaskProgram:
        from .program import program_for

        return program_for(self)

    @property
    def source_layout(self) -> SourceLayout:
        return self.program.source_layout

    @property
    def source_modality(self) -> Modality | None:
        """Mono source modality; None for NONE or TEXT_AUDIO layouts."""
        return self.source_layout.as_modality()

    @property
    def prediction_modality(self) -> PredictionModality:
        """Default prediction when the loader does not override ``prediction``.

        Training consumers must use ``ModelSample.prediction`` /
        ``ModelBatch.prediction_modality`` (resolved via
        ``resolve_prediction``), not this default.
        """
        return self.program.default.prediction

    @property
    def allowed_predictions(self) -> frozenset[PredictionModality]:
        return self.program.allowed_predictions

    @property
    def target_modality(self) -> Modality | None:
        """Mono item/decode modality; None when prediction is mixed by default."""
        prediction = self.prediction_modality
        if prediction is PredictionModality.TEXT:
            return Modality.TEXT
        if prediction is PredictionModality.AUDIO:
            return Modality.AUDIO
        return None

    @property
    def execution_signature(self) -> tuple[SourceLayout, PredictionModality]:
        """Default execution signature without a loader prediction override."""
        return (self.source_layout, self.prediction_modality)

    @property
    def uses_source_role(self) -> bool:
        return any(field.role is FieldRole.SOURCE for field in self.program.context)

    @property
    def templates(self) -> tuple[str, ...]:
        from .templates import TEMPLATES

        return TEMPLATES[self]

    def sample_template(self, index: Optional[int] = 0) -> str:
        from .templates import select_template

        return select_template(self, index)


def resolve_prediction(
    task: Task,
    override: PredictionModality | None = None,
) -> PredictionModality:
    """Resolve effective prediction modality for a task.

    ``override`` must be in ``task.allowed_predictions`` when provided.
    """
    return resolve_response(task, prediction=override).prediction


def resolve_response(
    task: Task,
    *,
    prediction: PredictionModality | None = None,
    trace: str | None = None,
) -> ResponseSpec:
    """Resolve one concrete response trace for a task invocation."""
    if not isinstance(task, Task):
        raise TypeError("response resolution requires a Task.")
    if prediction is not None and not isinstance(prediction, PredictionModality):
        raise TypeError("response prediction must be a PredictionModality or None.")
    try:
        return task.program.response(trace, prediction=prediction)
    except ValueError as error:
        raise ValueError(f"{task.value} {error}") from error


def execution_signature(
    task: Task,
    *,
    prediction: PredictionModality | None = None,
) -> tuple[object, PredictionModality]:
    return (task.source_layout, resolve_prediction(task, prediction))


def uses_source_ctc(task: Task) -> bool:
    """Whether the source audio transcript is latent to its hidden states."""
    if not isinstance(task, Task):
        raise TypeError("source CTC routing requires a Task.")
    # TEXT_AUDIO routes already expose the paired text and therefore do not
    # provide a clean audio-to-frozen-text alignment target.
    return task.source_layout is SourceLayout.AUDIO


def uses_target_ctc(
    task: Task,
    prediction: PredictionModality | None = None,
    *,
    trace: str | None = None,
) -> bool:
    """Whether a causal audio response lacks its own transcript as context.

    TTS is the deliberate counterexample: it predicts audio, but its target
    transcript is already the visible source. Mixed text/audio responses also
    expose target text before or alongside audio and are excluded.
    """
    response = resolve_response(task, prediction=prediction, trace=trace)
    if response.prediction is not PredictionModality.AUDIO:
        return False
    target_text = TaskField(FieldRole.TARGET, Modality.TEXT)
    return target_text not in task.program.context and target_text not in response.fields


__all__ = [
    "FieldRole",
    "PredictionModality",
    "Request",
    "ResponseLayout",
    "ResponseSpec",
    "SourceLayout",
    "Task",
    "TaskField",
    "TaskObjective",
    "TaskProgram",
    "execution_signature",
    "resolve_prediction",
    "resolve_response",
    "uses_source_ctc",
    "uses_target_ctc",
]
