from __future__ import annotations

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


class Request(TypedDict):
    """Task-level tensor request shared by data and generation services."""

    prompt_ids: Tensor
    task: Task
    audio_input_positions: Tensor | None
    prediction: NotRequired[PredictionModality | None]
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
    def source_layout(self) -> SourceLayout:
        if self is Task.MASKED_AR:
            return SourceLayout.TEXT_AUDIO
        if self in {
            Task.AUDIO_AR,
            Task.INTERLEAVED_AR,
            Task.PARALLEL_AR,
            Task.TEXT_AR,
        }:
            return SourceLayout.NONE
        if self in {Task.ASR, Task.S2ST, Task.S2TT}:
            return SourceLayout.AUDIO
        return SourceLayout.TEXT

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
        if self in {Task.ASR, Task.MT, Task.S2TT, Task.TEXT_AR, Task.T2TT}:
            return PredictionModality.TEXT
        if self in {Task.PARALLEL_AR, Task.MASKED_AR}:
            return PredictionModality.PARALLEL
        if self is Task.INTERLEAVED_AR:
            return PredictionModality.INTERLEAVED
        return PredictionModality.AUDIO

    @property
    def allowed_predictions(self) -> frozenset[PredictionModality]:
        if self in {Task.T2ST, Task.S2ST}:
            return frozenset(
                {PredictionModality.AUDIO, PredictionModality.PARALLEL}
            )
        if self is Task.MASKED_AR:
            return frozenset(
                {PredictionModality.PARALLEL, PredictionModality.INTERLEAVED}
            )
        return frozenset({self.prediction_modality})

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
        return self in {Task.MT, Task.S2ST, Task.S2TT, Task.T2ST, Task.T2TT}

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
    if override is None:
        return task.prediction_modality
    if override not in task.allowed_predictions:
        allowed = ", ".join(sorted(value.value for value in task.allowed_predictions))
        raise ValueError(
            f"{task.value} does not allow prediction={override.value}; "
            f"allowed: {allowed}."
        )
    return override


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
) -> bool:
    """Whether a causal audio response lacks its own transcript as context.

    TTS is the deliberate counterexample: it predicts audio, but its target
    transcript is already the visible source. Mixed text/audio responses also
    expose target text before or alongside audio and are excluded.
    """
    resolved = resolve_prediction(task, prediction)
    return resolved is PredictionModality.AUDIO and task is not Task.TTS


__all__ = [
    "PredictionModality",
    "Request",
    "SourceLayout",
    "Task",
    "execution_signature",
    "resolve_prediction",
    "uses_source_ctc",
    "uses_target_ctc",
]
