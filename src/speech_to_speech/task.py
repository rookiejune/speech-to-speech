from __future__ import annotations

from typing import Optional

from anydataset.types import Modality

from ._compat import StrEnum, auto
from .prediction import PredictionModality
from .source import SourceLayout


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
        ``task_spec.resolve_prediction``), not this default.
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


__all__ = ["Task"]
