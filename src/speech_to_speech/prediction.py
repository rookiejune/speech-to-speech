from __future__ import annotations

from anydataset.types import Modality

from ._compat import StrEnum, auto


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


__all__ = ["PredictionModality"]
