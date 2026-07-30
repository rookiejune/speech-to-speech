from __future__ import annotations

from anydataset.types import Modality

from ._compat import StrEnum, auto


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


__all__ = ["SourceLayout"]
