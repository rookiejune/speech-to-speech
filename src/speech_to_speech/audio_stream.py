from __future__ import annotations

from ._compat import StrEnum, auto


class AudioStream(StrEnum):
    """Non-text AR audio streams used by structured codecs."""

    GLOBAL = auto()
    SEMANTIC = auto()
