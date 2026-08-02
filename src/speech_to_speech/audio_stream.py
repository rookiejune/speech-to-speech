from __future__ import annotations

from ._compat import StrEnum, auto


class AudioStream(StrEnum):
    """Non-text AR audio streams."""

    ACOUSTIC = auto()
    SEMANTIC = auto()
