from __future__ import annotations

from ._compat import StrEnum, auto


class LoaderStepMode(StrEnum):
    WEIGHTED_WINDOW = auto()
    FUSED_JOINT = auto()
    SERIAL_JOINT = auto()


__all__ = ["LoaderStepMode"]
