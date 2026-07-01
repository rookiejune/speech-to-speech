from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .module import SpeechToSpeechDataModule

__all__ = ["SpeechToSpeechDataModule"]


def __getattr__(name: str) -> object:
    if name == "SpeechToSpeechDataModule":
        from .module import SpeechToSpeechDataModule

        return SpeechToSpeechDataModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}.")
