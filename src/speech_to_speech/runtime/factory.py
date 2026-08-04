from __future__ import annotations

from .config import AudioSequenceLayout, Config
from .core import Runtime


def runtime_for_sequence_layout(config: Config, layout: AudioSequenceLayout) -> Runtime:
    return Runtime(
        config,
        audio_sequence_layout=layout,
    )

__all__ = ["runtime_for_sequence_layout"]
