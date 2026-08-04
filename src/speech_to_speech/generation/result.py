from __future__ import annotations

from typing import TypedDict

from torch import Tensor
from typing_extensions import NotRequired

from ..audio import AudioCodes


class AudioOutput(TypedDict):
    features: Tensor | None
    codes: AudioCodes | None
    waveform: Tensor
    sample_rate: int


class Result(TypedDict):
    response_ids: Tensor
    audio: AudioOutput | None
    decode_error: NotRequired[dict[str, str]]


__all__ = ["AudioOutput", "Result"]
