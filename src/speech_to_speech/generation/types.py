from __future__ import annotations

from typing import TypedDict

from torch import Generator, Tensor
from typing_extensions import NotRequired

from ..prediction import PredictionModality
from ..codes import AudioCodes
from ..task import Task


class Request(TypedDict):
    prompt_ids: Tensor
    task: Task
    audio_input_positions: Tensor | None
    prediction: NotRequired[PredictionModality | None]
    semantic_reference_features: NotRequired[Tensor | None]
    semantic_reference_mask: NotRequired[Tensor | None]
    semantic_decode_generator: NotRequired[Generator | None]


class AudioOutput(TypedDict):
    features: Tensor | None
    codes: AudioCodes | None
    waveform: Tensor
    sample_rate: int


class AcousticGeneration(TypedDict):
    sequence: Tensor
    features: Tensor
    frame_counts: Tensor


class Result(TypedDict):
    response_ids: Tensor
    audio: AudioOutput | None
    decode_error: NotRequired[dict[str, str]]
