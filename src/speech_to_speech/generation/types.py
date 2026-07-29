from __future__ import annotations

from typing import TypedDict

from anytrain.codec import SemanticAcousticCodes
from torch import Tensor

from ..task import Task


class Request(TypedDict):
    prompt_ids: Tensor
    task: Task
    audio_input_positions: Tensor | None
    audio_context: SemanticAcousticCodes | None


class AudioOutput(TypedDict):
    features: Tensor | None
    codes: SemanticAcousticCodes | None
    waveform: Tensor
    sample_rate: int


class AcousticGeneration(TypedDict):
    sequence: Tensor
    features: Tensor
    frame_counts: Tensor


class Result(TypedDict):
    response_ids: Tensor
    audio: AudioOutput | None
