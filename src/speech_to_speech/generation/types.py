from __future__ import annotations

from typing import TypedDict

from torch import Tensor

from ..task import Task


class AcousticPrompt(TypedDict):
    codes: Tensor
    token_positions: Tensor


class Request(TypedDict):
    prompt_ids: Tensor
    task: Task
    acoustic_prompt: AcousticPrompt | None


class AudioOutput(TypedDict):
    features: Tensor | None
    waveform: Tensor
    sample_rate: int


class AcousticGeneration(TypedDict):
    sequence: Tensor
    features: Tensor
    frame_counts: Tensor


class Result(TypedDict):
    response_ids: Tensor
    audio: AudioOutput | None
