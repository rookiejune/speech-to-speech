"""Task-level tensor request shared by training and generation."""

from __future__ import annotations

from typing import TypedDict

from torch import Generator, Tensor
from typing_extensions import NotRequired

from .contract import PredictionModality, Task


class Request(TypedDict):
    prompt_ids: Tensor
    task: Task
    audio_input_positions: Tensor | None
    prediction: NotRequired[PredictionModality | None]
    semantic_reference_features: NotRequired[Tensor | None]
    semantic_reference_mask: NotRequired[Tensor | None]
    semantic_decode_generator: NotRequired[Generator | None]


__all__ = ["Request"]
