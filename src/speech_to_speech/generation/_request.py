from __future__ import annotations

import torch
from anydataset.types import Modality
from torch import Tensor

from .._tensor import is_signed_integer_dtype
from ..prediction import PredictionModality
from ..task import Task
from .audio import validate_audio_request
from .protocol import TokenGenerator
from .types import Request


def validate(request: Request, model: TokenGenerator) -> None:
    task = request["task"]
    if not isinstance(task, Task):
        raise TypeError("generation request task must be a Task.")
    prompt = _integer_tensor(request["prompt_ids"], "prompt ids", dimensions=1)
    if prompt.numel() == 0:
        raise ValueError("generation prompt must contain at least one token.")
    inside = torch.zeros_like(prompt, dtype=torch.bool)
    for start, end in model.runtime.layout.blocks.values():
        inside |= prompt.ge(start) & prompt.lt(end)
    if not bool(inside.all()):
        raise ValueError("prompt ids must belong to the runtime layout.")
    positions_value = request.get("audio_input_positions")
    if positions_value is not None:
        positions = _integer_tensor(
            positions_value,
            "audio input positions",
            dimensions=1,
        )
        if task.source_modality is not Modality.AUDIO:
            raise ValueError(
                "audio input positions require an audio-source generation task."
            )
        if bool((positions < 0).any()) or bool(
            (positions >= prompt.numel()).any()
        ):
            raise ValueError("audio input positions must be valid prompt positions.")
        if positions.numel() != torch.unique(positions).numel():
            raise ValueError("audio input positions must not repeat positions.")
        codec_start, codec_end = model.runtime.codec_audio_range
        selected = prompt.index_select(0, positions.to(device=prompt.device))
        if bool((selected < codec_start).any()) or bool(
            (selected >= codec_end).any()
        ):
            raise ValueError(
                "audio input positions must point to visible codec audio payload tokens."
            )
    if task.prediction_modality is PredictionModality.TEXT:
        if request.get("audio_context") is not None:
            raise ValueError("text generation requests cannot include audio context.")
        return
    if task.prediction_modality.is_mixed:
        if request.get("audio_context") is not None:
            raise ValueError("mixed AR generation requests cannot include audio context.")
        return
    validate_audio_request(request, model)


def _integer_tensor(value: object, name: str, *, dimensions: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if not is_signed_integer_dtype(value.dtype):
        raise TypeError(f"{name} must contain integer ids using a signed dtype.")
    if value.dim() != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions.")
    return value
