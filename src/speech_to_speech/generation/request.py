from __future__ import annotations

import torch
from torch import Tensor

from .._tensor import is_signed_integer_dtype
from ..runtime.audio_tokenizer import BiCodecAudioTokenizer
from ..task import (
    PredictionModality,
    Request,
    ResponseSpec,
    Task,
    normalize_language_code,
    resolve_response,
)
from .audio import has_semantic_decode_options, validate_audio_request
from .contract import TokenGenerator


def response_of(request: Request) -> ResponseSpec:
    if "prediction" in request:
        raise ValueError(
            "generation request prediction override is not supported; "
            "select a task response with trace."
        )
    trace = request.get("trace")
    if trace is not None and (not isinstance(trace, str) or not trace):
        raise TypeError("generation request trace must be a non-empty string.")
    return resolve_response(request["task"], trace=trace)


def target_language_of(
    request: Request,
    *,
    response: ResponseSpec | None = None,
) -> str | None:
    """Return the normalized language that selects MT response controls."""
    response = response_of(request) if response is None else response
    if not response.requires_target_language:
        return None
    value = request.get("target_language")
    if value is None:
        raise ValueError(
            "generation request target_language is required for MT response steps."
        )
    return normalize_language_code(value)


def validate(request: Request, model: TokenGenerator) -> None:
    if "audio_context" in request:
        raise ValueError(
            "generation request audio_context is not supported; place source audio "
            "in a task prompt such as TTS_VOICE_CLONE."
        )
    task = request["task"]
    if not isinstance(task, Task):
        raise TypeError("generation request task must be a Task.")
    response = response_of(request)
    prediction = response.prediction
    target_language_of(request, response=response)
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
        if not task.source_layout.includes_audio:
            raise ValueError(
                "audio input positions require an audio-source generation task."
            )
        if bool((positions < 0).any()) or bool(
            (positions >= prompt.numel()).any()
        ):
            raise ValueError("audio input positions must be valid prompt positions.")
        if positions.numel() != torch.unique(positions).numel():
            raise ValueError("audio input positions must not repeat positions.")
        codec_start, codec_end = getattr(
            model.runtime,
            "input_codec_audio_range",
            model.runtime.codec_audio_range,
        )
        selected = prompt.index_select(0, positions.to(device=prompt.device))
        if bool((selected < codec_start).any()) or bool(
            (selected >= codec_end).any()
        ):
            raise ValueError(
                "audio input positions must point to visible codec audio payload tokens."
            )
    if prediction is PredictionModality.TEXT:
        if has_semantic_decode_options(request):
            raise ValueError(
                "text generation requests cannot include semantic decode options."
            )
        return
    if prediction.is_mixed:
        if has_semantic_decode_options(request):
            raise ValueError(
                "mixed AR generation requests cannot include semantic decode options."
            )
        if (
            prediction is PredictionModality.INTERLEAVED
            and model.runtime.structured_full_sequence
            and isinstance(model.runtime.audio_tokenizer, BiCodecAudioTokenizer)
        ):
            raise ValueError(
                "INTERLEAVED generation does not support structured BiCodec audio sequences."
            )
        return
    validate_audio_request(request, model, prediction=prediction)


def _integer_tensor(value: object, name: str, *, dimensions: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if not is_signed_integer_dtype(value.dtype):
        raise TypeError(f"{name} must contain integer ids using a signed dtype.")
    if value.dim() != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions.")
    return value
