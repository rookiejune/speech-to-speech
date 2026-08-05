from __future__ import annotations

from functools import cached_property
from typing import Protocol, runtime_checkable

from anydataset.types import Modality
from anytrain.module.idspace import Layout
from torch import Tensor

from .._tensor import is_signed_integer_dtype
from ..runtime.backbone.contract import TextTokenizer
from ..task import (
    ControlToken,
    PredictionModality,
    ResponseControl,
    ResponseSpec,
    response_control_tokens,
)


class ResponseTextRuntime(Protocol):
    @cached_property
    def layout(self) -> Layout: ...

    @cached_property
    def text_tokenizer(self) -> TextTokenizer: ...

    @cached_property
    def lexical_text_vocab_size(self) -> int: ...

    @cached_property
    def control_token_ids(self) -> tuple[int, ...]: ...


@runtime_checkable
class ResponseStepRuntime(ResponseTextRuntime, Protocol):
    @cached_property
    def eos_token_id(self) -> int: ...

    @cached_property
    def boa_token_id(self) -> int: ...

    @cached_property
    def eoa_token_id(self) -> int: ...

    def control_token_id(self, token: ControlToken) -> int: ...


def response_text_ids(
    runtime: ResponseTextRuntime,
    response_ids: Tensor,
    *,
    prediction: PredictionModality,
) -> Tensor | None:
    """Project one generated response onto its global text token IDs."""
    if not isinstance(prediction, PredictionModality):
        raise TypeError("response prediction must be a PredictionModality.")
    if not isinstance(response_ids, Tensor):
        raise TypeError("response ids must be a Tensor.")
    if not is_signed_integer_dtype(response_ids.dtype):
        raise TypeError("response ids must contain integer ids using a signed dtype.")
    if response_ids.dim() != 1:
        raise ValueError("response ids must have shape [tokens].")
    if not prediction.supervises_text:
        return None

    start, end = runtime.layout.block(Modality.TEXT.value)
    inside = response_ids.ge(start) & response_ids.lt(end)
    if prediction is PredictionModality.TEXT:
        if not bool(inside.all()):
            raise ValueError("text response ids must belong to the runtime text block.")
        return response_ids
    return response_ids[inside]


def decode_text_ids(runtime: ResponseTextRuntime, token_ids: Tensor) -> str:
    """Decode lexical global IDs without exposing runtime controls to HF."""
    projected = response_text_ids(
        runtime,
        token_ids,
        prediction=PredictionModality.TEXT,
    )
    if projected is None:
        raise AssertionError("text projection unexpectedly returned no token sequence.")
    start, _ = runtime.layout.block(Modality.TEXT.value)
    lexical_end = start + runtime.lexical_text_vocab_size
    lexical = projected[projected.lt(lexical_end)]
    local_ids = (lexical - start).detach().cpu().tolist()
    return runtime.text_tokenizer.decode(local_ids, skip_special_tokens=True)


def decode_response_text(
    runtime: ResponseTextRuntime,
    response_ids: Tensor,
    *,
    prediction: PredictionModality,
) -> str | None:
    """Decode the text projection of a text or mixed generated response."""
    token_ids = response_text_ids(runtime, response_ids, prediction=prediction)
    if token_ids is None:
        return None
    return decode_text_ids(runtime, token_ids)


def decode_response_text_steps(
    runtime: ResponseStepRuntime,
    response_ids: Tensor,
    response: ResponseSpec,
    *,
    target_language: str | None = None,
) -> list[str | None]:
    """Decode ordered logical text fields without merging CoT stages."""
    if not isinstance(response, ResponseSpec):
        raise TypeError("response step decoding requires a ResponseSpec.")
    text_fields = [
        index
        for index, field in enumerate(response.fields)
        if field.modality is Modality.TEXT
    ]
    if not text_fields:
        return [None for _ in response.steps]
    if response.prediction is PredictionModality.INTERLEAVED:
        combined = decode_response_text(
            runtime,
            response_ids,
            prediction=response.prediction,
        )
        return [
            combined if index in text_fields else None
            for index, _ in enumerate(response.fields)
        ]
    if not isinstance(response_ids, Tensor) or response_ids.dim() != 1:
        raise ValueError("response step decoding requires a 1D Tensor.")
    cursor = 0
    values: list[str | None] = []
    for step in response.steps:
        if step.modality is Modality.TEXT:
            prefix_ids, end_id = _text_boundary_ids(
                runtime,
                step.control,
                target_language=target_language,
            )
            for prefix_id in prefix_ids:
                if cursor >= response_ids.numel() or int(
                    response_ids[cursor].item()
                ) != prefix_id:
                    raise ValueError(
                        "generated text response is missing its expected control prefix."
                    )
                cursor += 1
            stop = _next(response_ids, end_id, cursor)
            end = response_ids.numel() if stop is None else stop
            text_ids = response_ids[cursor:end]
            values.append(decode_text_ids(runtime, text_ids))
            cursor = end if stop is None else end + 1
            continue
        start = _next(response_ids, runtime.boa_token_id, cursor)
        if start is None:
            values.append(None)
            cursor = response_ids.numel()
            continue
        stop = _next(response_ids, runtime.eoa_token_id, start + 1)
        cursor = response_ids.numel() if stop is None else stop + 1
        values.append(None)
    return values


def _text_boundary_ids(
    runtime: ResponseStepRuntime,
    control: ResponseControl,
    *,
    target_language: str | None,
) -> tuple[tuple[int, ...], int]:
    controls = response_control_tokens(
        control,
        target_language=target_language,
    )
    if controls is not None:
        return (
            tuple(runtime.control_token_id(token) for token in controls.prefix),
            runtime.control_token_id(controls.end),
        )
    if control is ResponseControl.EOS:
        return (), runtime.eos_token_id
    raise ValueError("text response steps require EOS, ASR, or MT controls.")


def _next(token_ids: Tensor, token_id: int, start: int) -> int | None:
    positions = token_ids[start:].eq(token_id).nonzero(as_tuple=False)
    if positions.numel() == 0:
        return None
    return start + int(positions[0].item())


__all__ = [
    "ResponseTextRuntime",
    "ResponseStepRuntime",
    "decode_response_text",
    "decode_response_text_steps",
    "decode_text_ids",
    "response_text_ids",
]
