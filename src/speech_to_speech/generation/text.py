from __future__ import annotations

from functools import cached_property
from typing import Protocol

from anydataset.types import Modality
from anytrain.module.idspace import Layout
from torch import Tensor

from .._tensor import is_signed_integer_dtype
from ..task import PredictionModality
from ..runtime.tokenizer import TextTokenizer


class ResponseTextRuntime(Protocol):
    @cached_property
    def layout(self) -> Layout: ...

    @cached_property
    def text_tokenizer(self) -> TextTokenizer: ...


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
    """Decode global IDs that are already known to belong to the text block."""
    projected = response_text_ids(
        runtime,
        token_ids,
        prediction=PredictionModality.TEXT,
    )
    if projected is None:
        raise AssertionError("text projection unexpectedly returned no token sequence.")
    start, _ = runtime.layout.block(Modality.TEXT.value)
    local_ids = (projected - start).detach().cpu().tolist()
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


__all__ = [
    "ResponseTextRuntime",
    "decode_response_text",
    "decode_text_ids",
    "response_text_ids",
]
