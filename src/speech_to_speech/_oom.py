from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict, cast

import torch
from torch import Tensor


class Context(TypedDict):
    phase: str
    inputs: dict[str, object]


_CONTEXT_KEY = "speech_to_speech_oom_context"


def annotate(
    exception: BaseException,
    *,
    phase: str,
    inputs: Mapping[str, object],
) -> bool:
    """Attach JSON-safe input metadata while preserving the original OOM."""

    if not is_oom(exception):
        return False
    exception.__dict__[_CONTEXT_KEY] = Context(
        phase=phase,
        inputs=dict(inputs),
    )
    return True


def context(exception: BaseException) -> Context | None:
    value = exception.__dict__.get(_CONTEXT_KEY)
    if value is None:
        return None
    return cast(Context, value)


def is_oom(exception: BaseException) -> bool:
    return isinstance(exception, torch.OutOfMemoryError)


def tensor_report(value: Tensor | None) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


__all__ = ["Context", "annotate", "context", "is_oom", "tensor_report"]
