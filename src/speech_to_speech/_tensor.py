from __future__ import annotations

import torch


_SIGNED_INTEGER_DTYPES: frozenset[torch.dtype] = frozenset(
    (torch.int8, torch.int16, torch.int32, torch.int64)
)


def is_signed_integer_dtype(dtype: torch.dtype) -> bool:
    return dtype in _SIGNED_INTEGER_DTYPES


__all__ = ["is_signed_integer_dtype"]
