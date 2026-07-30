from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import torch
from torch import Tensor

from ..._tensor import is_signed_integer_dtype


def codebook_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("codebook sizes must be integers.")
    size = int(value)
    if size < 1:
        raise ValueError("codebook sizes must be positive.")
    return size


def frame_tensor(
    frames: Sequence[Sequence[int]] | Tensor,
    codebook_sizes: Sequence[int],
) -> Tensor:
    if isinstance(frames, Tensor):
        validate_ids(frames, "frames")
        tensor = frames.to(dtype=torch.long)
    else:
        tensor = torch.tensor(frames, dtype=torch.long)
    if tensor.dim() != 2 or tensor.size(1) != len(codebook_sizes):
        raise ValueError("frames must have shape [frames, codebooks].")
    return tensor


def token_tensor(token_ids: Sequence[int] | Tensor) -> Tensor:
    if isinstance(token_ids, Tensor):
        validate_ids(token_ids, "token ids")
        tensor = token_ids.to(dtype=torch.long)
    else:
        tensor = torch.tensor(token_ids, dtype=torch.long)
    if tensor.dim() != 1:
        raise ValueError("token id tensor must have shape [tokens].")
    return tensor


def validate_frame_ranges(frames: Tensor, codebook_sizes: Sequence[int]) -> None:
    for index, size in enumerate(codebook_sizes):
        validate_range(frames[:, index], f"codebook {index} frames", size)


def validate_ids(ids: Tensor, name: str) -> None:
    if not is_signed_integer_dtype(ids.dtype):
        raise TypeError(f"{name} must contain integer ids using a signed dtype.")


def validate_range(ids: Tensor, name: str, vocab_size: int) -> None:
    if bool(((ids < 0) | (ids >= vocab_size)).any()):
        raise ValueError(f"{name} must contain ids in [0, {vocab_size}).")
