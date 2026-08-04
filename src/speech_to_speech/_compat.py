from __future__ import annotations

import sys
from enum import Enum, auto

import torch
from torch import nn

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        @staticmethod
        def _generate_next_value_(
            name: str,
            start: int,
            count: int,
            last_values: list[str],
        ) -> str:
            return name.lower()

        def __str__(self) -> str:
            return self.value


def register(
    module: nn.Module,
    name: str,
    tensor: torch.Tensor,
    *,
    persistent: bool = True,
) -> None:
    """Register a buffer in both the Torch 2.4 codec env and newer Torch."""
    buffer_type = getattr(nn, "Buffer", None)
    if buffer_type is None:
        module.register_buffer(name, tensor, persistent=persistent)
        return
    setattr(module, name, buffer_type(tensor, persistent=persistent))


__all__ = ["StrEnum", "auto", "register"]
