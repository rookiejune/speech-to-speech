from __future__ import annotations

import torch
from torch import nn


def register(
    module: nn.Module,
    name: str,
    tensor: torch.Tensor,
    *,
    persistent: bool = True,
) -> None:
    """Register a buffer in both the Stable Codec Torch 2.4 env and newer Torch."""
    buffer_type = getattr(nn, "Buffer", None)
    if buffer_type is None:
        module.register_buffer(name, tensor, persistent=persistent)
        return
    setattr(module, name, buffer_type(tensor, persistent=persistent))
