from __future__ import annotations

from functools import partial
from typing import Any, cast

from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer


def enable_gradient_checkpointing(module: nn.Module) -> int:
    """Enable HF-style activation checkpointing on custom model layers."""
    checkpointing = partial(checkpoint, use_reentrant=False)
    count = 0
    for layer in module.modules():
        if not isinstance(layer, GradientCheckpointingLayer):
            continue
        checkpoint_layer = cast(Any, layer)
        checkpoint_layer._gradient_checkpointing_func = checkpointing
        checkpoint_layer.gradient_checkpointing = True
        count += 1
    return count


__all__ = ["GradientCheckpointingLayer", "enable_gradient_checkpointing"]
