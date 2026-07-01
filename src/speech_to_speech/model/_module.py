"""Shared module inspection helpers for model runtime code."""

from __future__ import annotations

import torch
from torch import nn


def module_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    for parameter in module.parameters():
        return parameter.dtype
    for buffer in module.buffers():
        return buffer.dtype
    return fallback
