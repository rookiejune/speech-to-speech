"""Outputs produced by model-side generation capabilities."""

from __future__ import annotations

from typing import TypedDict

from torch import Tensor


class AcousticGeneration(TypedDict):
    sequence: Tensor
    features: Tensor
    frame_counts: Tensor


__all__ = ["AcousticGeneration"]
