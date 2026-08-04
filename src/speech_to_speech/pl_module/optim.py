"""Optimizer configuration owned by Lightning training modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    name: str = "adamw"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01


__all__ = ["Config"]
