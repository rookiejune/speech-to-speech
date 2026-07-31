"""Internal loss package."""

from .preference import DPOObjective
from .rollout import GRPOObjective

__all__ = ["DPOObjective", "GRPOObjective"]
