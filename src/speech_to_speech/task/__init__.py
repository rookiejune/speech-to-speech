"""Shared task, modality, and routing contracts."""

from .contract import (
    PredictionModality,
    SourceLayout,
    Task,
    execution_signature,
    resolve_prediction,
    uses_source_ctc,
    uses_target_ctc,
)

__all__ = [
    "PredictionModality",
    "SourceLayout",
    "Task",
    "execution_signature",
    "resolve_prediction",
    "uses_source_ctc",
    "uses_target_ctc",
]
