"""Shared task, modality, and routing contracts."""

from .contract import (
    FieldRole,
    PredictionModality,
    Request,
    ResponseLayout,
    ResponseSpec,
    SourceLayout,
    Task,
    TaskField,
    TaskObjective,
    TaskProgram,
    execution_signature,
    resolve_prediction,
    resolve_response,
    uses_source_ctc,
    uses_target_ctc,
)
from .program import DIRECT, FULL_COT, PROGRAMS, TARGET_COT, program_for

__all__ = [
    "DIRECT",
    "FULL_COT",
    "FieldRole",
    "PROGRAMS",
    "PredictionModality",
    "Request",
    "ResponseLayout",
    "ResponseSpec",
    "SourceLayout",
    "TARGET_COT",
    "Task",
    "TaskField",
    "TaskObjective",
    "TaskProgram",
    "execution_signature",
    "program_for",
    "resolve_prediction",
    "resolve_response",
    "uses_source_ctc",
    "uses_target_ctc",
]
