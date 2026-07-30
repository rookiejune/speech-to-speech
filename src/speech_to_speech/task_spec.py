from __future__ import annotations

from .prediction import PredictionModality
from .task import Task


def resolve_prediction(
    task: Task,
    override: PredictionModality | None = None,
) -> PredictionModality:
    """Resolve effective prediction modality for a task.

    ``override`` must be in ``task.allowed_predictions`` when provided.
    """
    if override is None:
        return task.prediction_modality
    if override not in task.allowed_predictions:
        allowed = ", ".join(sorted(value.value for value in task.allowed_predictions))
        raise ValueError(
            f"{task.value} does not allow prediction={override.value}; "
            f"allowed: {allowed}."
        )
    return override


def execution_signature(
    task: Task,
    *,
    prediction: PredictionModality | None = None,
) -> tuple[object, PredictionModality]:
    return (task.source_layout, resolve_prediction(task, prediction))


__all__ = ["execution_signature", "resolve_prediction"]
