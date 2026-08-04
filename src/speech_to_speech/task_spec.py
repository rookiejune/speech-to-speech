from __future__ import annotations

from .prediction import PredictionModality
from .source import SourceLayout
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


def uses_source_ctc(task: Task) -> bool:
    """Whether the source audio transcript is latent to its hidden states."""
    if not isinstance(task, Task):
        raise TypeError("source CTC routing requires a Task.")
    # TEXT_AUDIO routes already expose the paired text and therefore do not
    # provide a clean audio-to-frozen-text alignment target.
    return task.source_layout is SourceLayout.AUDIO


def uses_target_ctc(
    task: Task,
    prediction: PredictionModality | None = None,
) -> bool:
    """Whether a causal audio response lacks its own transcript as context.

    TTS is the deliberate counterexample: it predicts audio, but its target
    transcript is already the visible source. Mixed text/audio responses also
    expose target text before or alongside audio and are excluded.
    """
    resolved = resolve_prediction(task, prediction)
    return resolved is PredictionModality.AUDIO and task is not Task.TTS


__all__ = [
    "execution_signature",
    "resolve_prediction",
    "uses_source_ctc",
    "uses_target_ctc",
]
