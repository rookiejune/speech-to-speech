from __future__ import annotations

from ..datamodule.types import ModelBatch
from .types import Request


def requests_from_batch(batch: ModelBatch) -> list[Request]:
    """Build unpadded inference requests from teacher-forcing samples."""
    requests: list[Request] = []
    prompt_lengths = batch.generation_prompt_lengths
    audio_contexts = batch.audio_contexts
    if prompt_lengths is None or audio_contexts is None:
        raise RuntimeError("model batch generation fields are unavailable.")
    for index, task in enumerate(batch.tasks):
        prompt_end = int(prompt_lengths[index].item())

        requests.append(
            Request(
                prompt_ids=batch.input_ids[index, :prompt_end],
                task=task,
                audio_context=audio_contexts[index],
            )
        )
    return requests
