from __future__ import annotations

from ..datamodule.types import ModelBatch
from .types import Request


def requests_from_batch(batch: ModelBatch) -> list[Request]:
    """Build unpadded inference requests from teacher-forcing samples."""
    requests: list[Request] = []
    for index, task in enumerate(batch.tasks):
        target_positions = (batch.token_labels[index] != -100).nonzero()
        if target_positions.numel() == 0:
            raise ValueError("teacher-forcing batch row has no target tokens.")
        prompt_end = int(target_positions[0].item())

        requests.append(
            Request(
                prompt_ids=batch.input_ids[index, :prompt_end],
                task=task,
            )
        )
    return requests
