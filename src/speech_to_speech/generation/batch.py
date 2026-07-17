from __future__ import annotations

from ..datamodule.types import ModelBatch
from .types import AcousticPrompt, Request


def requests_from_batch(batch: ModelBatch) -> list[Request]:
    """Build unpadded inference requests from teacher-forcing samples."""
    requests: list[Request] = []
    acoustic_mask = batch.acoustic_prompt_mask
    for index, task in enumerate(batch.tasks):
        target_positions = (batch.token_labels[index] != -100).nonzero()
        if target_positions.numel() == 0:
            raise ValueError("teacher-forcing batch row has no target tokens.")
        prompt_end = int(target_positions[0].item())

        acoustic_prompt = None
        if batch.acoustic_prompt is not None:
            if acoustic_mask is None:
                raise RuntimeError("acoustic prompt mask is unavailable.")
            row_mask = acoustic_mask[index]
            acoustic_prompt = AcousticPrompt(
                codes=batch.acoustic_prompt["codes"][index][row_mask],
                token_positions=batch.acoustic_prompt["token_positions"][index][row_mask],
            )

        requests.append(
            Request(
                prompt_ids=batch.input_ids[index, :prompt_end],
                task=task,
                acoustic_prompt=acoustic_prompt,
            )
        )
    return requests
