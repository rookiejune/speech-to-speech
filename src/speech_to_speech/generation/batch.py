from __future__ import annotations

from ..datamodule.batch import ModelBatch
from ..task import Request


def requests_from_batch(batch: ModelBatch) -> list[Request]:
    """Build unpadded inference requests from teacher-forcing samples."""
    requests: list[Request] = []
    prompt_lengths = batch.generation_prompt_lengths
    audio_input_positions = batch.audio_input_positions
    if prompt_lengths is None:
        raise RuntimeError("model batch generation fields are unavailable.")
    for index, (task, prediction) in enumerate(zip(batch.tasks, batch.predictions)):
        prompt_end = int(prompt_lengths[index].item())
        requests.append(
            Request(
                prompt_ids=batch.input_ids[index, :prompt_end],
                task=task,
                prediction=prediction,
                audio_input_positions=(
                    None
                    if audio_input_positions is None
                    else audio_input_positions[index][
                        audio_input_positions[index].ge(0)
                    ]
                ),
            )
        )
    return requests
