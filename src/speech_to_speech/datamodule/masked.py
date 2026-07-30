from __future__ import annotations

import torch
from anydataset.types import Modality
from torch import Generator, Tensor

from ..prediction import PredictionModality
from ..task import Task
from ..task_spec import resolve_prediction
from ._tokenization import token_ids
from .ar import pack_interleaved, pack_parallel
from .protocol import DataRuntime
from .types import ModelSample, Speech


def build_masked_sample(
    speech: Speech,
    task: Task,
    runtime: DataRuntime,
    *,
    prompt: str,
    prediction: PredictionModality | None = None,
    interleave_audio_frames: int = 25,
    mask_text_ratio: float = 0.5,
    mask_audio_ratio: float = 0.5,
    generator: Generator | None = None,
) -> ModelSample:
    if task is not Task.MASKED_AR:
        raise ValueError(f"{task.value} is not a masked autoregressive task.")
    prediction = resolve_prediction(task, prediction)
    if not prediction.is_mixed:
        raise ValueError("MASKED_AR requires PARALLEL or INTERLEAVED prediction.")
    _validate_ratio(mask_text_ratio, name="mask_text_ratio")
    _validate_ratio(mask_audio_ratio, name="mask_audio_ratio")
    if not hasattr(runtime, "mask_token_id"):
        raise AttributeError("MASKED_AR requires runtime.mask_token_id.")

    marker = token_ids(prompt, runtime.text_tokenizer)
    masked_source = _masked_source(
        speech,
        runtime,
        mask_text_ratio=mask_text_ratio,
        mask_audio_ratio=mask_audio_ratio,
        generator=generator,
    )
    prefix = torch.cat([marker, masked_source])
    if prediction is PredictionModality.PARALLEL:
        sample = pack_parallel(
            prefix,
            speech,
            task,
            runtime,
            prediction=prediction,
        )
    else:
        sample = pack_interleaved(
            prefix,
            speech,
            task,
            runtime,
            prediction=prediction,
            interleave_audio_frames=interleave_audio_frames,
        )
    return ModelSample(
        input_ids=sample.input_ids,
        token_labels=sample.token_labels,
        token_groups=sample.token_groups,
        acoustic_target=sample.acoustic_target,
        task=sample.task,
        prediction=sample.prediction,
        audio_seconds=sample.audio_seconds,
        generation_prompt_length=prefix.numel(),
        audio_input_positions=None,
        audio_context=None,
    )


def _masked_source(
    speech: Speech,
    runtime: DataRuntime,
    *,
    mask_text_ratio: float,
    mask_audio_ratio: float,
    generator: Generator | None,
) -> Tensor:
    mask_id = int(runtime.mask_token_id)
    text = runtime.layout.to_global(Modality.TEXT.value, speech.text_token_ids).clone()
    audio = runtime.layout.to_global(Modality.AUDIO.value, speech.audio_token_ids).clone()
    text[_mask_indices(text.numel(), mask_text_ratio, generator, device=text.device)] = (
        mask_id
    )
    audio[
        _mask_indices(audio.numel(), mask_audio_ratio, generator, device=audio.device)
    ] = mask_id
    return torch.cat(
        (
            text,
            audio.new_tensor([runtime.boa_token_id]),
            audio,
            audio.new_tensor([runtime.eoa_token_id]),
        )
    )


def _mask_indices(
    length: int,
    ratio: float,
    generator: Generator | None,
    *,
    device: torch.device,
) -> Tensor:
    if length == 0 or ratio <= 0.0:
        return torch.zeros(0, dtype=torch.long, device=device)
    count = min(length, max(1, int(round(length * ratio))))
    permutation = torch.randperm(length, generator=generator, device=device)
    return permutation[:count]


def _validate_ratio(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a float.")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")


__all__ = ["build_masked_sample"]
