from __future__ import annotations

import torch
from anydataset.types import Modality
from torch import Tensor

from ...prediction import PredictionModality
from ...runtime import AudioRepresentation
from ...task import Task
from .._helper.tokenization import token_ids
from ..protocol import DataRuntime
from ..types import AcousticTarget, ModelSample, Speech, Text


_AR_TASKS = frozenset(
    {
        Task.AUDIO_AR,
        Task.INTERLEAVED_AR,
        Task.PARALLEL_AR,
        Task.TEXT_AR,
    }
)


def is_ar_task(task: Task) -> bool:
    return task in _AR_TASKS


def build_ar_sample(
    target: Speech | Text,
    task: Task,
    runtime: DataRuntime,
    *,
    prompt: str,
    prediction: PredictionModality | None = None,
    interleave_audio_frames: int = 25,
) -> ModelSample:
    if not is_ar_task(task):
        raise ValueError(f"{task.value} is not an autoregressive task.")
    if task.source_modality is not None:
        raise ValueError(f"{task.value} must not use a source modality.")
    from ...task_spec import resolve_prediction

    prediction = resolve_prediction(task, prediction)
    marker = token_ids(prompt, runtime.text_tokenizer)

    if prediction is PredictionModality.TEXT:
        if not isinstance(target, Text):
            raise TypeError("TEXT_AR target must be Text.")
        response = _append_eos(
            runtime.layout.to_global(Modality.TEXT.value, target.text_token_ids),
            runtime,
        )
        return _pack(
            marker,
            response,
            task=task,
            prediction=prediction,
            supervise_from=0,
            acoustic_target=None,
            audio_seconds=0.0,
        )

    if not isinstance(target, Speech):
        raise TypeError(f"{task.value} target must be Speech.")

    if prediction is PredictionModality.AUDIO:
        audio = _boa_eoa(
            runtime.layout.to_global(Modality.AUDIO.value, target.audio_token_ids),
            runtime,
        )
        acoustic = _acoustic_target(
            target,
            runtime,
            audio_token_start=marker.numel() + 1,
        )
        return _pack(
            marker,
            audio,
            task=task,
            prediction=prediction,
            supervise_from=1,
            acoustic_target=acoustic,
            audio_seconds=_duration(target),
        )

    if prediction is PredictionModality.PARALLEL:
        return pack_parallel(
            marker,
            target,
            task,
            runtime,
            prediction=prediction,
        )

    if prediction is PredictionModality.INTERLEAVED:
        return pack_interleaved(
            marker,
            target,
            task,
            runtime,
            prediction=prediction,
            interleave_audio_frames=interleave_audio_frames,
        )

    raise ValueError(f"unsupported AR prediction modality: {prediction.value}")


def pack_parallel(
    marker: Tensor,
    speech: Speech,
    task: Task,
    runtime: DataRuntime,
    *,
    prediction: PredictionModality,
) -> ModelSample:
    text = _append_eos(
        runtime.layout.to_global(Modality.TEXT.value, speech.text_token_ids),
        runtime,
    )
    audio = _boa_eoa(
        runtime.layout.to_global(Modality.AUDIO.value, speech.audio_token_ids),
        runtime,
    )
    response = torch.cat([text, audio])
    labels = torch.full_like(torch.cat([marker, response]), -100)
    labels[marker.numel() : marker.numel() + text.numel()] = text
    labels[marker.numel() + text.numel() + 1 :] = audio[1:]
    acoustic = _acoustic_target(
        speech,
        runtime,
        audio_token_start=marker.numel() + text.numel() + 1,
    )
    return ModelSample.pack(
        prompt_ids=marker,
        response_ids=response,
        token_labels=labels,
        token_groups=None,
        acoustic_target=acoustic,
        task=task,
        prediction=prediction,
        audio_seconds=_duration(speech),
        audio_input_positions=None,
        audio_context=None,
    )


def pack_interleaved(
    marker: Tensor,
    speech: Speech,
    task: Task,
    runtime: DataRuntime,
    *,
    prediction: PredictionModality,
    interleave_audio_frames: int,
) -> ModelSample:
    if (
        isinstance(interleave_audio_frames, bool)
        or not isinstance(interleave_audio_frames, int)
        or interleave_audio_frames < 1
    ):
        raise ValueError("interleave_audio_frames must be a positive integer.")
    audio_local = speech.audio_token_ids
    text_local = speech.text_token_ids
    spans = speech.audio_token_spans
    if audio_local.numel() == 0:
        raise ValueError("interleaved prediction requires non-empty audio tokens.")
    chunks: list[tuple[Tensor, Tensor]] = []
    index = 0
    while index < audio_local.numel():
        end = index
        frames = 0
        while end < audio_local.numel():
            next_frames = frames + int(spans[end].item())
            if end > index and next_frames > interleave_audio_frames:
                break
            frames = next_frames
            end += 1
        chunks.append((audio_local[index:end], spans[index:end]))
        index = end

    text_pieces = _split_proportional(text_local, len(chunks))
    pieces: list[Tensor] = []
    label_pieces: list[Tensor] = []
    audio_token_positions: list[Tensor] = []
    cursor = marker.numel()
    for text_chunk, (audio_chunk, span_chunk) in zip(text_pieces, chunks):
        text_ids = runtime.layout.to_global(Modality.TEXT.value, text_chunk)
        audio_ids = _boa_eoa(
            runtime.layout.to_global(Modality.AUDIO.value, audio_chunk),
            runtime,
        )
        pieces.extend([text_ids, audio_ids])
        label_pieces.append(text_ids)
        label_pieces.append(
            torch.cat(
                [
                    audio_ids.new_full((1,), -100),
                    audio_ids[1:],
                ]
            )
        )
        audio_start = cursor + text_ids.numel() + 1
        audio_token_positions.append(
            torch.arange(
                audio_start,
                audio_start + audio_chunk.numel(),
                dtype=torch.long,
            )
        )
        cursor += text_ids.numel() + audio_ids.numel()

    response = torch.cat(pieces)
    response = _append_eos(response, runtime)
    content_labels = torch.cat(label_pieces)
    labels = torch.full((marker.numel() + response.numel(),), -100, dtype=torch.long)
    labels[marker.numel() : marker.numel() + content_labels.numel()] = content_labels
    labels[-1] = runtime.eos_token_id
    positions = torch.cat(audio_token_positions)
    expanded = torch.repeat_interleave(positions, spans)
    acoustic = _acoustic_from_positions(speech, runtime, expanded)
    return ModelSample.pack(
        prompt_ids=marker,
        response_ids=response,
        token_labels=labels,
        token_groups=None,
        acoustic_target=acoustic,
        task=task,
        prediction=prediction,
        audio_seconds=_duration(speech),
        audio_input_positions=None,
        audio_context=None,
    )


def _split_proportional(values: Tensor, parts: int) -> list[Tensor]:
    if parts < 1:
        raise ValueError("interleaved layout requires at least one chunk.")
    total = int(values.numel())
    if total == 0:
        return [values.new_empty((0,), dtype=values.dtype) for _ in range(parts)]
    out: list[Tensor] = []
    for index in range(parts):
        start = (index * total) // parts
        end = ((index + 1) * total) // parts
        out.append(values[start:end])
    return out


def _pack(
    marker: Tensor,
    response: Tensor,
    *,
    task: Task,
    prediction: PredictionModality,
    supervise_from: int,
    acoustic_target: AcousticTarget | None,
    audio_seconds: float,
) -> ModelSample:
    full = torch.cat([marker, response])
    labels = torch.full_like(full, -100)
    labels[marker.numel() + supervise_from :] = response[supervise_from:]
    return ModelSample.pack(
        prompt_ids=marker,
        response_ids=response,
        token_labels=labels,
        token_groups=None,
        acoustic_target=acoustic_target,
        task=task,
        prediction=prediction,
        audio_seconds=audio_seconds,
        audio_input_positions=None,
        audio_context=None,
    )


def _acoustic_target(
    speech: Speech,
    runtime: DataRuntime,
    *,
    audio_token_start: int,
) -> AcousticTarget | None:
    positions = torch.arange(
        audio_token_start,
        audio_token_start + speech.audio_token_ids.numel(),
        dtype=torch.long,
    )
    expanded = torch.repeat_interleave(positions, speech.audio_token_spans)
    return _acoustic_from_positions(speech, runtime, expanded)


def _acoustic_from_positions(
    speech: Speech,
    runtime: DataRuntime,
    token_positions: Tensor,
) -> AcousticTarget | None:
    if speech.acoustic_codes is None or runtime.semantic_codec_artifact is not None:
        return None
    if runtime.audio_representation is AudioRepresentation.FULL_CODEC_SEQUENCE:
        return None
    if token_positions.numel() != speech.acoustic_codes.size(0):
        raise ValueError("target acoustic frames and audio tokens must align.")
    return AcousticTarget(
        semantic_codes=speech.semantic_codes,
        codes=speech.acoustic_codes,
        token_positions=token_positions,
    )


def _duration(speech: Speech) -> float:
    if speech.duration_seconds is None:
        raise ValueError(
            "speech is missing duration_seconds; parse raw audio samples with a "
            "DataRuntime so duration can be read from metadata or inferred from "
            "codec frames."
        )
    return float(speech.duration_seconds)


def _boa_eoa(ids: Tensor, runtime: DataRuntime) -> Tensor:
    return torch.cat(
        (
            ids.new_tensor([runtime.boa_token_id]),
            ids,
            ids.new_tensor([runtime.eoa_token_id]),
        )
    )


def _append_eos(ids: Tensor, runtime: DataRuntime) -> Tensor:
    return torch.cat([ids, ids.new_tensor([runtime.eos_token_id])])


__all__ = ["build_ar_sample", "is_ar_task"]
