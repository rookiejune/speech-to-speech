from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict, cast

import torch
from anydataset.types import Modality
from torch import Tensor

from ..datamodule.types import ModelBatch, Task
from ..model.protocol import AcousticGeneration, SemanticGeneration
from .decode import decode_generated_audio, decode_generated_semantic


class AcousticPrompt(TypedDict):
    ids: Tensor
    positions: Tensor


class Request(TypedDict):
    prompt_ids: Tensor
    task: Task
    acoustic_prompt: AcousticPrompt | None


class AudioOutput(TypedDict):
    features: Tensor | None
    waveform: Tensor
    sample_rate: int


class Result(TypedDict):
    token_ids: Tensor
    audio: AudioOutput | None


def requests_from_batch(batch: ModelBatch) -> list[Request]:
    """Build unpadded inference requests from teacher-forcing samples."""
    requests: list[Request] = []
    acoustic_mask = batch.acoustic_input_mask
    for index, task in enumerate(batch.tasks):
        target_positions = (batch.labels[index] != -100).nonzero()
        if target_positions.numel() == 0:
            raise ValueError("teacher-forcing batch row has no target tokens.")
        prompt_end = int(target_positions[0].item())

        acoustic_prompt = None
        if batch.acoustic_input_ids is not None:
            if batch.acoustic_input_positions is None or acoustic_mask is None:
                raise RuntimeError("acoustic input fields are incomplete.")
            row_mask = acoustic_mask[index]
            acoustic_prompt = AcousticPrompt(
                ids=batch.acoustic_input_ids[index][row_mask],
                positions=batch.acoustic_input_positions[index][row_mask],
            )

        requests.append(
            Request(
                prompt_ids=batch.input_ids[index, :prompt_end],
                task=task,
                acoustic_prompt=acoustic_prompt,
            )
        )
    return requests


@torch.no_grad()
def generate(
    requests: Sequence[Request],
    model: SemanticGeneration,
    *,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
    use_cache: bool = True,
) -> list[Result]:
    """Generate one response and optional waveform for each inference request."""
    results: list[Result] = []
    device = model.backbone.get_input_embeddings().weight.device
    for request in requests:
        prompt = request["prompt_ids"].to(device=device)[None]
        task = request["task"]
        acoustic_prompt = request["acoustic_prompt"]
        if (
            acoustic_prompt is not None
            and task.source_modality is not Modality.AUDIO
        ):
            raise ValueError(
                f"{task.value} does not accept a source acoustic prompt."
            )
        acoustic_ids = (
            None
            if acoustic_prompt is None
            else acoustic_prompt["ids"].to(device=device)[None]
        )
        acoustic_positions = (
            None
            if acoustic_prompt is None
            else acoustic_prompt["positions"].to(device=device)[None]
        )
        if task.target_modality is Modality.AUDIO:
            if not model.runtime.codec.acoustic_codebook_sizes:
                sequence = model.generate_semantic(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    acoustic_input_ids=acoustic_ids,
                    acoustic_input_positions=acoustic_positions,
                    stop_token_id=model.runtime.eoa_token_id,
                    allowed_token_ids=model.runtime.audio_generation_allowed_ids,
                    do_sample=do_sample,
                    use_cache=use_cache,
                )
                token_ids = _response(
                    sequence[0], prompt.size(1), model.runtime.eoa_token_id
                )
                waveform = decode_generated_semantic(
                    token_ids[None],
                    codec=model.runtime.codec,
                    audio_tokenizer=model.runtime.audio_tokenizer,
                    audio_token_range=model.runtime.codec_audio_range,
                )[0]
                results.append(
                    Result(
                        token_ids=token_ids,
                        audio=AudioOutput(
                            features=None,
                            waveform=waveform,
                            sample_rate=model.runtime.codec.sample_rate,
                        ),
                    )
                )
                continue

            acoustic_model = cast(AcousticGeneration, model)
            sequence, features = acoustic_model.generate_audio(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                acoustic_input_ids=acoustic_ids,
                acoustic_input_positions=acoustic_positions,
                do_sample=do_sample,
                use_cache=use_cache,
            )
            token_ids = _response(
                sequence[0], prompt.size(1), model.runtime.eoa_token_id
            )
            waveform = decode_generated_audio(
                token_ids[None],
                features,
                codec=model.runtime.codec,
                audio_tokenizer=model.runtime.audio_tokenizer,
                audio_token_range=model.runtime.codec_audio_range,
            )[0]
            results.append(
                Result(
                    token_ids=token_ids,
                    audio=AudioOutput(
                        features=features[0],
                        waveform=waveform,
                        sample_rate=model.runtime.codec.sample_rate,
                    ),
                )
            )
            continue

        sequence = model.generate_semantic(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            acoustic_input_ids=acoustic_ids,
            acoustic_input_positions=acoustic_positions,
            stop_token_id=model.runtime.eos_token_id,
            allowed_token_ids=model.runtime.generation_allowed_ids(Modality.TEXT),
            do_sample=do_sample,
            use_cache=use_cache,
        )
        results.append(
            Result(
                token_ids=_response(
                    sequence[0], prompt.size(1), model.runtime.eos_token_id
                ),
                audio=None,
            )
        )
    return results


def _response(sequence: Tensor, prompt_length: int, stop_token_id: int) -> Tensor:
    response = sequence[prompt_length:]
    if response.numel() and int(response[-1].item()) == stop_token_id:
        return response[:-1]
    return response
