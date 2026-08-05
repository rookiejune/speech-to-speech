from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor

from ..datamodule.batch import ModelBatch
from ..task import PredictionModality, Request
from .audio import generate_audio_responses
from .contract import Result
from .contract import TokenGenerator
from .mixed import generate_mixed_responses
from .request import response_of, validate


@torch.no_grad()
def generate_responses(
    requests: Sequence[Request],
    model: TokenGenerator,
    *,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
    use_cache: bool = True,
) -> list[Result]:
    """Generate batched responses grouped by resolved task response."""
    results: list[Result | None] = [None] * len(requests)
    device = model.backbone.get_input_embeddings().weight.device
    groups: dict[tuple[object, str, PredictionModality], list[tuple[int, Request]]] = {}
    for index, request in enumerate(requests):
        validate(request, model)
        response = response_of(request)
        groups.setdefault(
            (request["task"], response.name, response.prediction),
            [],
        ).append((index, request))

    for (_, _, prediction), group in groups.items():
        prompt, prompt_mask, audio_input_positions = _inputs(
            [request for _, request in group],
            model,
            device,
        )
        response = response_of(group[0][1])
        if prediction.is_mixed or len(response.fields) > 1:
            mixed_results = generate_mixed_responses(
                [request for _, request in group],
                model,
                prompt,
                prompt_mask,
                audio_input_positions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                use_cache=use_cache,
            )
            for result, (result_index, _) in zip(mixed_results, group):
                results[result_index] = result
            continue
        if prediction is PredictionModality.AUDIO:
            audio_results = generate_audio_responses(
                [request for _, request in group],
                model,
                prompt,
                prompt_mask,
                audio_input_positions,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                use_cache=use_cache,
            )
            if len(audio_results) != len(group):
                raise RuntimeError("audio generation returned the wrong row count.")
            for result, (result_index, _) in zip(audio_results, group):
                results[result_index] = result
            continue
        if prediction is not PredictionModality.TEXT:
            raise ValueError(f"unsupported prediction modality: {prediction.value}")

        from anydataset.types import Modality

        sequence = model.generate_tokens(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_attention_mask=prompt_mask,
            audio_input_positions=audio_input_positions,
            stop_token_id=model.runtime.eos_token_id,
            generation_modality=Modality.TEXT,
            do_sample=do_sample,
            use_cache=use_cache,
        )
        responses = [
            _response(sequence[row], prompt.size(1), model.runtime.eos_token_id)
            for row in range(len(group))
        ]
        for token_ids, (result_index, _) in zip(responses, group):
            results[result_index] = Result(response_ids=token_ids, audio=None)

    if any(result is None for result in results):
        raise RuntimeError("generation did not produce every requested result.")
    return cast(list[Result], results)


def _response(sequence: Tensor, prompt_length: int, stop_token_id: int) -> Tensor:
    response = sequence[prompt_length:]
    stops = response.eq(stop_token_id).nonzero()
    if stops.numel():
        return response[: int(stops[0].item())]
    return response


def _inputs(
    requests: list[Request],
    model: TokenGenerator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor | None]:
    prompts = [request["prompt_ids"].to(device=device) for request in requests]
    width = max(prompt.numel() for prompt in prompts)
    prompt = torch.full(
        (len(prompts), width),
        model.runtime.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    prompt_mask = torch.zeros_like(prompt, dtype=torch.bool)
    for row, value in enumerate(prompts):
        prompt[row, -value.numel() :] = value
        prompt_mask[row, -value.numel() :] = True

    position_values = [request.get("audio_input_positions") for request in requests]
    if not any(value is not None for value in position_values):
        return prompt, prompt_mask, None
    width_frames = max(
        0 if value is None else int(value.numel()) for value in position_values
    )
    positions = torch.full(
        (len(requests), width_frames),
        -1,
        dtype=torch.long,
        device=device,
    )
    for row, value in enumerate(position_values):
        if value is not None and value.numel():
            offset = width - prompts[row].numel()
            positions[row, : value.numel()] = value.to(device=device) + offset
    return prompt, prompt_mask, positions


def requests_from_batch(batch: ModelBatch) -> list[Request]:
    """Build unpadded inference requests from teacher-forcing samples."""
    requests: list[Request] = []
    prompt_lengths = batch.generation_prompt_lengths
    audio_input_positions = batch.audio_input_positions
    traces = batch.response_traces
    if prompt_lengths is None:
        raise RuntimeError("model batch generation fields are unavailable.")
    for index, (task, prediction) in enumerate(zip(batch.tasks, batch.predictions)):
        prompt_end = int(prompt_lengths[index].item())
        requests.append(
            Request(
                prompt_ids=batch.input_ids[index, :prompt_end],
                task=task,
                prediction=prediction,
                trace=traces[index],
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
