from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypedDict

import torch
from anydataset.types import Modality
from torch import Tensor

from ..model.generation import GenerationOutput
from ..task import PredictionModality
from ._request import prediction_of, validate
from ..task import Request


class RolloutRow(TypedDict):
    version: int
    index: int
    task: str
    prediction: str
    prompt_ids: list[int]
    response_ids: list[int]
    response_logprobs: list[float]
    finish_reason: str


class _RolloutRuntime(Protocol):
    pad_token_id: int
    eos_token_id: int


class RolloutGenerator(Protocol):
    runtime: _RolloutRuntime

    def generate_tokens_with_logprobs(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        prompt_attention_mask: Tensor | None = None,
        audio_input_positions: Tensor | None = None,
        stop_token_id: int | None = None,
        generation_modality: Modality | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> GenerationOutput: ...


def generate_rollouts(
    requests: Sequence[Request],
    model: RolloutGenerator,
    *,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
    use_cache: bool = True,
) -> list[RolloutRow]:
    rows: list[RolloutRow] = []
    for request in requests:
        prediction = prediction_of(request)
        if prediction is not PredictionModality.TEXT:
            raise ValueError("rollout logprobs currently support text prediction only.")
        validate(request, model)  # type: ignore[arg-type]
    if not requests:
        return rows

    prompt, prompt_mask, audio_input_positions = _inputs(requests, model)
    output = model.generate_tokens_with_logprobs(
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
    if output.token_logprobs is None or output.token_logprob_mask is None:
        raise RuntimeError("generation did not return rollout logprobs.")
    for index, request in enumerate(requests):
        response, logprobs, finish_reason = _response(
            output.sequences[index],
            output.token_logprobs[index],
            output.token_logprob_mask[index],
            prompt.size(1),
            model.runtime.eos_token_id,
        )
        rows.append(
            RolloutRow(
                version=1,
                index=index,
                task=request["task"].value,
                prediction=PredictionModality.TEXT.value,
                prompt_ids=_ids(request["prompt_ids"]),
                response_ids=_ids(response),
                response_logprobs=[float(value) for value in logprobs.tolist()],
                finish_reason=finish_reason,
            )
        )
    return rows


def write_rollouts_jsonl(path: str | Path, rows: Sequence[RolloutRow]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _inputs(
    requests: Sequence[Request],
    model: RolloutGenerator,
) -> tuple[Tensor, Tensor, Tensor | None]:
    prompts = [request["prompt_ids"] for request in requests]
    device = prompts[0].device
    width = max(prompt.numel() for prompt in prompts)
    prompt = torch.full(
        (len(prompts), width),
        model.runtime.pad_token_id,
        dtype=prompts[0].dtype,
        device=device,
    )
    prompt_mask = torch.zeros_like(prompt, dtype=torch.bool)
    for row, value in enumerate(prompts):
        prompt[row, -value.numel() :] = value
        prompt_mask[row, -value.numel() :] = True
    positions = _audio_input_positions(requests, width, device)
    return prompt, prompt_mask, positions


def _audio_input_positions(
    requests: Sequence[Request],
    prompt_width: int,
    device: torch.device,
) -> Tensor | None:
    values = [request.get("audio_input_positions") for request in requests]
    if all(value is None for value in values):
        return None
    width = max(0 if value is None else value.numel() for value in values)
    positions = torch.full((len(values), width), -1, dtype=torch.long, device=device)
    for row, value in enumerate(values):
        if value is None:
            continue
        offset = prompt_width - requests[row]["prompt_ids"].numel()
        positions[row, : value.numel()] = value.to(device=device) + offset
    return positions


def _response(
    sequence: Tensor,
    token_logprobs: Tensor,
    token_logprob_mask: Tensor,
    prompt_length: int,
    stop_token_id: int,
) -> tuple[Tensor, Tensor, str]:
    response = sequence[prompt_length:]
    valid_steps = int(token_logprob_mask.sum().item())
    logprobs = token_logprobs[:valid_steps]
    response = response[:valid_steps]
    stops = response.eq(stop_token_id).nonzero(as_tuple=False).flatten()
    if stops.numel():
        end = int(stops[0].item())
        return response[:end], logprobs[:end], "eos"
    return response, logprobs, "length"


def _ids(tensor: Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().tolist()]
