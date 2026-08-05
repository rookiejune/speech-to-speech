from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypedDict

import torch
from anydataset.types import Modality
from torch import Tensor

from ..model.generation import GenerationOutput
from ..task import (
    ControlToken,
    PredictionModality,
    Request,
    ResponseControl,
    response_control_tokens,
)
from .request import response_of, target_language_of, validate


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
    control_token_ids: tuple[int, ...]

    def control_token_id(self, token: ControlToken) -> int: ...

    def generation_allowed_ids(self, modality: Modality) -> tuple[int, ...]: ...


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
        allowed_token_ids: Sequence[int] | Tensor | None = None,
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
    rows: list[RolloutRow | None] = [None] * len(requests)
    groups: dict[tuple[ResponseControl, str | None], list[tuple[int, Request]]] = {}
    for index, request in enumerate(requests):
        response = response_of(request)
        if response.prediction is not PredictionModality.TEXT:
            raise ValueError("rollout logprobs currently support text prediction only.")
        if len(response.fields) != 1:
            raise ValueError(
                "rollout logprobs currently support single-step text responses only."
            )
        validate(request, model)  # type: ignore[arg-type]
        groups.setdefault(
            (
                response.steps[0].control,
                target_language_of(request, response=response),
            ),
            [],
        ).append((index, request))
    if not requests:
        return []

    for group in groups.values():
        group_requests = [request for _, request in group]
        response_spec = response_of(group_requests[0])
        target_language = target_language_of(
            group_requests[0],
            response=response_spec,
        )
        stop_token_id, allowed_token_ids = _text_generation_contract(
            model.runtime,
            response_spec.steps[0].control,
            target_language=target_language,
        )
        prompt, prompt_mask, audio_input_positions = _inputs(group_requests, model)
        output = model.generate_tokens_with_logprobs(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_attention_mask=prompt_mask,
            audio_input_positions=audio_input_positions,
            stop_token_id=stop_token_id,
            generation_modality=None,
            allowed_token_ids=allowed_token_ids,
            do_sample=do_sample,
            use_cache=use_cache,
        )
        if output.token_logprobs is None or output.token_logprob_mask is None:
            raise RuntimeError("generation did not return rollout logprobs.")
        for row, (index, request) in enumerate(group):
            response, logprobs, finish_reason = _response(
                output.sequences[row],
                output.token_logprobs[row],
                output.token_logprob_mask[row],
                prompt.size(1),
                stop_token_id,
                finish_reason=(
                    "eos"
                    if stop_token_id == model.runtime.eos_token_id
                    else "control"
                ),
            )
            rows[index] = RolloutRow(
                version=1,
                index=index,
                task=request["task"].value,
                prediction=response_of(request).prediction.value,
                prompt_ids=_ids(request["prompt_ids"]),
                response_ids=_ids(response),
                response_logprobs=[float(value) for value in logprobs.tolist()],
                finish_reason=finish_reason,
            )
    if any(row is None for row in rows):
        raise RuntimeError("rollout generation did not produce every requested row.")
    return [row for row in rows if row is not None]


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
    *,
    finish_reason: str,
) -> tuple[Tensor, Tensor, str]:
    response = sequence[prompt_length:]
    valid_steps = int(token_logprob_mask.sum().item())
    logprobs = token_logprobs[:valid_steps]
    response = response[:valid_steps]
    stops = response.eq(stop_token_id).nonzero(as_tuple=False).flatten()
    if stops.numel():
        end = int(stops[0].item()) + 1
        return response[:end], logprobs[:end], finish_reason
    return response, logprobs, "length"


def _text_generation_contract(
    runtime: _RolloutRuntime,
    control: ResponseControl,
    *,
    target_language: str | None,
) -> tuple[int, tuple[int, ...]]:
    controls = response_control_tokens(
        control,
        target_language=target_language,
    )
    if controls is not None:
        prefix_ids = tuple(
            runtime.control_token_id(token) for token in controls.prefix
        )
        stop_token_id = runtime.control_token_id(controls.end)
    elif control is ResponseControl.EOS:
        prefix_ids = ()
        stop_token_id = runtime.eos_token_id
    else:
        raise ValueError("rollout text responses require EOS, ASR, or MT controls.")
    blocked = {runtime.eos_token_id, *runtime.control_token_ids}
    lexical = tuple(
        token_id
        for token_id in runtime.generation_allowed_ids(Modality.TEXT)
        if token_id not in blocked
    )
    return stop_token_id, (*lexical, *prefix_ids, stop_token_id)


def _ids(tensor: Tensor) -> list[int]:
    return [int(value) for value in tensor.detach().cpu().tolist()]
