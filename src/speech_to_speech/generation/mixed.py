from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, auto

import torch
from anydataset.types import Modality
from torch import Tensor

from ..prediction import PredictionModality
from ..task import Task
from .audio import decode_token_audio_rows
from .protocol import TokenGenerator
from .types import Request, Result


class _State(Enum):
    TEXT = auto()
    FORCE_BOA = auto()
    AUDIO = auto()
    DONE = auto()


@torch.no_grad()
def generate_mixed_responses(
    requests: Sequence[Request],
    model: TokenGenerator,
    prompt: Tensor,
    prompt_mask: Tensor,
    audio_input_positions: Tensor | None,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    use_cache: bool,
) -> list[Result]:
    if not requests:
        return []
    prediction = requests[0]["task"].prediction_modality
    if not prediction.is_mixed:
        raise ValueError("mixed generation requires PARALLEL or INTERLEAVED prediction.")
    if any(request["task"].prediction_modality is not prediction for request in requests):
        raise ValueError("mixed generation batch must share prediction modality.")

    runtime = model.runtime
    device = prompt.device
    batch = prompt.size(0)
    capacity = prompt.size(1) + max_new_tokens
    generated = prompt.new_full((batch, capacity), runtime.pad_token_id)
    generated[:, : prompt.size(1)] = prompt
    attention = torch.zeros_like(generated, dtype=torch.bool)
    attention[:, : prompt.size(1)] = prompt_mask
    length = prompt.size(1)
    states = [_State.TEXT for _ in range(batch)]
    finished = [False] * batch
    past = None
    audio_past = None
    cache_rows: list[int] | None = None
    input_ids = generated[:, :length]
    positions = audio_input_positions

    text_allowed = torch.tensor(
        runtime.generation_allowed_ids(Modality.TEXT),
        dtype=torch.long,
        device=device,
    )
    audio_allowed = torch.tensor(
        runtime.audio_generation_allowed_ids,
        dtype=torch.long,
        device=device,
    )
    boa_id = runtime.boa_token_id
    eoa_id = runtime.eoa_token_id
    eos_id = runtime.eos_token_id

    for _ in range(max_new_tokens):
        if all(finished):
            break
        active = torch.tensor(
            [index for index, done in enumerate(finished) if not done],
            dtype=torch.long,
            device=device,
        )
        active_list = active.tolist()
        if (
            use_cache
            and past is not None
            and cache_rows is not None
            and cache_rows != active_list
        ):
            indices = torch.tensor(
                [cache_rows.index(row) for row in active_list],
                dtype=torch.long,
                device=device,
            )
            past.batch_select_indices(indices)
            audio_past = model.audio_output_adapter_batch_select(audio_past, indices)
            cache_rows = active_list

        active_input = input_ids.index_select(0, active)
        active_mask = attention.index_select(0, active)[:, :length]
        active_positions = (
            None if positions is None else positions.index_select(0, active)
        )
        # Force-BOA rows emit a constant; they still join the model step when any
        # other active row needs logits so backbone/audio caches stay aligned.
        forced = []
        sampled_rows = []
        for local, row in enumerate(active_list):
            if states[row] is _State.FORCE_BOA:
                forced.append((row, boa_id))
            else:
                sampled_rows.append((local, row))

        next_ids = generated.new_full((batch,), runtime.pad_token_id)
        if sampled_rows:
            union_parts = [
                text_allowed,
                audio_allowed,
                text_allowed.new_tensor([boa_id, eos_id]),
            ]
            union = torch.unique(torch.cat(union_parts))
            output = model.generation_step(
                active_input,
                attention_mask=active_mask,
                output_hidden_states=False,
                token_ids=union,
                modality=None,
                past_key_values=past,
                use_cache=use_cache,
                audio_input_positions=active_positions,
                audio_output_past=audio_past,
            )
            if output.logits is None:
                raise RuntimeError("model did not return generation logits.")
            logits = output.logits[:, -1] / temperature
            past = output.past_key_values
            audio_past = output.audio_output_past
            if use_cache:
                if past is None:
                    raise RuntimeError("backbone did not return a generation cache.")
                cache_rows = active_list
            for local, row in sampled_rows:
                state = states[row]
                if state is _State.TEXT:
                    if prediction is PredictionModality.PARALLEL:
                        allowed = torch.unique(
                            torch.cat([text_allowed, text_allowed.new_tensor([eos_id])])
                        )
                    else:
                        allowed = torch.unique(
                            torch.cat(
                                [
                                    text_allowed,
                                    text_allowed.new_tensor([boa_id, eos_id]),
                                ]
                            )
                        )
                elif state is _State.AUDIO:
                    allowed = audio_allowed
                else:
                    raise AssertionError(f"unexpected mixed generation state: {state}")
                row_logits = _restrict(logits[local], union, allowed)
                if top_p < 1.0:
                    row_logits = _top_p(row_logits, top_p)
                choice = (
                    torch.distributions.Categorical(logits=row_logits).sample()
                    if do_sample
                    else row_logits.argmax()
                )
                token = int(allowed[choice].item())
                next_ids[row] = token
                if state is _State.TEXT:
                    if token == eos_id:
                        if prediction is PredictionModality.PARALLEL:
                            states[row] = _State.FORCE_BOA
                        else:
                            states[row] = _State.DONE
                            finished[row] = True
                    elif token == boa_id:
                        states[row] = _State.AUDIO
                elif state is _State.AUDIO and token == eoa_id:
                    if prediction is PredictionModality.PARALLEL:
                        states[row] = _State.DONE
                        finished[row] = True
                    else:
                        states[row] = _State.TEXT

        for row, token in forced:
            next_ids[row] = token
            states[row] = _State.AUDIO

        generated[:, length] = next_ids
        for row in range(batch):
            if finished[row] and next_ids[row].item() == runtime.pad_token_id:
                attention[row, length] = False
            elif not finished[row] or next_ids[row].item() != runtime.pad_token_id:
                attention[row, length] = True
            if states[row] is _State.DONE:
                finished[row] = True
        length += 1
        if use_cache:
            input_ids = next_ids.unsqueeze(1)
            positions = None
        else:
            past = None
            audio_past = None
            cache_rows = None
            input_ids = generated[:, :length]
            # Keep original source positions for full recomputes.

    results: list[Result] = []
    response_rows: list[Tensor] = []
    for row in range(batch):
        response = generated[row, prompt.size(1) : length]
        response = response[attention[row, prompt.size(1) : length]]
        response_rows.append(response)
    audios = decode_token_audio_rows(response_rows, model)
    for response, audio in zip(response_rows, audios):
        results.append(Result(response_ids=response, audio=audio))
    return results


def _restrict(logits: Tensor, union: Tensor, allowed: Tensor) -> Tensor:
    matches = allowed[:, None].eq(union[None, :])
    if not bool(matches.any(dim=1).all()):
        raise RuntimeError("allowed mixed-generation ids escape the union set.")
    return logits.index_select(0, matches.to(dtype=torch.long).argmax(dim=1))


def _top_p(logits: Tensor, top_p: float) -> Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    result = torch.full_like(logits, torch.finfo(logits.dtype).min)
    result.scatter_(0, sorted_indices, sorted_logits)
    return result


def supports_mixed(task: Task) -> bool:
    return task.prediction_modality.is_mixed


__all__ = ["generate_mixed_responses", "supports_mixed"]
