from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

import torch
from anydataset.types import Modality
from torch import Tensor
from transformers.cache_utils import Cache

from ..prediction import PredictionModality
from ..task import Task
from ._request import prediction_of
from .audio import decode_token_audio_rows
from .protocol import TokenGenerator
from .types import Request, Result


class _State(Enum):
    TEXT = auto()
    FORCE_BOA = auto()
    AUDIO = auto()
    DONE = auto()


@dataclass(frozen=True)
class _TokenSets:
    text: Tensor
    audio: Tensor
    union: Tensor
    boa_id: int
    eoa_id: int
    eos_id: int
    pad_id: int


@dataclass
class _MixedLoopState:
    generated: Tensor
    attention: Tensor
    input_ids: Tensor
    positions: Tensor | None
    length: int
    states: list[_State]
    finished: list[bool]
    past: Cache | None = None
    audio_past: object | None = None
    cache_rows: list[int] | None = None

    @property
    def batch_size(self) -> int:
        return len(self.states)

    def active_rows(self, device: torch.device) -> Tensor:
        return torch.tensor(
            [index for index, done in enumerate(self.finished) if not done],
            dtype=torch.long,
            device=device,
        )

    def done(self) -> bool:
        return all(self.finished)


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
    prediction = _validate_mixed_batch(requests)
    if prediction is None:
        return []
    device = prompt.device
    tokens = _token_sets(model, device)
    state = _initial_state(
        prompt,
        prompt_mask,
        audio_input_positions,
        max_new_tokens,
        pad_token_id=tokens.pad_id,
    )

    for _ in range(max_new_tokens):
        if state.done():
            break
        active = state.active_rows(device)
        active_list = active.tolist()
        _align_cache(state, model, active_list, device=device, use_cache=use_cache)
        forced, sampled_rows = _split_forced_rows(
            state.states,
            active_list,
            boa_id=tokens.boa_id,
        )
        next_ids = state.generated.new_full((state.batch_size,), tokens.pad_id)
        if sampled_rows:
            logits = _sampled_logits(
                state,
                model,
                active,
                active_list,
                tokens,
                temperature=temperature,
                use_cache=use_cache,
            )
            _sample_rows(
                state,
                next_ids,
                sampled_rows,
                logits,
                tokens,
                prediction,
                top_p=top_p,
                do_sample=do_sample,
            )
        _force_rows(state, next_ids, forced)
        _advance_loop(state, next_ids, pad_token_id=tokens.pad_id, use_cache=use_cache)
    return _mixed_results(state, prompt.size(1), model)


def _validate_mixed_batch(
    requests: Sequence[Request],
) -> PredictionModality | None:
    if not requests:
        return None
    prediction = prediction_of(requests[0])
    if not prediction.is_mixed:
        raise ValueError("mixed generation requires PARALLEL or INTERLEAVED prediction.")
    if any(prediction_of(request) is not prediction for request in requests):
        raise ValueError("mixed generation batch must share prediction modality.")
    return prediction


def _token_sets(model: TokenGenerator, device: torch.device) -> _TokenSets:
    runtime = model.runtime
    text = torch.tensor(
        runtime.generation_allowed_ids(Modality.TEXT),
        dtype=torch.long,
        device=device,
    )
    audio = torch.tensor(
        runtime.audio_generation_allowed_ids,
        dtype=torch.long,
        device=device,
    )
    return _TokenSets(
        text=text,
        audio=audio,
        union=torch.unique(
            torch.cat([text, audio, text.new_tensor([runtime.boa_token_id, runtime.eos_token_id])])
        ),
        boa_id=runtime.boa_token_id,
        eoa_id=runtime.eoa_token_id,
        eos_id=runtime.eos_token_id,
        pad_id=runtime.pad_token_id,
    )


def _initial_state(
    prompt: Tensor,
    prompt_mask: Tensor,
    audio_input_positions: Tensor | None,
    max_new_tokens: int,
    *,
    pad_token_id: int,
) -> _MixedLoopState:
    capacity = prompt.size(1) + max_new_tokens
    generated = prompt.new_full((prompt.size(0), capacity), pad_token_id)
    generated[:, : prompt.size(1)] = prompt
    attention = torch.zeros_like(generated, dtype=torch.bool)
    attention[:, : prompt.size(1)] = prompt_mask
    return _MixedLoopState(
        generated=generated,
        attention=attention,
        input_ids=generated[:, : prompt.size(1)],
        positions=audio_input_positions,
        length=prompt.size(1),
        states=[_State.TEXT for _ in range(prompt.size(0))],
        finished=[False] * prompt.size(0),
    )


def _align_cache(
    state: _MixedLoopState,
    model: TokenGenerator,
    active_rows: list[int],
    *,
    device: torch.device,
    use_cache: bool,
) -> None:
    if (
        not use_cache
        or state.past is None
        or state.cache_rows is None
        or state.cache_rows == active_rows
    ):
        return
    indices = torch.tensor(
        [state.cache_rows.index(row) for row in active_rows],
        dtype=torch.long,
        device=device,
    )
    state.past.batch_select_indices(indices)
    state.audio_past = model.audio_output_adapter_batch_select(state.audio_past, indices)
    state.cache_rows = active_rows


def _split_forced_rows(
    states: Sequence[_State],
    active_rows: Sequence[int],
    *,
    boa_id: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    forced = []
    sampled = []
    for local, row in enumerate(active_rows):
        if states[row] is _State.FORCE_BOA:
            forced.append((row, boa_id))
        else:
            sampled.append((local, row))
    return forced, sampled


def _sampled_logits(
    state: _MixedLoopState,
    model: TokenGenerator,
    active: Tensor,
    active_rows: list[int],
    tokens: _TokenSets,
    *,
    temperature: float,
    use_cache: bool,
) -> Tensor:
    active_input = state.input_ids.index_select(0, active)
    active_mask = state.attention.index_select(0, active)[:, : state.length]
    active_positions = None if state.positions is None else state.positions.index_select(0, active)
    output = model.generation_step(
        active_input,
        attention_mask=active_mask,
        output_hidden_states=False,
        token_ids=tokens.union,
        modality=None,
        past_key_values=state.past,
        use_cache=use_cache,
        audio_input_positions=active_positions,
        audio_output_past=state.audio_past,
    )
    if output.logits is None:
        raise RuntimeError("model did not return generation logits.")
    state.past = output.past_key_values
    state.audio_past = output.audio_output_past
    if use_cache:
        if state.past is None:
            raise RuntimeError("backbone did not return a generation cache.")
        state.cache_rows = active_rows
    return output.logits[:, -1] / temperature


def _sample_rows(
    state: _MixedLoopState,
    next_ids: Tensor,
    sampled_rows: Sequence[tuple[int, int]],
    logits: Tensor,
    tokens: _TokenSets,
    prediction: PredictionModality,
    *,
    top_p: float,
    do_sample: bool,
) -> None:
    for local, row in sampled_rows:
        allowed = _allowed_for_state(state.states[row], prediction, tokens)
        token = _sample_mixed_token(
            logits[local],
            tokens.union,
            allowed,
            top_p=top_p,
            do_sample=do_sample,
        )
        next_ids[row] = token
        _advance_row_state(state, row, token, prediction, tokens)


def _allowed_for_state(
    state: _State,
    prediction: PredictionModality,
    tokens: _TokenSets,
) -> Tensor:
    if state is _State.TEXT:
        if prediction is PredictionModality.PARALLEL:
            return torch.unique(torch.cat([tokens.text, tokens.text.new_tensor([tokens.eos_id])]))
        return torch.unique(
            torch.cat([tokens.text, tokens.text.new_tensor([tokens.boa_id, tokens.eos_id])])
        )
    if state is _State.AUDIO:
        return tokens.audio
    raise AssertionError(f"unexpected mixed generation state: {state}")


def _sample_mixed_token(
    logits: Tensor,
    union: Tensor,
    allowed: Tensor,
    *,
    top_p: float,
    do_sample: bool,
) -> int:
    row_logits = _restrict(logits, union, allowed)
    if top_p < 1.0:
        row_logits = _top_p(row_logits, top_p)
    choice = (
        torch.distributions.Categorical(logits=row_logits).sample()
        if do_sample
        else row_logits.argmax()
    )
    return int(allowed[choice].item())


def _advance_row_state(
    state: _MixedLoopState,
    row: int,
    token: int,
    prediction: PredictionModality,
    tokens: _TokenSets,
) -> None:
    row_state = state.states[row]
    if row_state is _State.TEXT:
        if token == tokens.eos_id:
            if prediction is PredictionModality.PARALLEL:
                state.states[row] = _State.FORCE_BOA
            else:
                state.states[row] = _State.DONE
                state.finished[row] = True
        elif token == tokens.boa_id:
            state.states[row] = _State.AUDIO
    elif row_state is _State.AUDIO and token == tokens.eoa_id:
        if prediction is PredictionModality.PARALLEL:
            state.states[row] = _State.DONE
            state.finished[row] = True
        else:
            state.states[row] = _State.TEXT


def _force_rows(
    state: _MixedLoopState,
    next_ids: Tensor,
    forced: Sequence[tuple[int, int]],
) -> None:
    for row, token in forced:
        next_ids[row] = token
        state.states[row] = _State.AUDIO


def _advance_loop(
    state: _MixedLoopState,
    next_ids: Tensor,
    *,
    pad_token_id: int,
    use_cache: bool,
) -> None:
    state.generated[:, state.length] = next_ids
    for row in range(state.batch_size):
        emitted = int(next_ids[row].item())
        if state.finished[row] and emitted == pad_token_id:
            state.attention[row, state.length] = False
        elif not state.finished[row] or emitted != pad_token_id:
            state.attention[row, state.length] = True
        if state.states[row] is _State.DONE:
            state.finished[row] = True
    state.length += 1
    if use_cache:
        state.input_ids = next_ids.unsqueeze(1)
        state.positions = None
        return
    state.past = None
    state.audio_past = None
    state.cache_rows = None
    state.input_ids = state.generated[:, : state.length]


def _mixed_results(
    state: _MixedLoopState,
    prompt_width: int,
    model: TokenGenerator,
) -> list[Result]:
    response_rows = []
    for row in range(state.batch_size):
        response = state.generated[row, prompt_width : state.length]
        response = response[state.attention[row, prompt_width : state.length]]
        response_rows.append(response)
    audios = decode_token_audio_rows(response_rows, model)
    return [
        Result(response_ids=response, audio=audio)
        for response, audio in zip(response_rows, audios)
    ]


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
