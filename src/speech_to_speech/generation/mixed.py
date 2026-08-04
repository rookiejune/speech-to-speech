from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

import torch
from anydataset.types import Modality
from torch import Tensor
from transformers.cache_utils import Cache

from ..task import PredictionModality, Task
from ._request import prediction_of
from .audio import decode_token_audio_results
from .protocol import TokenGenerator
from ..task import Request
from .result import Result


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
    parallel_text_mask: Tensor
    interleaved_text_mask: Tensor
    audio_mask: Tensor
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
    states: Tensor
    active_mask: Tensor
    past: Cache | None = None
    audio_head_past: object | None = None
    input_validated: bool = False

    @property
    def batch_size(self) -> int:
        return self.generated.size(0)


_DEVICE_DONE_CHECK_INTERVAL = 16


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

    for step in range(max_new_tokens):
        logits = _mixed_logits(
            state,
            model,
            tokens,
            temperature=temperature,
            use_cache=use_cache,
        )
        next_ids = _mixed_next_ids(
            logits,
            state.states,
            tokens,
            prediction,
            top_p=top_p,
            do_sample=do_sample,
        )
        _advance_loop(
            state,
            next_ids,
            prediction=prediction,
            tokens=tokens,
            use_cache=use_cache,
        )
        if _all_rows_finished(
            state.active_mask,
            step=step,
            max_new_tokens=max_new_tokens,
        ):
            break
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
    union = torch.unique(
        torch.cat([text, audio, text.new_tensor([runtime.boa_token_id, runtime.eos_token_id])])
    )
    text_mask = _member_mask(union, text)
    parallel_text_mask = text_mask | union.eq(runtime.eos_token_id)
    return _TokenSets(
        text=text,
        audio=audio,
        union=union,
        parallel_text_mask=parallel_text_mask,
        interleaved_text_mask=parallel_text_mask | union.eq(runtime.boa_token_id),
        audio_mask=_member_mask(union, audio),
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
        states=torch.full(
            (prompt.size(0),),
            _State.TEXT.value,
            dtype=torch.int8,
            device=prompt.device,
        ),
        active_mask=torch.ones(prompt.size(0), dtype=torch.bool, device=prompt.device),
    )


def _mixed_logits(
    state: _MixedLoopState,
    model: TokenGenerator,
    tokens: _TokenSets,
    *,
    temperature: float,
    use_cache: bool,
) -> Tensor:
    validate_input = not state.input_validated
    output = model.generation_step(
        state.input_ids,
        attention_mask=state.attention[:, : state.length],
        output_hidden_states=False,
        token_ids=tokens.union,
        token_kind="mixed",
        modality=None,
        past_key_values=state.past,
        use_cache=use_cache,
        audio_input_positions=state.positions,
        audio_head_past=state.audio_head_past,
        input_modalities=(
            None
            if validate_input
            else frozenset((Modality.TEXT, Modality.AUDIO))
        ),
        validate_input=validate_input,
        validate_audio_input_positions=validate_input,
    )
    if output.logits is None:
        raise RuntimeError("model did not return generation logits.")
    state.past = output.past_key_values
    state.audio_head_past = output.audio_head_past
    state.input_validated = True
    if use_cache:
        if state.past is None:
            raise RuntimeError("backbone did not return a generation cache.")
    return output.logits[:, -1] / temperature


def _mixed_next_ids(
    logits: Tensor,
    states: Tensor,
    tokens: _TokenSets,
    prediction: PredictionModality,
    *,
    top_p: float,
    do_sample: bool,
) -> Tensor:
    text_rows = states.eq(_State.TEXT.value)
    audio_rows = states.eq(_State.AUDIO.value)
    sampled_rows = text_rows | audio_rows
    text_mask = (
        tokens.parallel_text_mask
        if prediction is PredictionModality.PARALLEL
        else tokens.interleaved_text_mask
    )
    row_logits = logits[sampled_rows]
    allowed = torch.where(
        text_rows[sampled_rows, None],
        text_mask[None, :],
        tokens.audio_mask[None, :],
    )
    row_logits = row_logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
    if top_p < 1.0:
        row_logits = _top_p(row_logits, top_p)
    choices = (
        torch.multinomial(
            row_logits.softmax(dim=-1),
            1,
            replacement=True,
        ).squeeze(-1)
        if do_sample
        else row_logits.argmax(dim=-1)
    )
    sampled_ids = tokens.union.index_select(0, choices)
    next_ids = torch.full_like(states, tokens.pad_id, dtype=torch.long)
    next_ids.masked_scatter_(sampled_rows, sampled_ids)
    return torch.where(states.eq(_State.FORCE_BOA.value), tokens.boa_id, next_ids)


def _next_states(
    states: Tensor,
    next_ids: Tensor,
    prediction: PredictionModality,
    tokens: _TokenSets,
) -> Tensor:
    text_rows = states.eq(_State.TEXT.value)
    audio_rows = states.eq(_State.AUDIO.value)
    next_states = states.clone()
    next_states.masked_fill_(states.eq(_State.FORCE_BOA.value), _State.AUDIO.value)
    if prediction is PredictionModality.PARALLEL:
        next_states.masked_fill_(
            text_rows & next_ids.eq(tokens.eos_id),
            _State.FORCE_BOA.value,
        )
        next_states.masked_fill_(
            audio_rows & next_ids.eq(tokens.eoa_id),
            _State.DONE.value,
        )
        return next_states
    next_states.masked_fill_(
        text_rows & next_ids.eq(tokens.eos_id),
        _State.DONE.value,
    )
    next_states.masked_fill_(
        text_rows & next_ids.eq(tokens.boa_id),
        _State.AUDIO.value,
    )
    next_states.masked_fill_(
        audio_rows & next_ids.eq(tokens.eoa_id),
        _State.TEXT.value,
    )
    return next_states


def _advance_loop(
    state: _MixedLoopState,
    next_ids: Tensor,
    *,
    prediction: PredictionModality,
    tokens: _TokenSets,
    use_cache: bool,
) -> None:
    emitted = state.active_mask
    next_ids = torch.where(emitted, next_ids, tokens.pad_id)
    state.generated[:, state.length] = next_ids
    state.attention[:, state.length] = emitted
    state.states = _next_states(state.states, next_ids, prediction, tokens)
    state.active_mask = state.states.ne(_State.DONE.value)
    state.length += 1
    if use_cache:
        state.input_ids = next_ids.unsqueeze(1)
        state.positions = None
        return
    state.past = None
    state.audio_head_past = None
    state.input_ids = state.generated[:, : state.length]


def _all_rows_finished(
    active_mask: Tensor,
    *,
    step: int,
    max_new_tokens: int,
) -> bool:
    if step + 1 >= max_new_tokens:
        return False
    if active_mask.device.type != "cpu" and (step + 1) % _DEVICE_DONE_CHECK_INTERVAL:
        return False
    return not bool(active_mask.any())


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
    return decode_token_audio_results(response_rows, model)


def _member_mask(union: Tensor, allowed: Tensor) -> Tensor:
    return union[:, None].eq(allowed[None, :]).any(dim=1)


def _top_p(logits: Tensor, top_p: float) -> Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    result = torch.full_like(logits, torch.finfo(logits.dtype).min)
    result.scatter_(-1, sorted_indices, sorted_logits)
    return result


def supports_mixed(task: Task) -> bool:
    return task.prediction_modality.is_mixed


__all__ = ["generate_mixed_responses", "supports_mixed"]
