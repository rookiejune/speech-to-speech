from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

import torch
from anydataset.types import Modality
from torch import Tensor
from transformers.cache_utils import Cache

from ..runtime.audio_schema import AudioTokenSpec
from ..task import (
    PredictionModality,
    Request,
    ResponseControl,
    ResponseSpec,
    response_control_tokens,
)
from .audio import audio_generation_variant, decode_token_audio_results
from .contract import TokenGenerator
from .contract import Result
from .request import response_of, target_language_of


class _State(Enum):
    PREFIX = auto()
    TEXT = auto()
    AUDIO_SCHEMA = auto()
    AUDIO = auto()
    DONE = auto()


@dataclass(frozen=True)
class _TokenSets:
    lexical: Tensor
    audio: Tensor
    union: Tensor
    text_step_masks: Tensor
    interleaved_text_mask: Tensor
    prefix_ids: Tensor
    prefix_lengths: Tensor
    end_ids: Tensor
    boa_id: int
    audio_schema_id: int
    eoa_id: int
    eos_id: int
    pad_id: int
    codec_start: int
    audio_spec: AudioTokenSpec | None


@dataclass
class _MixedLoopState:
    generated: Tensor
    attention: Tensor
    input_ids: Tensor
    positions: Tensor | None
    length: int
    states: Tensor
    step_indices: Tensor
    prefix_indices: Tensor
    audio_starts: Tensor
    audio_variants: tuple[str, ...]
    active_mask: Tensor
    past: Cache | None = None
    audio_head_past: object | None = None
    input_validated: bool = False

    @property
    def batch_size(self) -> int:
        return self.generated.size(0)


_DEVICE_DONE_CHECK_INTERVAL = 16


@torch.no_grad()
def generate_program_responses(
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
    response, target_language = _validate_program_batch(requests)
    if response is None:
        return []
    prediction = response.prediction
    device = prompt.device
    tokens = _token_sets(
        model,
        response,
        target_language=target_language,
        device=device,
    )
    state = _initial_state(
        prompt,
        prompt_mask,
        audio_input_positions,
        max_new_tokens,
        response=response,
        tokens=tokens,
        audio_variants=(
            tuple(audio_generation_variant(request, model) for request in requests)
            if response.prediction.supervises_audio
            else ("",) * len(requests)
        ),
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
            state,
            tokens,
            prediction,
            top_p=top_p,
            do_sample=do_sample,
        )
        _advance_loop(
            state,
            next_ids,
            prediction=prediction,
            response=response,
            tokens=tokens,
            use_cache=use_cache,
        )
        if _all_rows_finished(
            state.active_mask,
            step=step,
            max_new_tokens=max_new_tokens,
        ):
            break
    return _program_results(
        state,
        prompt.size(1),
        model,
        response,
        requests,
    )


def _validate_program_batch(
    requests: Sequence[Request],
) -> tuple[ResponseSpec | None, str | None]:
    if not requests:
        return None, None
    response = response_of(requests[0])
    if (
        not response.prediction.is_mixed
        and len(response.steps) < 2
        and not response.uses_control_tokens
    ):
        raise ValueError(
            "program generation requires a controlled, mixed, or multi-step response."
        )
    if any(response_of(request) != response for request in requests):
        raise ValueError("program generation batch must share one response spec.")
    target_language = target_language_of(requests[0], response=response)
    if any(
        target_language_of(request, response=response) != target_language
        for request in requests[1:]
    ):
        raise ValueError("program generation batch must share one target language.")
    return response, target_language


def _token_sets(
    model: TokenGenerator,
    response: ResponseSpec,
    *,
    target_language: str | None,
    device: torch.device,
) -> _TokenSets:
    runtime = model.runtime
    blocked_text_ids = {runtime.eos_token_id, *runtime.control_token_ids}
    lexical = torch.tensor(
        tuple(
            token_id
            for token_id in runtime.generation_allowed_ids(Modality.TEXT)
            if token_id not in blocked_text_ids
        ),
        dtype=torch.long,
        device=device,
    )
    codec_start, codec_end = runtime.codec_audio_range
    audio = torch.tensor(
        (
            (*range(codec_start, codec_end), runtime.eoa_token_id)
            if response.prediction.supervises_audio
            else ()
        ),
        dtype=torch.long,
        device=device,
    )
    prefixes: list[tuple[int, ...]] = []
    end_ids: list[int] = []
    for step in response.steps:
        controls = response_control_tokens(
            step.control,
            target_language=target_language,
        )
        if controls is not None:
            prefixes.append(
                tuple(runtime.control_token_id(token) for token in controls.prefix)
            )
            end_ids.append(runtime.control_token_id(controls.end))
        elif step.control is ResponseControl.EOS:
            prefixes.append(())
            end_ids.append(runtime.eos_token_id)
        elif step.control is ResponseControl.AUDIO:
            prefixes.append(
                (runtime.boa_token_id, runtime.audio_schema_token_id)
            )
            end_ids.append(runtime.eoa_token_id)
        else:  # pragma: no cover - exhaustive ResponseControl mapping
            raise AssertionError(f"unsupported response control: {step.control.value}")
    prefix_lengths = torch.tensor(
        [len(prefix) for prefix in prefixes],
        dtype=torch.long,
        device=device,
    )
    prefix_ids = torch.full(
        (len(prefixes), max(1, max(len(prefix) for prefix in prefixes))),
        -1,
        dtype=torch.long,
        device=device,
    )
    for index, prefix in enumerate(prefixes):
        if prefix:
            prefix_ids[index, : len(prefix)] = torch.tensor(
                prefix,
                dtype=torch.long,
                device=device,
            )
    end_id_tensor = torch.tensor(end_ids, dtype=torch.long, device=device)
    boundary_ids = torch.cat(
        (
            prefix_ids[prefix_ids.ge(0)],
            end_id_tensor,
        )
    )
    union = torch.unique(torch.cat((lexical, audio, boundary_ids)))
    lexical_mask = _member_mask(union, lexical)
    text_step_masks = torch.stack(
        tuple(
            (
                lexical_mask | union.eq(end_ids[index])
                if step.modality is Modality.TEXT
                else torch.zeros_like(lexical_mask)
            )
            for index, step in enumerate(response.steps)
        )
    )
    return _TokenSets(
        lexical=lexical,
        audio=audio,
        union=union,
        text_step_masks=text_step_masks,
        interleaved_text_mask=(
            lexical_mask
            | union.eq(runtime.eos_token_id)
            | union.eq(runtime.boa_token_id)
        ),
        prefix_ids=prefix_ids,
        prefix_lengths=prefix_lengths,
        end_ids=end_id_tensor,
        boa_id=runtime.boa_token_id,
        audio_schema_id=runtime.audio_schema_token_id,
        eoa_id=runtime.eoa_token_id,
        eos_id=runtime.eos_token_id,
        pad_id=runtime.pad_token_id,
        codec_start=codec_start,
        audio_spec=(
            runtime.output_audio_token_spec
            if response.prediction.supervises_audio
            else None
        ),
    )


def _initial_state(
    prompt: Tensor,
    prompt_mask: Tensor,
    audio_input_positions: Tensor | None,
    max_new_tokens: int,
    *,
    response: ResponseSpec,
    tokens: _TokenSets,
    audio_variants: tuple[str, ...],
    pad_token_id: int,
) -> _MixedLoopState:
    if len(audio_variants) != prompt.size(0):
        raise ValueError("audio grammar variants must align with generation rows.")
    first_state = (
        _State.TEXT
        if response.prediction is PredictionModality.INTERLEAVED
        else (
            _State.PREFIX
            if int(tokens.prefix_lengths[0].item()) > 0
            else (
                _State.TEXT
                if response.fields[0].modality is Modality.TEXT
                else _State.AUDIO
            )
        )
    )
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
            first_state.value,
            dtype=torch.int8,
            device=prompt.device,
        ),
        step_indices=torch.zeros(
            prompt.size(0),
            dtype=torch.long,
            device=prompt.device,
        ),
        prefix_indices=torch.zeros(
            prompt.size(0),
            dtype=torch.long,
            device=prompt.device,
        ),
        audio_starts=torch.full(
            (prompt.size(0),),
            -1,
            dtype=torch.long,
            device=prompt.device,
        ),
        audio_variants=audio_variants,
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
        token_kind=("mixed" if tokens.audio.numel() else "text"),
        modality=None,
        past_key_values=state.past,
        use_cache=use_cache,
        audio_input_positions=state.positions,
        audio_head_past=state.audio_head_past,
        input_modalities=(
            None
            if validate_input
            else (
                frozenset((Modality.TEXT, Modality.AUDIO))
                if tokens.audio.numel()
                else frozenset((Modality.TEXT,))
            )
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
    state: _MixedLoopState,
    tokens: _TokenSets,
    prediction: PredictionModality,
    *,
    top_p: float,
    do_sample: bool,
) -> Tensor:
    states = state.states
    prefix_rows = states.eq(_State.PREFIX.value)
    text_rows = states.eq(_State.TEXT.value)
    schema_rows = states.eq(_State.AUDIO_SCHEMA.value)
    audio_rows = states.eq(_State.AUDIO.value)
    sampled_rows = prefix_rows | text_rows | schema_rows | audio_rows
    text_masks = (
        tokens.interleaved_text_mask.expand(states.size(0), -1)
        if prediction is PredictionModality.INTERLEAVED
        else tokens.text_step_masks.index_select(
            0,
            state.step_indices.clamp_max(tokens.text_step_masks.size(0) - 1),
        )
    )
    selected_steps = state.step_indices.clamp_max(tokens.prefix_ids.size(0) - 1)
    selected_prefixes = tokens.prefix_ids.index_select(0, selected_steps)
    expected_prefix_ids = selected_prefixes.gather(
        1,
        state.prefix_indices.clamp_max(selected_prefixes.size(1) - 1).unsqueeze(1),
    ).squeeze(1)
    prefix_masks = tokens.union[None, :].eq(expected_prefix_ids[:, None])
    schema_mask = tokens.union.eq(tokens.audio_schema_id).expand(states.size(0), -1)
    allowed_rows = _audio_candidate_masks(state, tokens, audio_rows)
    allowed_rows = torch.where(schema_rows[:, None], schema_mask, allowed_rows)
    allowed_rows = torch.where(text_rows[:, None], text_masks, allowed_rows)
    allowed_rows = torch.where(prefix_rows[:, None], prefix_masks, allowed_rows)
    row_logits = logits[sampled_rows]
    allowed = allowed_rows[sampled_rows]
    if allowed.numel() and not bool(allowed.any(dim=1).all()):
        raise RuntimeError("program grammar produced an empty generation candidate set.")
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
    return next_ids


def _audio_candidate_masks(
    state: _MixedLoopState,
    tokens: _TokenSets,
    audio_rows: Tensor,
) -> Tensor:
    masks = torch.zeros(
        (state.batch_size, tokens.union.numel()),
        dtype=torch.bool,
        device=tokens.union.device,
    )
    if bool(audio_rows.any()) and tokens.audio_spec is None:
        raise RuntimeError("audio generation requires an output audio token spec.")
    spec = tokens.audio_spec
    for row_tensor in audio_rows.nonzero(as_tuple=False).flatten():
        row = int(row_tensor.item())
        payload_start = int(state.audio_starts[row].item())
        if payload_start < 0 or payload_start > state.length:
            raise RuntimeError("audio grammar state is missing its payload boundary.")
        local_prefix = (
            state.generated[row, payload_start : state.length].to(dtype=torch.long)
            - tokens.codec_start
        )
        if spec is None:  # pragma: no cover - guarded for static narrowing
            raise AssertionError("audio token spec disappeared during generation.")
        candidates = spec.next_candidates(
            local_prefix,
            variants=(state.audio_variants[row],),
        )
        row_mask = masks[row]
        for marker_id in candidates.marker_ids:
            row_mask |= tokens.union.eq(tokens.codec_start + marker_id)
        for start, end in candidates.token_ranges:
            row_mask |= tokens.union.ge(tokens.codec_start + start) & tokens.union.lt(
                tokens.codec_start + end
            )
        if candidates.allows_eoa:
            row_mask |= tokens.union.eq(tokens.eoa_id)
    return masks


def _next_states(
    states: Tensor,
    step_indices: Tensor,
    prefix_indices: Tensor,
    next_ids: Tensor,
    prediction: PredictionModality,
    response: ResponseSpec,
    tokens: _TokenSets,
) -> tuple[Tensor, Tensor, Tensor]:
    prefix_rows = states.eq(_State.PREFIX.value)
    text_rows = states.eq(_State.TEXT.value)
    schema_rows = states.eq(_State.AUDIO_SCHEMA.value)
    audio_rows = states.eq(_State.AUDIO.value)
    next_states = states.clone()
    next_prefix_indices = prefix_indices + prefix_rows.to(dtype=prefix_indices.dtype)
    step_modalities = states.new_tensor(
        [
            (
                _State.TEXT.value
                if step.modality is Modality.TEXT
                else _State.AUDIO.value
            )
            for step in response.steps
        ]
    )
    after_prefix = step_modalities.index_select(
        0,
        step_indices.clamp_max(len(response.steps) - 1),
    )
    prefix_lengths = tokens.prefix_lengths.index_select(
        0,
        step_indices.clamp_max(len(response.steps) - 1),
    )
    prefix_done = prefix_rows & next_prefix_indices.ge(prefix_lengths)
    next_states = torch.where(prefix_done, after_prefix, next_states)
    if prediction is PredictionModality.INTERLEAVED:
        next_states.masked_fill_(
            text_rows & next_ids.eq(tokens.eos_id),
            _State.DONE.value,
        )
        next_states.masked_fill_(
            text_rows & next_ids.eq(tokens.boa_id),
            _State.AUDIO_SCHEMA.value,
        )
        next_states.masked_fill_(
            schema_rows & next_ids.eq(tokens.audio_schema_id),
            _State.AUDIO.value,
        )
        next_states.masked_fill_(
            audio_rows & next_ids.eq(tokens.eoa_id),
            _State.TEXT.value,
        )
        return next_states, step_indices, next_prefix_indices
    expected_end_ids = tokens.end_ids.index_select(
        0,
        step_indices.clamp_max(tokens.end_ids.size(0) - 1),
    )
    return _advance_response_steps(
        next_states,
        step_indices,
        next_prefix_indices,
        text_rows & next_ids.eq(expected_end_ids),
        audio_rows & next_ids.eq(expected_end_ids),
        response,
        tokens,
    )


def _advance_response_steps(
    states: Tensor,
    step_indices: Tensor,
    prefix_indices: Tensor,
    text_done: Tensor,
    audio_done: Tensor,
    response: ResponseSpec,
    tokens: _TokenSets,
) -> tuple[Tensor, Tensor, Tensor]:
    completed = text_done | audio_done
    next_indices = step_indices + completed.to(dtype=step_indices.dtype)
    finished = completed & next_indices.ge(len(response.steps))
    states.masked_fill_(finished, _State.DONE.value)
    continuing = completed & ~finished
    modality_states = states.new_tensor(
        [
            (
                _State.TEXT.value
                if step.modality is Modality.TEXT
                else _State.AUDIO.value
            )
            for step in response.steps
        ]
    )
    field_states = torch.where(
        tokens.prefix_lengths.gt(0),
        states.new_full(tokens.prefix_lengths.shape, _State.PREFIX.value),
        modality_states,
    )
    selected = field_states.index_select(
        0,
        next_indices.clamp_max(len(response.steps) - 1),
    )
    states = torch.where(continuing, selected, states)
    prefix_indices = torch.where(
        continuing,
        torch.zeros_like(prefix_indices),
        prefix_indices,
    )
    return states, next_indices, prefix_indices


def _advance_loop(
    state: _MixedLoopState,
    next_ids: Tensor,
    *,
    prediction: PredictionModality,
    response: ResponseSpec,
    tokens: _TokenSets,
    use_cache: bool,
) -> None:
    emitted = state.active_mask
    previous_states = state.states
    next_ids = torch.where(emitted, next_ids, tokens.pad_id)
    state.generated[:, state.length] = next_ids
    state.attention[:, state.length] = emitted
    next_states, next_steps, next_prefixes = _next_states(
        state.states,
        state.step_indices,
        state.prefix_indices,
        next_ids,
        prediction,
        response,
        tokens,
    )
    entered_audio = previous_states.ne(_State.AUDIO.value) & next_states.eq(
        _State.AUDIO.value
    )
    left_audio = previous_states.eq(_State.AUDIO.value) & next_states.ne(
        _State.AUDIO.value
    )
    state.audio_starts = torch.where(
        entered_audio,
        torch.full_like(state.audio_starts, state.length + 1),
        state.audio_starts,
    )
    state.audio_starts.masked_fill_(left_audio, -1)
    state.states = next_states
    state.step_indices = next_steps
    state.prefix_indices = next_prefixes
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


def _program_results(
    state: _MixedLoopState,
    prompt_width: int,
    model: TokenGenerator,
    response: ResponseSpec,
    requests: Sequence[Request],
) -> list[Result]:
    response_rows = []
    for row in range(state.batch_size):
        response_ids = state.generated[row, prompt_width : state.length]
        response_ids = response_ids[
            state.attention[row, prompt_width : state.length]
        ]
        response_rows.append(response_ids)
    if not response.prediction.supervises_audio:
        return [Result(response_ids=row, audio=None) for row in response_rows]
    return decode_token_audio_results(response_rows, model, requests=requests)


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


__all__ = ["generate_program_responses"]
