from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers.cache_utils import Cache

from ._helper import top_p_filter
from .protocol import TokenModelRuntime


@dataclass
class GenerationStepResult:
    """One autoregressive step: logits plus backbone and audio-output caches."""

    logits: Tensor
    past_key_values: Cache | None
    audio_output_past: object | None
    hidden_states: tuple[Tensor, ...] | None = None


@dataclass
class GenerationOutput:
    """Generated token sequences plus optional per-token generation metadata."""

    sequences: Tensor
    token_logprobs: Tensor | None = None
    token_logprob_mask: Tensor | None = None
    audio_condition: Tensor | None = None
    frame_spans: Tensor | None = None


@dataclass
class _GenerationLoopState:
    generated: Tensor
    attention_mask: Tensor
    input_ids: Tensor
    active_mask: Tensor
    audio_input_positions: Tensor | None
    length: int
    past_key_values: Cache | None = None
    audio_output_past: object | None = None


_DEVICE_DONE_CHECK_INTERVAL = 16


class GenerationStepModel(Protocol):
    layout: Layout
    runtime: TokenModelRuntime
    audio_token_frame_spans: Tensor

    def generation_step(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor,
        output_hidden_states: bool,
        token_ids: Tensor | None,
        token_kind: str | None = None,
        modality: Modality | None,
        past_key_values: Cache | None,
        use_cache: bool,
        audio_input_positions: Tensor | None = None,
        audio_output_past: object | None = None,
    ) -> GenerationStepResult: ...

    def audio_output_adapter_batch_select(
        self,
        past_key_values: object | None,
        indices: Tensor,
    ) -> object | None: ...


def generate_sequence_full(
    model: GenerationStepModel,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_attention_mask: Tensor | None,
    audio_input_positions: Tensor | None,
    stop_token_id: int | None,
    generation_modality: Modality | None,
    allowed_token_ids: Sequence[int] | Tensor | None,
    token_kind: str | None = None,
    do_sample: bool,
    use_cache: bool,
    collect_audio_condition: bool,
    collect_logprobs: bool = False,
    min_new_tokens: int = 0,
) -> GenerationOutput:
    prompt_attention_mask = _validate_generation_inputs(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_p=top_p,
        prompt_attention_mask=prompt_attention_mask,
        audio_input_positions=audio_input_positions,
        generation_modality=generation_modality,
        allowed_token_ids=allowed_token_ids,
    )
    generation_token_ids = _generation_token_ids(
        allowed_token_ids,
        prompt_ids,
        model.layout,
    )
    stop_logit_index = (
        _stop_logit_index(
            stop_token_id,
            generation_token_ids,
            generation_modality,
            model.layout,
        )
        if min_new_tokens
        else None
    )
    state = _initial_loop_state(
        prompt_ids,
        prompt_attention_mask,
        max_new_tokens,
        audio_input_positions,
    )
    batch_size = prompt_ids.size(0)
    condition_steps: list[Tensor] = []
    span_steps: list[Tensor] = []
    logprob_steps: list[Tensor] = []
    logprob_mask_steps: list[Tensor] = []

    for step in range(max_new_tokens):
        output = _generation_loop_step(
            model,
            state,
            collect_audio_condition=collect_audio_condition,
            generation_token_ids=generation_token_ids,
            token_kind=token_kind,
            generation_modality=generation_modality,
            use_cache=use_cache,
        )
        logits = _sampling_logits(
            output.logits,
            temperature,
            top_p=top_p,
            step=step,
            min_new_tokens=min_new_tokens,
            stop_logit_index=stop_logit_index,
        )
        next_indices = _sample_next_indices(logits, do_sample, state.active_mask)
        if collect_logprobs:
            _append_logprobs(
                logprob_steps,
                logprob_mask_steps,
                logits,
                next_indices,
                state.active_mask,
            )
        next_ids = _selected_token_ids(
            next_indices,
            generation_token_ids,
            generation_modality,
            model.layout,
        )
        if collect_audio_condition:
            _append_audio_condition(
                condition_steps,
                span_steps,
                model,
                output,
                next_ids,
                state.active_mask,
            )
        _advance_generation_state(
            state,
            output,
            next_ids,
            stop_token_id=stop_token_id,
            use_cache=use_cache,
        )
        if _all_rows_finished(
            state.active_mask,
            step=step,
            max_new_tokens=max_new_tokens,
        ):
            break

    generated_steps = _generated_steps(
        state.attention_mask,
        prompt_width=prompt_ids.size(1),
        attempted_steps=state.length - prompt_ids.size(1),
        stop_token_id=stop_token_id,
    )
    return _build_generation_output(
        state.generated[:, : prompt_ids.size(1) + generated_steps],
        prompt_ids,
        batch_size=batch_size,
        logprob_steps=logprob_steps[:generated_steps],
        logprob_mask_steps=logprob_mask_steps[:generated_steps],
        condition_steps=condition_steps[:generated_steps],
        span_steps=span_steps[:generated_steps],
        collect_logprobs=collect_logprobs,
    )


def _validate_generation_inputs(
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_attention_mask: Tensor | None,
    audio_input_positions: Tensor | None,
    generation_modality: Modality | None,
    allowed_token_ids: Sequence[int] | Tensor | None,
) -> Tensor:
    if (
        max_new_tokens < 0
        or min_new_tokens < 0
        or min_new_tokens > max_new_tokens
        or temperature <= 0
        or not 0 < top_p <= 1
    ):
        raise ValueError("invalid generation parameters")
    if generation_modality is not None and generation_modality not in {
        Modality.TEXT,
        Modality.AUDIO,
    }:
        raise ValueError(f"unsupported generation modality: {generation_modality.value}")
    if generation_modality is not None and allowed_token_ids is not None:
        raise ValueError("generation modality and allowed token ids cannot both be provided.")
    if prompt_ids.dim() != 2 or prompt_ids.size(0) < 1:
        raise ValueError("generation requires at least one prompt row.")
    if prompt_attention_mask is None:
        prompt_attention_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    if prompt_attention_mask.shape != prompt_ids.shape:
        raise ValueError("prompt attention mask must align with prompt ids.")
    if audio_input_positions is not None and (
        audio_input_positions.dim() != 2 or audio_input_positions.size(0) != prompt_ids.size(0)
    ):
        raise ValueError("audio_input_positions must have shape [batch, frames].")
    if not bool(prompt_attention_mask.any(dim=1).all()):
        raise ValueError("each generation prompt must contain at least one token.")
    return prompt_attention_mask


def _initial_loop_state(
    prompt_ids: Tensor,
    prompt_attention_mask: Tensor,
    max_new_tokens: int,
    audio_input_positions: Tensor | None,
) -> _GenerationLoopState:
    prompt_width = prompt_ids.size(1)
    capacity = prompt_width + max_new_tokens
    generated = prompt_ids.new_empty(prompt_ids.size(0), capacity)
    generated[:, :prompt_width] = prompt_ids
    attention_mask = torch.zeros_like(generated, dtype=torch.bool)
    attention_mask[:, :prompt_width] = prompt_attention_mask
    batch_size = prompt_ids.size(0)
    return _GenerationLoopState(
        generated=generated,
        attention_mask=attention_mask,
        input_ids=generated[:, :prompt_width],
        active_mask=torch.ones(batch_size, dtype=torch.bool, device=prompt_ids.device),
        audio_input_positions=audio_input_positions,
        length=prompt_width,
    )


def _generation_loop_step(
    model: GenerationStepModel,
    state: _GenerationLoopState,
    *,
    collect_audio_condition: bool,
    generation_token_ids: Tensor | None,
    token_kind: str | None,
    generation_modality: Modality | None,
    use_cache: bool,
) -> GenerationStepResult:
    if token_kind is not None:
        output = model.generation_step(
            state.input_ids,
            attention_mask=state.attention_mask[:, : state.length],
            output_hidden_states=collect_audio_condition,
            token_ids=generation_token_ids,
            token_kind=token_kind,
            modality=generation_modality,
            past_key_values=state.past_key_values,
            use_cache=use_cache,
            audio_input_positions=state.audio_input_positions,
            audio_output_past=state.audio_output_past,
        )
    else:
        output = model.generation_step(
            state.input_ids,
            attention_mask=state.attention_mask[:, : state.length],
            output_hidden_states=collect_audio_condition,
            token_ids=generation_token_ids,
            modality=generation_modality,
            past_key_values=state.past_key_values,
            use_cache=use_cache,
            audio_input_positions=state.audio_input_positions,
            audio_output_past=state.audio_output_past,
        )
    if output.logits is None:
        raise RuntimeError("model did not return generation logits.")
    return output


def _sampling_logits(
    logits: Tensor,
    temperature: float,
    *,
    top_p: float,
    step: int,
    min_new_tokens: int,
    stop_logit_index: int | None,
) -> Tensor:
    logits = logits[:, -1] / temperature
    if (
        stop_logit_index is not None
        and stop_logit_index < logits.size(1)
        and step < min_new_tokens
    ):
        logits[:, stop_logit_index] = float("-inf")
        if step == 0 and not bool(torch.isfinite(logits).any(dim=1).all()):
            raise ValueError("minimum generation length left no non-stop token to sample.")
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)
    return logits


def _sample_next_indices(
    logits: Tensor,
    do_sample: bool,
    sampled_rows: Tensor,
) -> Tensor:
    if not do_sample:
        return logits.argmax(dim=-1)
    sampled = torch.multinomial(
        logits[sampled_rows].softmax(dim=-1),
        1,
        replacement=True,
    ).squeeze(-1)
    return torch.zeros_like(sampled_rows, dtype=torch.long).masked_scatter(
        sampled_rows,
        sampled,
    )


def _append_logprobs(
    logprob_steps: list[Tensor],
    logprob_mask_steps: list[Tensor],
    logits: Tensor,
    next_indices: Tensor,
    active_mask: Tensor,
) -> None:
    step_logprobs = logits.log_softmax(dim=-1)
    active_token_logprobs = step_logprobs.gather(
        1,
        next_indices.unsqueeze(1),
    ).squeeze(1)
    logprob_steps.append(active_token_logprobs.masked_fill(~active_mask, 0))
    logprob_mask_steps.append(active_mask.clone())


def _selected_token_ids(
    next_indices: Tensor,
    generation_token_ids: Tensor | None,
    generation_modality: Modality | None,
    layout: Layout,
) -> Tensor:
    if generation_token_ids is not None:
        return generation_token_ids.index_select(0, next_indices)
    if generation_modality is not None:
        start, _ = layout.blocks[generation_modality.value]
        return next_indices + start
    return next_indices


def _append_audio_condition(
    condition_steps: list[Tensor],
    span_steps: list[Tensor],
    model: GenerationStepModel,
    output: GenerationStepResult,
    next_ids: Tensor,
    active_mask: Tensor,
) -> None:
    if output.hidden_states is None:
        raise RuntimeError("model did not return generation hidden states.")
    codec_start, codec_end = model.runtime.codec_audio_range
    codec_tokens = next_ids.ge(codec_start) & next_ids.lt(codec_end)
    local_ids = (next_ids - codec_start).clamp(0, model.audio_token_frame_spans.numel() - 1)
    spans = model.audio_token_frame_spans.index_select(0, local_ids)
    span_steps.append(spans.masked_fill(~(codec_tokens & active_mask), 0))
    condition = output.hidden_states[-1][:, -1]
    condition_steps.append(condition.masked_fill(~active_mask[:, None], 0))


def _advance_generation_state(
    state: _GenerationLoopState,
    output: GenerationStepResult,
    next_ids: Tensor,
    *,
    stop_token_id: int | None,
    use_cache: bool,
) -> None:
    emitted = state.active_mask
    written_ids = (
        next_ids
        if stop_token_id is None
        else torch.where(emitted, next_ids, stop_token_id)
    )
    state.generated[:, state.length] = written_ids
    state.attention_mask[:, state.length] = emitted
    state.length += 1
    if stop_token_id is not None:
        state.active_mask = emitted & next_ids.ne(stop_token_id)
    if use_cache:
        _advance_cached_state(state, output, written_ids)
    else:
        _advance_full_recompute_state(state)


def _advance_cached_state(
    state: _GenerationLoopState,
    output: GenerationStepResult,
    next_ids: Tensor,
) -> None:
    state.audio_input_positions = None
    state.past_key_values = output.past_key_values
    if state.past_key_values is None:
        raise RuntimeError("backbone did not return a generation cache.")
    state.audio_output_past = output.audio_output_past
    state.input_ids = next_ids.unsqueeze(-1)


def _advance_full_recompute_state(
    state: _GenerationLoopState,
) -> None:
    state.audio_output_past = None
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


def _generated_steps(
    attention_mask: Tensor,
    *,
    prompt_width: int,
    attempted_steps: int,
    stop_token_id: int | None,
) -> int:
    if stop_token_id is None or attempted_steps == 0:
        return attempted_steps
    generated_attention = attention_mask[
        :, prompt_width : prompt_width + attempted_steps
    ]
    # Device generation may run a few masked steps between completion checks.
    return int(generated_attention.any(dim=0).sum().item())


def _build_generation_output(
    generated: Tensor,
    prompt_ids: Tensor,
    *,
    batch_size: int,
    logprob_steps: list[Tensor],
    logprob_mask_steps: list[Tensor],
    condition_steps: list[Tensor],
    span_steps: list[Tensor],
    collect_logprobs: bool,
) -> GenerationOutput:
    token_logprobs: Tensor | None = None
    token_logprob_mask: Tensor | None = None
    if collect_logprobs:
        if logprob_steps:
            token_logprobs = torch.stack(logprob_steps, dim=1)
            token_logprob_mask = torch.stack(logprob_mask_steps, dim=1)
        else:
            token_logprobs = torch.empty(
                batch_size,
                0,
                dtype=torch.float32,
                device=prompt_ids.device,
            )
            token_logprob_mask = torch.zeros(
                batch_size,
                0,
                dtype=torch.bool,
                device=prompt_ids.device,
            )
    if not span_steps:
        return GenerationOutput(
            sequences=generated,
            token_logprobs=token_logprobs,
            token_logprob_mask=token_logprob_mask,
        )
    frame_spans = torch.stack(span_steps, dim=1)
    frame_counts = frame_spans.sum(dim=1)
    if bool(frame_counts.eq(0).any()):
        raise ValueError("an audio generation row produced no codec-decodable tokens.")
    token_conditions = torch.stack(condition_steps, dim=1)
    condition = pad_sequence(
        [
            torch.repeat_interleave(
                token_conditions[row],
                frame_spans[row],
                dim=0,
            )
            for row in range(prompt_ids.size(0))
        ],
        batch_first=True,
    )
    return GenerationOutput(
        sequences=generated,
        token_logprobs=token_logprobs,
        token_logprob_mask=token_logprob_mask,
        audio_condition=condition,
        frame_spans=frame_spans,
    )


def generate_sequence(
    model: GenerationStepModel,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_attention_mask: Tensor | None,
    audio_input_positions: Tensor | None,
    stop_token_id: int | None,
    generation_modality: Modality | None,
    allowed_token_ids: Sequence[int] | Tensor | None,
    token_kind: str | None = None,
    do_sample: bool,
    use_cache: bool,
    collect_audio_condition: bool,
    min_new_tokens: int = 0,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    output = generate_sequence_full(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        prompt_attention_mask=prompt_attention_mask,
        audio_input_positions=audio_input_positions,
        stop_token_id=stop_token_id,
        generation_modality=generation_modality,
        allowed_token_ids=allowed_token_ids,
        token_kind=token_kind,
        do_sample=do_sample,
        use_cache=use_cache,
        collect_audio_condition=collect_audio_condition,
        min_new_tokens=min_new_tokens,
    )
    return output.sequences, output.audio_condition, output.frame_spans


def _stop_logit_index(
    stop_token_id: int | None,
    generation_token_ids: Tensor | None,
    generation_modality: Modality | None,
    layout: Layout,
) -> int | None:
    if stop_token_id is None:
        return None
    if generation_token_ids is not None:
        stop = generation_token_ids.eq(stop_token_id)
        if not bool(stop.any()):
            return None
        if generation_token_ids.numel() == 1:
            raise ValueError("minimum generation length left no non-stop token to sample.")
        return int(stop.nonzero(as_tuple=False)[0].item())
    elif generation_modality is not None:
        start, end = layout.blocks[generation_modality.value]
        if not start <= stop_token_id < end:
            return None
        if end - start == 1:
            raise ValueError("minimum generation length left no non-stop token to sample.")
        return stop_token_id - start
    return stop_token_id


def _generation_token_ids(
    allowed_token_ids: Sequence[int] | Tensor | None,
    prompt_ids: Tensor,
    layout: Layout,
) -> Tensor | None:
    if allowed_token_ids is None:
        return None
    token_ids = torch.as_tensor(
        allowed_token_ids,
        device=prompt_ids.device,
        dtype=torch.long,
    )
    if token_ids.dim() != 1 or token_ids.numel() == 0:
        raise ValueError("allowed_token_ids must be a non-empty 1D sequence.")
    if token_ids.unique().numel() != token_ids.numel():
        raise ValueError("allowed_token_ids must not contain duplicates.")
    text_start, text_end = layout.blocks["text"]
    audio_start, audio_end = layout.blocks["audio"]
    text_mask = token_ids.ge(text_start) & token_ids.lt(text_end)
    audio_mask = token_ids.ge(audio_start) & token_ids.lt(audio_end)
    if not bool((text_mask | audio_mask).all()):
        raise ValueError("allowed_token_ids contains an invalid vocabulary id.")
    return token_ids
