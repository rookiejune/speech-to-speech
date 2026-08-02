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

from ..audio_stream import AudioStream
from ..runtime.audio_tokenizer import BiCodecAudioTokenizer
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
    active_rows: Tensor
    audio_input_positions: Tensor | None
    length: int
    past_key_values: Cache | None = None
    audio_output_past: object | None = None


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


class _RowGenerator:
    def __init__(
        self,
        model: GenerationStepModel,
        prompt_ids: Tensor,
        *,
        attention_mask: Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        use_cache: bool,
        grammar: str,
        audio_input_positions: Tensor | None,
    ) -> None:
        self._model = model
        self.sequence = prompt_ids.clone()
        self._attention = attention_mask.clone()
        self._current = self.sequence
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._do_sample = do_sample
        self._use_cache = use_cache
        self._grammar = grammar
        self._past_key_values: Cache | None = None
        self._audio_output_past: object | None = None
        self._audio_input_positions = audio_input_positions
        self._emitted = 0

    def step(self, allowed: Tensor) -> int:
        if self._emitted >= self._max_new_tokens:
            raise ValueError(f"{self._grammar} exceeded max_new_tokens.")
        output = self._model.generation_step(
            self._current,
            attention_mask=self._attention,
            output_hidden_states=False,
            token_ids=allowed,
            modality=None,
            past_key_values=self._past_key_values,
            use_cache=self._use_cache,
            audio_input_positions=(
                self._audio_input_positions
                if not self._use_cache or self._past_key_values is None
                else None
            ),
            audio_output_past=self._audio_output_past,
        )
        if output.logits is None:
            raise RuntimeError("model did not return generation logits.")
        logits = output.logits[:, -1] / self._temperature
        if allowed.numel() == 1:
            index = logits.argmax(dim=-1)
        else:
            if self._top_p < 1.0:
                logits = top_p_filter(logits, self._top_p)
            index = (
                torch.distributions.Categorical(logits=logits).sample()
                if self._do_sample
                else logits.argmax(dim=-1)
            )
        next_id = int(allowed[index].item())
        self.sequence = torch.cat(
            (self.sequence, self.sequence.new_tensor([[next_id]])),
            dim=1,
        )
        self._attention = torch.cat(
            (self._attention, self._attention.new_ones((1, 1))),
            dim=1,
        )
        self._emitted += 1
        if self._use_cache:
            self._past_key_values = output.past_key_values
            if self._past_key_values is None:
                raise RuntimeError("backbone did not return a generation cache.")
            self._audio_output_past = output.audio_output_past
            self._current = self.sequence[:, -1:]
        else:
            self._current = self.sequence
        return next_id


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
            batch_size=batch_size,
            collect_audio_condition=collect_audio_condition,
            generation_token_ids=generation_token_ids,
            generation_modality=generation_modality,
            use_cache=use_cache,
        )
        logits = _sampling_logits(
            output.logits,
            temperature,
            top_p=top_p,
            step=step,
            min_new_tokens=min_new_tokens,
            stop_token_id=stop_token_id,
            generation_token_ids=generation_token_ids,
            generation_modality=generation_modality,
            layout=model.layout,
        )
        next_indices = _sample_next_indices(logits, do_sample)
        if collect_logprobs:
            _append_logprobs(
                logprob_steps,
                logprob_mask_steps,
                logits,
                next_indices,
                state.active_rows,
                batch_size=batch_size,
                device=prompt_ids.device,
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
                state.active_rows,
                batch_size=batch_size,
            )
        if not _advance_generation_state(
            state,
            model,
            output,
            next_ids,
            stop_token_id=stop_token_id,
            use_cache=use_cache,
            batch_size=batch_size,
        ):
            break

    return _build_generation_output(
        state.generated[:, : state.length],
        prompt_ids,
        batch_size=batch_size,
        logprob_steps=logprob_steps,
        logprob_mask_steps=logprob_mask_steps,
        condition_steps=condition_steps,
        span_steps=span_steps,
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
        raise ValueError(
            f"unsupported generation modality: {generation_modality.value}"
        )
    if generation_modality is not None and allowed_token_ids is not None:
        raise ValueError(
            "generation modality and allowed token ids cannot both be provided."
        )
    if prompt_ids.dim() != 2 or prompt_ids.size(0) < 1:
        raise ValueError("generation requires at least one prompt row.")
    if prompt_attention_mask is None:
        prompt_attention_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    if prompt_attention_mask.shape != prompt_ids.shape:
        raise ValueError("prompt attention mask must align with prompt ids.")
    if audio_input_positions is not None and (
        audio_input_positions.dim() != 2
        or audio_input_positions.size(0) != prompt_ids.size(0)
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
        active_rows=torch.arange(batch_size, dtype=torch.long, device=prompt_ids.device),
        audio_input_positions=audio_input_positions,
        length=prompt_width,
    )


def _generation_loop_step(
    model: GenerationStepModel,
    state: _GenerationLoopState,
    *,
    batch_size: int,
    collect_audio_condition: bool,
    generation_token_ids: Tensor | None,
    generation_modality: Modality | None,
    use_cache: bool,
) -> GenerationStepResult:
    active_attention_mask = (
        state.attention_mask
        if state.active_rows.numel() == batch_size
        else state.attention_mask.index_select(0, state.active_rows)
    )
    output = model.generation_step(
        state.input_ids,
        attention_mask=active_attention_mask[:, : state.length],
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
    stop_token_id: int | None,
    generation_token_ids: Tensor | None,
    generation_modality: Modality | None,
    layout: Layout,
) -> Tensor:
    logits = logits[:, -1] / temperature
    if stop_token_id is not None and step < min_new_tokens:
        _suppress_stop(
            logits,
            stop_token_id,
            generation_token_ids,
            generation_modality,
            layout,
        )
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)
    return logits


def _sample_next_indices(logits: Tensor, do_sample: bool) -> Tensor:
    return (
        torch.distributions.Categorical(logits=logits).sample()
        if do_sample
        else logits.argmax(dim=-1)
    )


def _append_logprobs(
    logprob_steps: list[Tensor],
    logprob_mask_steps: list[Tensor],
    logits: Tensor,
    next_indices: Tensor,
    active_rows: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> None:
    step_logprobs = logits.log_softmax(dim=-1)
    active_token_logprobs = step_logprobs.gather(
        1,
        next_indices.unsqueeze(1),
    ).squeeze(1)
    token_logprobs = active_token_logprobs.new_zeros(batch_size)
    token_logprobs.index_copy_(0, active_rows, active_token_logprobs)
    token_logprob_mask = torch.zeros(
        batch_size,
        dtype=torch.bool,
        device=device,
    )
    token_logprob_mask.index_fill_(0, active_rows, True)
    logprob_steps.append(token_logprobs)
    logprob_mask_steps.append(token_logprob_mask)


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
    active_rows: Tensor,
    *,
    batch_size: int,
) -> None:
    if output.hidden_states is None:
        raise RuntimeError("model did not return generation hidden states.")
    codec_start, codec_end = model.runtime.codec_audio_range
    codec_tokens = next_ids.ge(codec_start) & next_ids.lt(codec_end)
    local_ids = (next_ids - codec_start).clamp(
        0, model.audio_token_frame_spans.numel() - 1
    )
    spans = model.audio_token_frame_spans.index_select(0, local_ids)
    step_spans = spans.new_zeros(batch_size)
    step_spans.index_copy_(
        0,
        active_rows,
        spans.masked_fill(~codec_tokens, 0),
    )
    span_steps.append(step_spans)
    active_condition = output.hidden_states[-1][:, -1]
    step_condition = active_condition.new_zeros(
        batch_size, active_condition.size(-1)
    )
    step_condition.index_copy_(0, active_rows, active_condition)
    condition_steps.append(step_condition)


def _advance_generation_state(
    state: _GenerationLoopState,
    model: GenerationStepModel,
    output: GenerationStepResult,
    next_ids: Tensor,
    *,
    stop_token_id: int | None,
    use_cache: bool,
    batch_size: int,
) -> bool:
    _write_generated_tokens(state, next_ids, stop_token_id, batch_size=batch_size)
    continuing_rows = _continuing_rows(next_ids, stop_token_id)
    if continuing_rows is not None and continuing_rows.numel() == 0:
        return False
    if continuing_rows is not None:
        state.active_rows = state.active_rows.index_select(0, continuing_rows)
    if use_cache:
        _advance_cached_state(state, model, output, next_ids, continuing_rows)
    else:
        _advance_full_recompute_state(state, continuing_rows)
    return True


def _write_generated_tokens(
    state: _GenerationLoopState,
    next_ids: Tensor,
    stop_token_id: int | None,
    *,
    batch_size: int,
) -> None:
    if state.active_rows.numel() == batch_size:
        state.generated[:, state.length] = next_ids
    else:
        if stop_token_id is None:
            raise RuntimeError("generation rows became inactive without a stop token.")
        state.generated[:, state.length] = stop_token_id
        state.generated[state.active_rows, state.length] = next_ids
    state.length += 1
    state.attention_mask[state.active_rows, state.length - 1] = True


def _continuing_rows(next_ids: Tensor, stop_token_id: int | None) -> Tensor | None:
    if stop_token_id is None:
        return None
    return next_ids.ne(stop_token_id).nonzero(as_tuple=False).flatten()


def _advance_cached_state(
    state: _GenerationLoopState,
    model: GenerationStepModel,
    output: GenerationStepResult,
    next_ids: Tensor,
    continuing_rows: Tensor | None,
) -> None:
    state.audio_input_positions = None
    state.past_key_values = output.past_key_values
    if state.past_key_values is None:
        raise RuntimeError("backbone did not return a generation cache.")
    state.audio_output_past = output.audio_output_past
    if continuing_rows is not None and continuing_rows.numel() != next_ids.numel():
        state.past_key_values.batch_select_indices(continuing_rows)
        state.audio_output_past = model.audio_output_adapter_batch_select(
            state.audio_output_past,
            continuing_rows,
        )
    state.input_ids = (
        next_ids
        if continuing_rows is None
        else next_ids.index_select(0, continuing_rows)
    ).unsqueeze(-1)


def _advance_full_recompute_state(
    state: _GenerationLoopState,
    continuing_rows: Tensor | None,
) -> None:
    state.audio_output_past = None
    if state.audio_input_positions is not None and continuing_rows is not None:
        state.audio_input_positions = state.audio_input_positions.index_select(
            0, continuing_rows
        )
    state.input_ids = (
        state.generated[:, : state.length]
        if continuing_rows is None
        else state.generated.index_select(0, state.active_rows)[:, : state.length]
    )


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
        do_sample=do_sample,
        use_cache=use_cache,
        collect_audio_condition=collect_audio_condition,
        min_new_tokens=min_new_tokens,
    )
    return output.sequences, output.audio_condition, output.frame_spans


def _suppress_stop(
    logits: Tensor,
    stop_token_id: int,
    generation_token_ids: Tensor | None,
    generation_modality: Modality | None,
    layout: Layout,
) -> None:
    if generation_token_ids is not None:
        stop = generation_token_ids.eq(stop_token_id)
        if not bool(stop.any()):
            return
        logits[:, stop] = float("-inf")
    elif generation_modality is not None:
        start, end = layout.blocks[generation_modality.value]
        if not start <= stop_token_id < end:
            return
        logits[:, stop_token_id - start] = float("-inf")
    else:
        if not 0 <= stop_token_id < logits.size(1):
            return
        logits[:, stop_token_id] = float("-inf")
    if not bool(torch.isfinite(logits).any(dim=1).all()):
        raise ValueError("minimum generation length left no non-stop token to sample.")


def generate_flattened_sequence(
    model: GenerationStepModel,
    prompt_ids: Tensor,
    *,
    codebook_ranges: Sequence[tuple[int, int]],
    codebook_token_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_attention_mask: Tensor | None,
    audio_input_positions: Tensor | None = None,
    do_sample: bool,
    use_cache: bool,
) -> Tensor:
    """Generate frame-aligned codebook blocks with the flattened codec grammar."""
    if prompt_ids.dim() != 2 or prompt_ids.size(0) < 1:
        raise ValueError("generation requires at least one prompt row.")
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("invalid generation parameters")
    if not codebook_ranges or len(codebook_ranges) != len(codebook_token_ids):
        raise ValueError("flattened generation requires one marker per codebook range.")
    if any(start < 0 or end <= start for start, end in codebook_ranges):
        raise ValueError("flattened generation codebook ranges must be non-empty.")
    minimum_tokens = 2 * len(codebook_ranges) + 1
    if max_new_tokens < minimum_tokens:
        raise ValueError(
            "flattened full sequence max_new_tokens must include codebook "
            f"markers, one payload per codebook, and EOA ({minimum_tokens} minimum)."
        )
    if prompt_attention_mask is None:
        prompt_attention_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    if prompt_attention_mask.shape != prompt_ids.shape:
        raise ValueError("prompt attention mask must align with prompt ids.")
    if not bool(prompt_attention_mask.any(dim=1).all()):
        raise ValueError("each generation prompt must contain at least one token.")

    audio_start, _ = model.runtime.codec_audio_range
    if len(codebook_ranges) == 1:
        prefix = prompt_ids.new_tensor([audio_start + codebook_token_ids[0]])
        batch_prefix = prefix.unsqueeze(0).expand(prompt_ids.size(0), -1)
        prefixed = torch.cat((prompt_ids, batch_prefix), dim=1)
        prefix_mask = torch.ones_like(batch_prefix, dtype=torch.bool)
        attention = torch.cat((prompt_attention_mask, prefix_mask), dim=1)
        start, end = codebook_ranges[0]
        allowed = (
            *range(audio_start + start, audio_start + end),
            model.runtime.eoa_token_id,
        )
        generated, _, _ = generate_sequence(
            model,
            prefixed,
            max_new_tokens=max_new_tokens - prefix.numel(),
            temperature=temperature,
            top_p=top_p,
            prompt_attention_mask=attention,
            audio_input_positions=(
                None
                if audio_input_positions is None
                else torch.cat(
                    (
                        audio_input_positions,
                        audio_input_positions.new_full(
                            (audio_input_positions.size(0), prefix.numel()),
                            -1,
                        ),
                    ),
                    dim=1,
                )
            ),
            stop_token_id=model.runtime.eoa_token_id,
            generation_modality=None,
            allowed_token_ids=allowed,
            do_sample=do_sample,
            use_cache=use_cache,
            collect_audio_condition=False,
            min_new_tokens=1,
        )
        return generated

    rows = [
        _generate_flattened_row(
            model,
            prompt_ids[row : row + 1],
            prompt_attention_mask=prompt_attention_mask[row : row + 1],
            audio_input_positions=(
                None
                if audio_input_positions is None
                else audio_input_positions[row : row + 1]
            ),
            codebook_ranges=codebook_ranges,
            codebook_token_ids=codebook_token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=use_cache,
        )
        for row in range(prompt_ids.size(0))
    ]
    width = max(row.size(1) for row in rows)
    output = prompt_ids.new_full((len(rows), width), model.runtime.eoa_token_id)
    for index, row in enumerate(rows):
        output[index, : row.size(1)] = row[0]
    return output


def _generate_flattened_row(
    model: GenerationStepModel,
    prompt_ids: Tensor,
    *,
    prompt_attention_mask: Tensor,
    audio_input_positions: Tensor | None,
    codebook_ranges: Sequence[tuple[int, int]],
    codebook_token_ids: Sequence[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    use_cache: bool,
) -> Tensor:
    row = _RowGenerator(
        model,
        prompt_ids,
        attention_mask=prompt_attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        use_cache=use_cache,
        grammar="flattened full sequence",
        audio_input_positions=audio_input_positions,
    )

    audio_start, _ = model.runtime.codec_audio_range

    def local(value: int) -> Tensor:
        return prompt_ids.new_tensor([audio_start + value], dtype=torch.long)

    def range_ids(value: tuple[int, int]) -> Tensor:
        start, end = value
        return torch.arange(
            audio_start + start,
            audio_start + end,
            device=prompt_ids.device,
            dtype=torch.long,
        )

    row.step(local(codebook_token_ids[0]))
    first_ids = range_ids(codebook_ranges[0])
    row.step(first_ids)
    frame_count = 1
    transition = local(codebook_token_ids[1])
    while row.step(torch.cat((first_ids, transition))) != int(transition.item()):
        frame_count += 1

    for codebook in range(1, len(codebook_ranges)):
        if codebook > 1:
            row.step(local(codebook_token_ids[codebook]))
        payload_ids = range_ids(codebook_ranges[codebook])
        for _ in range(frame_count):
            row.step(payload_ids)
    row.step(prompt_ids.new_tensor([model.runtime.eoa_token_id], dtype=torch.long))
    return row.sequence


def generate_bicodec_sequence(
    model: GenerationStepModel,
    prompt_ids: Tensor,
    *,
    tokenizer: BiCodecAudioTokenizer,
    streams: Sequence[AudioStream],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    prompt_attention_mask: Tensor | None,
    audio_input_positions: Tensor | None = None,
    do_sample: bool,
    use_cache: bool,
) -> Tensor:
    """Generate one constrained structured BiCodec sequence per prompt row."""
    streams = _audio_streams(streams)
    if prompt_ids.dim() != 2 or prompt_ids.size(0) < 1:
        raise ValueError("generation requires at least one prompt row.")
    if max_new_tokens < 1 or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("invalid generation parameters")
    if prompt_attention_mask is None:
        prompt_attention_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    if prompt_attention_mask.shape != prompt_ids.shape:
        raise ValueError("prompt attention mask must align with prompt ids.")

    rows = []
    for row in range(prompt_ids.size(0)):
        rows.append(
            _generate_bicodec_row(
                model,
                prompt_ids[row : row + 1],
                prompt_attention_mask=prompt_attention_mask[row : row + 1],
                audio_input_positions=(
                    None
                    if audio_input_positions is None
                    else audio_input_positions[row : row + 1]
                ),
                tokenizer=tokenizer,
                streams=streams,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                use_cache=use_cache,
            )
        )
    width = max(row.size(1) for row in rows)
    output = prompt_ids.new_full((len(rows), width), model.runtime.eoa_token_id)
    for index, row in enumerate(rows):
        output[index, : row.size(1)] = row[0]
    return output


def _generate_bicodec_row(
    model: GenerationStepModel,
    prompt_ids: Tensor,
    *,
    prompt_attention_mask: Tensor,
    audio_input_positions: Tensor | None,
    tokenizer: BiCodecAudioTokenizer,
    streams: tuple[AudioStream, ...],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    use_cache: bool,
) -> Tensor:
    row = _RowGenerator(
        model,
        prompt_ids,
        attention_mask=prompt_attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        use_cache=use_cache,
        grammar="BiCodec full sequence",
        audio_input_positions=audio_input_positions,
    )

    audio_start, _ = model.runtime.audio_head_range

    def local(value: int) -> Tensor:
        return prompt_ids.new_tensor([audio_start + value], dtype=torch.long)

    def range_ids(start: int, size: int) -> Tensor:
        return torch.arange(
            audio_start + start,
            audio_start + start + size,
            device=prompt_ids.device,
            dtype=torch.long,
        )

    if AudioStream.ACOUSTIC in streams:
        row.step(local(tokenizer.acoustic_token_id))
        for _ in range(tokenizer.acoustic_unit_length):
            for start, end in tokenizer.acoustic_token_ranges:
                row.step(range_ids(start, end - start))

    if AudioStream.SEMANTIC in streams:
        row.step(local(tokenizer.semantic_token_id))
        semantic_ids = range_ids(*tokenizer.semantic_token_range)
        row.step(semantic_ids)
        end = local(tokenizer.end_token_id)
        while row.step(torch.cat((semantic_ids, end))) != int(end.item()):
            pass
    else:
        row.step(local(tokenizer.end_token_id))
    row.step(prompt_ids.new_tensor([model.runtime.eoa_token_id], dtype=torch.long))
    return row.sequence


def _audio_streams(streams: Sequence[AudioStream]) -> tuple[AudioStream, ...]:
    values = tuple(streams)
    if not values:
        raise ValueError("BiCodec generation requires at least one output stream.")
    if any(not isinstance(stream, AudioStream) for stream in values):
        raise TypeError("BiCodec generation streams must contain AudioStream values.")
    unknown = set(values) - {AudioStream.ACOUSTIC, AudioStream.SEMANTIC}
    if unknown:
        labels = ", ".join(sorted(stream.value for stream in unknown))
        raise ValueError(f"BiCodec generation streams do not support: {labels}.")
    return tuple(
        stream
        for stream in (AudioStream.ACOUSTIC, AudioStream.SEMANTIC)
        if stream in values
    )


def _rows(value: Tensor | None, rows: Tensor) -> Tensor | None:
    if value is None or value.size(0) == rows.numel():
        return value
    return value.index_select(0, rows)


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
