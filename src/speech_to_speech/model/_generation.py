from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from anydataset.types import Modality
from anytrain.idspace import Layout
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

from ._sampling import top_p_filter
from .protocol import TokenModelRuntime


class GenerationStepModel(Protocol):
    layout: Layout
    runtime: TokenModelRuntime
    audio_token_frame_spans: nn.Buffer

    def generation_step(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor,
        acoustic_prompt_codes: Tensor | None,
        acoustic_prompt_positions: Tensor | None,
        acoustic_prompt_mask: Tensor | None,
        output_hidden_states: bool,
        token_ids: Tensor | None,
        modality: Modality | None,
        past_key_values: Cache | None,
        use_cache: bool,
    ) -> CausalLMOutputWithPast: ...


def generate_sequence(
    model: GenerationStepModel,
    prompt_ids: Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    acoustic_prompt_codes: Tensor | None,
    acoustic_prompt_positions: Tensor | None,
    acoustic_prompt_mask: Tensor | None,
    prompt_attention_mask: Tensor | None,
    stop_token_id: int | None,
    generation_modality: Modality | None,
    allowed_token_ids: Sequence[int] | Tensor | None,
    do_sample: bool,
    use_cache: bool,
    collect_audio_condition: bool,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    if max_new_tokens < 0 or temperature <= 0 or not 0 < top_p <= 1:
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
    if not bool(prompt_attention_mask.any(dim=1).all()):
        raise ValueError("each generation prompt must contain at least one token.")

    generation_token_ids = _generation_token_ids(
        allowed_token_ids,
        prompt_ids,
        model.layout,
    )
    prompt_width = prompt_ids.size(1)
    capacity = prompt_width + max_new_tokens
    generated = prompt_ids.new_empty(prompt_ids.size(0), capacity)
    generated[:, :prompt_width] = prompt_ids
    attention_mask = torch.zeros_like(generated, dtype=torch.bool)
    attention_mask[:, :prompt_width] = prompt_attention_mask
    length = prompt_width
    input_ids = generated[:, :length]
    past_key_values: Cache | None = None
    condition_steps: list[Tensor] = []
    span_steps: list[Tensor] = []
    batch_size = prompt_ids.size(0)
    active_rows = torch.arange(batch_size, dtype=torch.long, device=prompt_ids.device)
    for _ in range(max_new_tokens):
        inject_acoustic = past_key_values is None
        active_attention_mask = (
            attention_mask
            if active_rows.numel() == batch_size
            else attention_mask.index_select(0, active_rows)
        )
        output = model.generation_step(
            input_ids,
            attention_mask=active_attention_mask[:, :length],
            acoustic_prompt_codes=(
                _rows(acoustic_prompt_codes, active_rows) if inject_acoustic else None
            ),
            acoustic_prompt_positions=(
                _rows(acoustic_prompt_positions, active_rows)
                if inject_acoustic
                else None
            ),
            acoustic_prompt_mask=(
                _rows(acoustic_prompt_mask, active_rows) if inject_acoustic else None
            ),
            output_hidden_states=collect_audio_condition,
            token_ids=generation_token_ids,
            modality=generation_modality,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        if output.logits is None:
            raise RuntimeError("model did not return generation logits.")
        logits = output.logits[:, -1] / temperature
        if top_p < 1.0:
            logits = top_p_filter(logits, top_p)
        next_indices = (
            torch.distributions.Categorical(logits=logits).sample()
            if do_sample
            else logits.argmax(dim=-1)
        )
        if generation_token_ids is not None:
            next_ids = generation_token_ids.index_select(0, next_indices)
        elif generation_modality is not None:
            start, _ = model.layout.blocks[generation_modality.value]
            next_ids = next_indices + start
        else:
            next_ids = next_indices

        if collect_audio_condition:
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

        if active_rows.numel() == batch_size:
            generated[:, length] = next_ids
        else:
            if stop_token_id is None:
                raise RuntimeError(
                    "generation rows became inactive without a stop token."
                )
            generated[:, length] = stop_token_id
            generated[active_rows, length] = next_ids
        length += 1
        attention_mask[active_rows, length - 1] = True

        continuing_rows: Tensor | None = None
        if stop_token_id is not None:
            continuing = next_ids.ne(stop_token_id)
            if not bool(continuing.any()):
                break
            continuing_rows = continuing.nonzero(as_tuple=False).flatten()
            active_rows = active_rows.index_select(0, continuing_rows)
        if use_cache:
            past_key_values = output.past_key_values
            if past_key_values is None:
                raise RuntimeError("backbone did not return a generation cache.")
            if (
                continuing_rows is not None
                and continuing_rows.numel() != next_ids.numel()
            ):
                past_key_values.batch_select_indices(continuing_rows)
            input_ids = (
                next_ids
                if continuing_rows is None
                else next_ids.index_select(0, continuing_rows)
            ).unsqueeze(-1)
        else:
            input_ids = (
                generated[:, :length]
                if continuing_rows is None
                else generated.index_select(0, active_rows)[:, :length]
            )

    generated = generated[:, :length]
    if not span_steps:
        return generated, None, None
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
    return generated, condition, frame_spans


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
