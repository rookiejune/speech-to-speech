from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

from ..audio_route import AudioStream
from ..runtime.audio_tokenizer import BiCodecAudioTokenizer
from ._sampling import top_p_filter
from .protocol import TokenModelRuntime


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
    ) -> CausalLMOutputWithPast: ...


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
            self._current = self.sequence[:, -1:]
        else:
            self._current = self.sequence
        return next_id


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
    for step in range(max_new_tokens):
        active_attention_mask = (
            attention_mask
            if active_rows.numel() == batch_size
            else attention_mask.index_select(0, active_rows)
        )
        output = model.generation_step(
            input_ids,
            attention_mask=active_attention_mask[:, :length],
            output_hidden_states=collect_audio_condition,
            token_ids=generation_token_ids,
            modality=generation_modality,
            past_key_values=past_key_values,
            use_cache=use_cache,
            audio_input_positions=audio_input_positions,
        )
        if output.logits is None:
            raise RuntimeError("model did not return generation logits.")
        logits = output.logits[:, -1] / temperature
        if stop_token_id is not None and step < min_new_tokens:
            _suppress_stop(
                logits,
                stop_token_id,
                generation_token_ids,
                generation_modality,
                model.layout,
            )
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
            audio_input_positions = None
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
            if audio_input_positions is not None and continuing_rows is not None:
                audio_input_positions = audio_input_positions.index_select(
                    0, continuing_rows
                )
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
    codec_token_id: int,
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
    minimum_tokens = 2 * len(codebook_ranges) + 2
    if max_new_tokens < minimum_tokens:
        raise ValueError(
            "flattened full sequence max_new_tokens must include codec/codebook "
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
        prefix = prompt_ids.new_tensor(
            [
                audio_start + codec_token_id,
                audio_start + codebook_token_ids[0],
            ]
        )
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
            max_new_tokens=max_new_tokens - prefix.numel() - 1,
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
        continuation = generated[:, prefixed.size(1) :]
        if bool(continuation.eq(model.runtime.eoa_token_id).any(dim=1).all()):
            return generated
        return torch.cat(
            (
                generated,
                generated.new_full(
                    (generated.size(0), 1),
                    model.runtime.eoa_token_id,
                ),
            ),
            dim=1,
        )

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
            codec_token_id=codec_token_id,
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
    codec_token_id: int,
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

    row.step(local(codec_token_id))
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

    row.step(local(tokenizer.codec_token_id))
    if AudioStream.GLOBAL in streams:
        row.step(local(tokenizer.global_token_id))
        for _ in range(tokenizer.global_unit_length):
            for start, end in tokenizer.global_token_ranges:
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
    normalized = tuple(
        AudioStream.GLOBAL if stream is AudioStream.ACOUSTIC else stream
        for stream in values
    )
    if len(normalized) != len(set(normalized)):
        raise ValueError(
            "BiCodec generation streams must not contain both global and legacy "
            "acoustic streams."
        )
    return tuple(
        stream
        for stream in (AudioStream.GLOBAL, AudioStream.SEMANTIC)
        if stream in normalized
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
