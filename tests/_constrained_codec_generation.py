from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers.cache_utils import Cache

from speech_to_speech.audio import AudioStream
from speech_to_speech.model._helper import top_p_filter
from speech_to_speech.model.generation import (
    GenerationEngine,
    GenerationOptions,
    GenerationRequest,
    GenerationStepModel,
)
from speech_to_speech.runtime.audio_tokenizer import BiCodecAudioTokenizer


@dataclass(frozen=True)
class _RowGenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    use_cache: bool
    grammar: str


@dataclass(frozen=True)
class _TokenIdSpace:
    prompt_ids: Tensor
    offset: int

    def token(self, value: int) -> Tensor:
        return self.prompt_ids.new_tensor([self.offset + value], dtype=torch.long)

    def range(self, start: int, end: int) -> Tensor:
        return torch.arange(
            self.offset + start,
            self.offset + end,
            device=self.prompt_ids.device,
            dtype=torch.long,
        )

    def sized_range(self, start: int, size: int) -> Tensor:
        return self.range(start, start + size)


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
        self._audio_head_past: object | None = None
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
            audio_head_past=self._audio_head_past,
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
            self._audio_head_past = output.audio_head_past
            self._current = self.sequence[:, -1:]
        else:
            self._current = self.sequence
        return next_id


def generate_marker_block_flattened_codec_sequence_for_test(
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
    """Legacy marker/block codec generator kept only for focused tests."""
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
            "flattened marker/block max_new_tokens must include codebook "
            f"markers, one payload per codebook, and EOA ({minimum_tokens} minimum)."
        )
    if prompt_attention_mask is None:
        prompt_attention_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    if prompt_attention_mask.shape != prompt_ids.shape:
        raise ValueError("prompt attention mask must align with prompt ids.")
    if not bool(prompt_attention_mask.any(dim=1).all()):
        raise ValueError("each generation prompt must contain at least one token.")

    if len(codebook_ranges) == 1:
        return _generate_single_codebook_flattened_sequence(
            model,
            prompt_ids,
            prompt_attention_mask,
            audio_input_positions,
            codebook_range=codebook_ranges[0],
            codebook_token_id=codebook_token_ids[0],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=use_cache,
        )

    config = _RowGenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        use_cache=use_cache,
        grammar="flattened marker/block test sequence",
    )
    return _generate_prompt_rows(
        prompt_ids,
        prompt_attention_mask,
        audio_input_positions,
        pad_token_id=model.runtime.eoa_token_id,
        generate=lambda row_ids, row_mask, row_positions: _generate_flattened_row(
            model,
            row_ids,
            prompt_attention_mask=row_mask,
            audio_input_positions=row_positions,
            codebook_ranges=codebook_ranges,
            codebook_token_ids=codebook_token_ids,
            config=config,
        ),
    )


def generate_marker_stream_bicodec_sequence_for_test(
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
    """Legacy marker/stream BiCodec generator kept only for focused tests."""
    streams = _audio_streams(streams)
    if prompt_ids.dim() != 2 or prompt_ids.size(0) < 1:
        raise ValueError("generation requires at least one prompt row.")
    if max_new_tokens < 1 or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("invalid generation parameters")
    if prompt_attention_mask is None:
        prompt_attention_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
    if prompt_attention_mask.shape != prompt_ids.shape:
        raise ValueError("prompt attention mask must align with prompt ids.")

    config = _RowGenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        use_cache=use_cache,
        grammar="BiCodec marker/stream test sequence",
    )
    return _generate_prompt_rows(
        prompt_ids,
        prompt_attention_mask,
        audio_input_positions,
        pad_token_id=model.runtime.eoa_token_id,
        generate=lambda row_ids, row_mask, row_positions: _generate_bicodec_row(
            model,
            row_ids,
            prompt_attention_mask=row_mask,
            audio_input_positions=row_positions,
            tokenizer=tokenizer,
            streams=streams,
            config=config,
        ),
    )


def _generate_single_codebook_flattened_sequence(
    model: GenerationStepModel,
    prompt_ids: Tensor,
    prompt_attention_mask: Tensor,
    audio_input_positions: Tensor | None,
    *,
    codebook_range: tuple[int, int],
    codebook_token_id: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    use_cache: bool,
) -> Tensor:
    audio_start, _ = model.runtime.codec_audio_range
    token_ids = _TokenIdSpace(prompt_ids, audio_start)
    prefix = token_ids.token(codebook_token_id)
    batch_prefix = prefix.unsqueeze(0).expand(prompt_ids.size(0), -1)
    prefixed = torch.cat((prompt_ids, batch_prefix), dim=1)
    prefix_mask = torch.ones_like(batch_prefix, dtype=torch.bool)
    attention = torch.cat((prompt_attention_mask, prefix_mask), dim=1)
    start, end = codebook_range
    allowed = (*range(audio_start + start, audio_start + end), model.runtime.eoa_token_id)
    output = GenerationEngine(model).generate(
        GenerationRequest(
            prompt_ids=prefixed,
            prompt_attention_mask=attention,
            audio_input_positions=_with_prefix_audio_positions(
                audio_input_positions,
                prefix.numel(),
            ),
            stop_token_id=model.runtime.eoa_token_id,
            allowed_token_ids=allowed,
        ),
        GenerationOptions(
            max_new_tokens=max_new_tokens - prefix.numel(),
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=use_cache,
            min_new_tokens=1,
        ),
    )
    return output.sequences


def _generate_flattened_row(
    model: GenerationStepModel,
    prompt_ids: Tensor,
    *,
    prompt_attention_mask: Tensor,
    audio_input_positions: Tensor | None,
    codebook_ranges: Sequence[tuple[int, int]],
    codebook_token_ids: Sequence[int],
    config: _RowGenerationConfig,
) -> Tensor:
    row = _row_generator(
        model,
        prompt_ids,
        attention_mask=prompt_attention_mask,
        audio_input_positions=audio_input_positions,
        config=config,
    )

    audio_start, _ = model.runtime.codec_audio_range
    token_ids = _TokenIdSpace(prompt_ids, audio_start)

    row.step(token_ids.token(codebook_token_ids[0]))
    first_ids = token_ids.range(*codebook_ranges[0])
    row.step(first_ids)
    frame_count = 1
    transition = token_ids.token(codebook_token_ids[1])
    while row.step(torch.cat((first_ids, transition))) != int(transition.item()):
        frame_count += 1

    for codebook in range(1, len(codebook_ranges)):
        if codebook > 1:
            row.step(token_ids.token(codebook_token_ids[codebook]))
        payload_ids = token_ids.range(*codebook_ranges[codebook])
        for _ in range(frame_count):
            row.step(payload_ids)
    row.step(prompt_ids.new_tensor([model.runtime.eoa_token_id], dtype=torch.long))
    return row.sequence


def _generate_bicodec_row(
    model: GenerationStepModel,
    prompt_ids: Tensor,
    *,
    prompt_attention_mask: Tensor,
    audio_input_positions: Tensor | None,
    tokenizer: BiCodecAudioTokenizer,
    streams: tuple[AudioStream, ...],
    config: _RowGenerationConfig,
) -> Tensor:
    row = _row_generator(
        model,
        prompt_ids,
        attention_mask=prompt_attention_mask,
        audio_input_positions=audio_input_positions,
        config=config,
    )

    audio_start, _ = model.runtime.audio_head_range
    token_ids = _TokenIdSpace(prompt_ids, audio_start)

    if AudioStream.GLOBAL in streams:
        row.step(token_ids.token(tokenizer.global_token_id))
        for _ in range(tokenizer.global_unit_length):
            for start, end in tokenizer.global_token_ranges:
                row.step(token_ids.sized_range(start, end - start))

    if AudioStream.SEMANTIC in streams:
        row.step(token_ids.token(tokenizer.semantic_token_id))
        semantic_ids = token_ids.range(*tokenizer.semantic_token_range)
        row.step(semantic_ids)
        end = token_ids.token(tokenizer.end_token_id)
        while row.step(torch.cat((semantic_ids, end))) != int(end.item()):
            pass
    else:
        row.step(token_ids.token(tokenizer.end_token_id))
    row.step(prompt_ids.new_tensor([model.runtime.eoa_token_id], dtype=torch.long))
    return row.sequence


def _row_generator(
    model: GenerationStepModel,
    prompt_ids: Tensor,
    *,
    attention_mask: Tensor,
    audio_input_positions: Tensor | None,
    config: _RowGenerationConfig,
) -> _RowGenerator:
    return _RowGenerator(
        model,
        prompt_ids,
        attention_mask=attention_mask,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=config.do_sample,
        use_cache=config.use_cache,
        grammar=config.grammar,
        audio_input_positions=audio_input_positions,
    )


def _generate_prompt_rows(
    prompt_ids: Tensor,
    prompt_attention_mask: Tensor,
    audio_input_positions: Tensor | None,
    *,
    pad_token_id: int,
    generate: Callable[[Tensor, Tensor, Tensor | None], Tensor],
) -> Tensor:
    rows = [
        generate(
            prompt_ids[row : row + 1],
            prompt_attention_mask[row : row + 1],
            None if audio_input_positions is None else audio_input_positions[row : row + 1],
        )
        for row in range(prompt_ids.size(0))
    ]
    width = max(row.size(1) for row in rows)
    output = prompt_ids.new_full((len(rows), width), pad_token_id)
    for index, row in enumerate(rows):
        output[index, : row.size(1)] = row[0]
    return output


def _with_prefix_audio_positions(
    audio_input_positions: Tensor | None,
    prefix_length: int,
) -> Tensor | None:
    if audio_input_positions is None:
        return None
    return torch.cat(
        (
            audio_input_positions,
            audio_input_positions.new_full(
                (audio_input_positions.size(0), prefix_length),
                -1,
            ),
        ),
        dim=1,
    )


def _audio_streams(streams: Sequence[AudioStream]) -> tuple[AudioStream, ...]:
    values = tuple(streams)
    if not values:
        raise ValueError("BiCodec generation requires at least one output stream.")
    if any(not isinstance(stream, AudioStream) for stream in values):
        raise TypeError("BiCodec generation streams must contain AudioStream values.")
    unknown = set(values) - {AudioStream.GLOBAL, AudioStream.SEMANTIC}
    if unknown:
        labels = ", ".join(sorted(stream.value for stream in unknown))
        raise ValueError(f"BiCodec generation streams do not support: {labels}.")
    return tuple(
        stream for stream in (AudioStream.GLOBAL, AudioStream.SEMANTIC) if stream in values
    )
