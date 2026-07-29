from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from anydataset.types import Modality
from anytrain.codec import SemanticAcousticCodes
from torch import Tensor

from .._tensor import is_signed_integer_dtype
from ..runtime import AudioRepresentation
from ..runtime.audio_tokenizer import BiCodecAudioTokenizer
from ..audio_route import PromptSource, StreamSource
from ..runtime.types import (
    acoustic_codec,
    codec_sample_rate,
    frame_codec,
    structured_codec,
)
from ..task import Task
from .decode import (
    decode_generated_audio,
    decode_generated_bicodec_full,
    decode_generated_bicodec_route,
    decode_generated_frame_codes,
    decode_generated_semantic,
)
from .protocol import AcousticFeatureGeneration, FullCodecSequenceGenerator, TokenGenerator
from .types import AcousticGeneration, AudioOutput, Request, Result


@torch.no_grad()
def generate_responses(
    requests: Sequence[Request],
    model: TokenGenerator,
    *,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = True,
    use_cache: bool = True,
) -> list[Result]:
    """Generate batched responses grouped by target modality."""
    results: list[Result | None] = [None] * len(requests)
    device = model.backbone.get_input_embeddings().weight.device
    groups: dict[Modality, list[tuple[int, Request]]] = {}
    for index, request in enumerate(requests):
        _validate_request(request, model)
        task = request["task"]
        groups.setdefault(task.target_modality, []).append((index, request))

    for modality, group in groups.items():
        prompt, prompt_mask, audio_input_positions = _inputs(
            [request for _, request in group],
            model,
            device,
        )
        response_start = prompt.size(1)
        stop_token_id = (
            model.runtime.eoa_token_id
            if modality is Modality.AUDIO
            else model.runtime.eos_token_id
        )
        acoustic_generation: AcousticGeneration | None = None
        if (
            modality is Modality.AUDIO
            and model.runtime.acoustic_side_channel
            and isinstance(model, AcousticFeatureGeneration)
        ):
            if audio_input_positions is None:
                acoustic_generation = model.generate_audio_features(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    prompt_attention_mask=prompt_mask,
                    do_sample=do_sample,
                    use_cache=use_cache,
                )
            else:
                acoustic_generation = model.generate_audio_features(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    prompt_attention_mask=prompt_mask,
                    audio_input_positions=audio_input_positions,
                    do_sample=do_sample,
                    use_cache=use_cache,
                )
            sequence = acoustic_generation["sequence"]
        elif (
            modality is Modality.AUDIO
            and model.runtime.audio_representation
            is AudioRepresentation.FULL_CODEC_SEQUENCE
        ):
            if not isinstance(model, FullCodecSequenceGenerator):
                raise TypeError(
                    "full codec sequence requires constrained token generation."
                )
            if audio_input_positions is None:
                sequence = model.generate_full_codec_sequence(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    prompt_attention_mask=prompt_mask,
                    do_sample=do_sample,
                    use_cache=use_cache,
                )
            else:
                sequence = model.generate_full_codec_sequence(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    prompt_attention_mask=prompt_mask,
                    audio_input_positions=audio_input_positions,
                    do_sample=do_sample,
                    use_cache=use_cache,
                )
        else:
            if audio_input_positions is None:
                sequence = model.generate_tokens(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    prompt_attention_mask=prompt_mask,
                    stop_token_id=stop_token_id,
                    generation_modality=modality,
                    do_sample=do_sample,
                    use_cache=use_cache,
                )
            else:
                sequence = model.generate_tokens(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    prompt_attention_mask=prompt_mask,
                    audio_input_positions=audio_input_positions,
                    stop_token_id=stop_token_id,
                    generation_modality=modality,
                    do_sample=do_sample,
                    use_cache=use_cache,
                )

        responses = [
            _response(sequence[row], response_start, stop_token_id)
            for row in range(len(group))
        ]
        if modality is Modality.TEXT:
            for token_ids, (result_index, _) in zip(responses, group):
                results[result_index] = Result(response_ids=token_ids, audio=None)
            continue

        route = model.runtime.audio_route
        tokenizer = model.runtime.audio_tokenizer
        if route is not None and isinstance(tokenizer, BiCodecAudioTokenizer):
            codec = structured_codec(model.runtime.codec)
            for token_ids, (result_index, request) in zip(responses, group):
                waveform, codes = decode_generated_bicodec_route(
                    token_ids,
                    request.get("audio_context"),
                    route=route,
                    codec=codec,
                    audio_tokenizer=tokenizer,
                    audio_token_range=model.runtime.codec_audio_range,
                )
                results[result_index] = Result(
                    response_ids=token_ids,
                    audio=AudioOutput(
                        features=None,
                        codes=codes,
                        waveform=waveform,
                        sample_rate=codec_sample_rate(codec),
                    ),
                )
            continue

        if acoustic_generation is None:
            features = None
            frame_counts = _frame_counts(responses, model)
        else:
            features = acoustic_generation["features"]
            frame_counts = acoustic_generation["frame_counts"]
        row_features, waveforms = _decode_rows(
            responses,
            features,
            frame_counts,
            model,
        )
        for row, (result_index, _) in enumerate(group):
            decoder = (
                model.runtime.codec
                if model.runtime.audio_representation is AudioRepresentation.FULL_CODEC_SEQUENCE
                else model.runtime.semantic_codec
                if acoustic_generation is None
                else model.runtime.codec
            )
            results[result_index] = Result(
                response_ids=responses[row],
                audio=AudioOutput(
                    features=row_features[row],
                    codes=None,
                    waveform=waveforms[row],
                    sample_rate=codec_sample_rate(decoder),
                ),
            )

    if any(result is None for result in results):
        raise RuntimeError("generation did not produce every requested result.")
    return cast(list[Result], results)


def _response(sequence: Tensor, prompt_length: int, stop_token_id: int) -> Tensor:
    response = sequence[prompt_length:]
    stops = response.eq(stop_token_id).nonzero()
    if stops.numel():
        return response[: int(stops[0].item())]
    return response


def _inputs(
    requests: list[Request],
    model: TokenGenerator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor | None]:
    prompts = [request["prompt_ids"].to(device=device) for request in requests]
    width = max(prompt.numel() for prompt in prompts)
    prompt = torch.full(
        (len(prompts), width),
        model.runtime.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    prompt_mask = torch.zeros_like(prompt, dtype=torch.bool)
    for row, value in enumerate(prompts):
        prompt[row, -value.numel() :] = value
        prompt_mask[row, -value.numel() :] = True

    position_values = [request.get("audio_input_positions") for request in requests]
    if not any(value is not None for value in position_values):
        return prompt, prompt_mask, None
    width_frames = max(
        0 if value is None else int(value.numel()) for value in position_values
    )
    positions = torch.full(
        (len(requests), width_frames),
        -1,
        dtype=torch.long,
        device=device,
    )
    for row, value in enumerate(position_values):
        if value is not None and value.numel():
            offset = width - prompts[row].numel()
            positions[row, : value.numel()] = value.to(device=device) + offset
    return prompt, prompt_mask, positions


def _validate_request(request: Request, model: TokenGenerator) -> None:
    task = request["task"]
    if not isinstance(task, Task):
        raise TypeError("generation request task must be a Task.")
    prompt = _integer_tensor(request["prompt_ids"], "prompt ids", dimensions=1)
    if prompt.numel() == 0:
        raise ValueError("generation prompt must contain at least one token.")
    inside = torch.zeros_like(prompt, dtype=torch.bool)
    for start, end in model.runtime.layout.blocks.values():
        inside |= prompt.ge(start) & prompt.lt(end)
    if not bool(inside.all()):
        raise ValueError("prompt ids must belong to the runtime layout.")
    positions_value = request.get("audio_input_positions")
    if positions_value is not None:
        positions = _integer_tensor(
            positions_value,
            "audio input positions",
            dimensions=1,
        )
        if task.source_modality is not Modality.AUDIO:
            raise ValueError(
                "audio input positions require an audio-source generation task."
            )
        if bool((positions < 0).any()) or bool(
            (positions >= prompt.numel()).any()
        ):
            raise ValueError("audio input positions must be valid prompt positions.")
        if positions.numel() != torch.unique(positions).numel():
            raise ValueError("audio input positions must not repeat positions.")
        codec_start, codec_end = model.runtime.codec_audio_range
        selected = prompt.index_select(0, positions.to(device=prompt.device))
        if bool((selected < codec_start).any()) or bool(
            (selected >= codec_end).any()
        ):
            raise ValueError(
                "audio input positions must point to visible codec audio payload tokens."
            )
    route = model.runtime.audio_route
    context = request.get("audio_context")
    if context is not None and not isinstance(context, SemanticAcousticCodes):
        raise TypeError("generation audio context must be SemanticAcousticCodes.")
    if task.target_modality is Modality.TEXT:
        if context is not None:
            raise ValueError("text generation requests cannot include audio context.")
        return
    if route is None:
        if context is not None:
            raise ValueError(
                "generation requests without an audio route cannot include audio context."
            )
        return

    prompt_streams = route.prompt.canonical_streams
    requires_context = (
        route.prompt.source is PromptSource.REFERENCE and bool(prompt_streams)
    ) or (
        route.decode.semantic is StreamSource.PROMPT
        or route.decode.acoustic is StreamSource.PROMPT
    )
    if requires_context and context is None:
        raise ValueError("audio route requires structured prompt audio context.")
    if not prompt_streams:
        if context is not None:
            raise ValueError(
                "audio route without prompt streams cannot include audio context."
            )
        return
    if context is None:
        raise ValueError("audio route prompt streams require audio context.")

    tokenizer = model.runtime.audio_tokenizer
    if not isinstance(tokenizer, BiCodecAudioTokenizer):
        raise TypeError("structured prompt streams require BiCodecAudioTokenizer.")
    local_ids = tokenizer.encode_streams(context, prompt_streams)
    global_ids = model.runtime.layout.to_global(
        Modality.AUDIO.value,
        local_ids,
    ).to(device=prompt.device)
    expected = torch.cat(
        (
            prompt.new_tensor([model.runtime.boa_token_id]),
            global_ids,
            prompt.new_tensor(
                [model.runtime.eoa_token_id, model.runtime.boa_token_id]
            ),
        )
    )
    if prompt.numel() < expected.numel() or not torch.equal(
        prompt[-expected.numel() :],
        expected,
    ):
        raise ValueError(
            "generation audio context does not serialize to the prompt suffix."
        )



def _integer_tensor(value: object, name: str, *, dimensions: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if not is_signed_integer_dtype(value.dtype):
        raise TypeError(f"{name} must contain integer ids using a signed dtype.")
    if value.dim() != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions.")
    return value


def _frame_counts(token_rows: list[Tensor], model: TokenGenerator) -> Tensor:
    if any(token_ids.numel() == 0 for token_ids in token_rows):
        raise ValueError("audio generation produced no codec-decodable tokens.")
    start, _ = model.runtime.codec_audio_range
    counts = []
    span_lookup = model.audio_token_frame_spans
    for token_ids in token_rows:
        local = token_ids - start
        if bool((local < 0).any()) or bool((local >= span_lookup.numel()).any()):
            raise ValueError("audio generation produced non-codec audio tokens.")
        spans = span_lookup.index_select(0, local.to(device=span_lookup.device))
        counts.append(spans.sum().to(device=local.device))
    return torch.stack(counts)


def _decode_rows(
    token_rows: list[Tensor],
    features: Tensor | None,
    frame_counts: Tensor,
    model: TokenGenerator,
) -> tuple[list[Tensor | None], list[Tensor]]:
    frame_counts = _integer_tensor(
        frame_counts,
        "generated audio frame counts",
        dimensions=1,
    )
    if frame_counts.shape != (len(token_rows),):
        raise ValueError("generated audio frame counts must provide one value per row.")
    counts = frame_counts.detach().cpu().tolist()
    if any(count < 1 for count in counts):
        raise ValueError("each audio generation row must contain at least one frame.")

    if features is not None:
        if features.dim() != 3 or features.size(0) != len(token_rows):
            raise ValueError(
                "generated acoustic features must have shape [batch, frames, dim]."
            )
        if any(count > features.size(1) for count in counts):
            raise ValueError("generated frame count exceeds acoustic feature padding.")
        row_features: list[Tensor | None] = [
            features[row, :count] for row, count in enumerate(counts)
        ]
    else:
        row_features = [None] * len(token_rows)

    groups: dict[tuple[int, int], list[int]] = {}
    for row, (token_ids, count) in enumerate(zip(token_rows, counts)):
        groups.setdefault((token_ids.numel(), count), []).append(row)

    waveforms: list[Tensor | None] = [None] * len(token_rows)
    for rows in groups.values():
        token_batch = torch.stack([token_rows[row] for row in rows])
        first_features = row_features[rows[0]]
        if first_features is None:
            if model.runtime.audio_representation is AudioRepresentation.FULL_CODEC_SEQUENCE:
                if model.runtime.structured_full_sequence:
                    tokenizer = model.runtime.audio_tokenizer
                    if not isinstance(tokenizer, BiCodecAudioTokenizer):
                        raise TypeError("BiCodec full sequence requires BiCodecAudioTokenizer.")
                    decoded = decode_generated_bicodec_full(
                        token_batch,
                        codec=structured_codec(model.runtime.codec),
                        audio_tokenizer=tokenizer,
                        audio_token_range=model.runtime.codec_audio_range,
                    )
                else:
                    decoded = decode_generated_frame_codes(
                        token_batch,
                        codec=frame_codec(model.runtime.codec),
                        audio_tokenizer=model.runtime.audio_tokenizer,
                        audio_token_range=model.runtime.codec_audio_range,
                    )
            else:
                decoded = decode_generated_semantic(
                    token_batch,
                    codec=model.runtime.semantic_codec,
                    audio_tokenizer=model.runtime.audio_tokenizer,
                    audio_token_range=model.runtime.codec_audio_range,
                )
        else:
            feature_batch = torch.stack(
                [cast(Tensor, row_features[row]) for row in rows]
            )
            decoded = decode_generated_audio(
                token_batch,
                feature_batch,
                codec=acoustic_codec(model.runtime.codec),
                audio_tokenizer=model.runtime.audio_tokenizer,
                audio_token_range=model.runtime.codec_audio_range,
            )
        if decoded.dim() < 1 or decoded.size(0) != len(rows):
            raise ValueError("codec decode must preserve the generation batch axis.")
        for row, waveform in zip(rows, decoded):
            waveforms[row] = waveform

    if any(waveform is None for waveform in waveforms):
        raise RuntimeError("codec decode did not produce every generation row.")
    return row_features, cast(list[Tensor], waveforms)
