from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from anydataset.types import Modality
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .._tensor import is_signed_integer_dtype
from ..task import Task
from .decode import decode_generated_audio, decode_generated_semantic
from .protocol import AcousticFeatureGenerator, TokenGenerator
from .types import AcousticGeneration, AcousticPrompt, AudioOutput, Request, Result


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
    groups: dict[tuple[Modality, bool], list[tuple[int, Request]]] = {}
    for index, request in enumerate(requests):
        _validate_request(request, model)
        task = request["task"]
        acoustic_prompt = request["acoustic_prompt"]
        key = task.target_modality, acoustic_prompt is not None
        groups.setdefault(key, []).append((index, request))

    for (modality, _), group in groups.items():
        prompt, prompt_mask, acoustic_codes, token_positions, acoustic_mask = _inputs(
            [request for _, request in group], model, device
        )
        stop_token_id = (
            model.runtime.eoa_token_id
            if modality is Modality.AUDIO
            else model.runtime.eos_token_id
        )
        acoustic_generation: AcousticGeneration | None = None
        if modality is Modality.AUDIO and model.runtime.codec.acoustic_codebook_sizes:
            if not isinstance(model, AcousticFeatureGenerator):
                raise TypeError(
                    "a codec with acoustic codebooks requires an "
                    "AcousticFeatureGenerator."
                )
            acoustic_generation = model.generate_audio_features(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                acoustic_prompt_codes=acoustic_codes,
                acoustic_prompt_positions=token_positions,
                acoustic_prompt_mask=acoustic_mask,
                prompt_attention_mask=prompt_mask,
                do_sample=do_sample,
                use_cache=use_cache,
            )
            sequence = acoustic_generation["sequence"]
        else:
            sequence = model.generate_tokens(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                acoustic_prompt_codes=acoustic_codes,
                acoustic_prompt_positions=token_positions,
                acoustic_prompt_mask=acoustic_mask,
                prompt_attention_mask=prompt_mask,
                stop_token_id=stop_token_id,
                generation_modality=modality,
                do_sample=do_sample,
                use_cache=use_cache,
            )

        responses = [
            _response(sequence[row], prompt.size(1), stop_token_id)
            for row in range(len(group))
        ]
        if modality is Modality.TEXT:
            for token_ids, (result_index, _) in zip(responses, group):
                results[result_index] = Result(response_ids=token_ids, audio=None)
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
            results[result_index] = Result(
                response_ids=responses[row],
                audio=AudioOutput(
                    features=row_features[row],
                    waveform=waveforms[row],
                    sample_rate=model.runtime.codec.sample_rate,
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
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
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

    acoustic = [request["acoustic_prompt"] for request in requests]
    if all(value is None for value in acoustic):
        return prompt, prompt_mask, None, None, None
    if any(value is None for value in acoustic):
        raise ValueError("a generation batch must use one source modality.")
    values = cast(list[AcousticPrompt], acoustic)
    codes = pad_sequence(
        [value["codes"].to(device=device, dtype=torch.long) for value in values],
        batch_first=True,
        padding_value=-1,
    )
    token_positions = pad_sequence(
        [
            value["token_positions"].to(device=device, dtype=torch.long)
            + width
            - prompts[row].numel()
            for row, value in enumerate(values)
        ],
        batch_first=True,
        padding_value=-1,
    )
    mask = token_positions.ge(0)
    return prompt, prompt_mask, codes, token_positions, mask


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

    acoustic_prompt = request["acoustic_prompt"]
    if acoustic_prompt is None:
        return
    if task.source_modality is not Modality.AUDIO:
        raise ValueError(f"{task.value} does not accept a source acoustic prompt.")

    sizes = model.runtime.codec.acoustic_codebook_sizes
    if not sizes:
        raise ValueError(
            "a codec without acoustic codebooks does not accept an acoustic prompt."
        )
    codes = _integer_tensor(
        acoustic_prompt["codes"], "acoustic prompt codes", dimensions=2
    )
    if codes.size(0) == 0:
        raise ValueError("acoustic prompt codes must contain at least one frame.")
    if codes.size(1) != len(sizes):
        raise ValueError("acoustic prompt codes must match the codec codebooks.")
    limits = torch.tensor(sizes, device=codes.device, dtype=torch.long)
    if bool(((codes < 0) | (codes >= limits)).any()):
        raise ValueError("acoustic prompt code is outside its codec codebook.")

    positions = _integer_tensor(
        acoustic_prompt["token_positions"],
        "acoustic prompt positions",
        dimensions=1,
    )
    if positions.numel() != codes.size(0):
        raise ValueError(
            "acoustic prompt codes and positions must share the frame axis."
        )
    if bool(((positions < 0) | (positions >= prompt.numel())).any()):
        raise ValueError("acoustic prompt positions must point inside the prompt.")
    source_ids = prompt.index_select(
        0,
        positions.to(device=prompt.device, dtype=torch.long),
    )
    start, end = model.runtime.codec_audio_range
    if bool(((source_ids < start) | (source_ids >= end)).any()):
        raise ValueError(
            "acoustic prompt positions must point to codec-decodable audio tokens."
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
    token_lengths = [token_ids.numel() for token_ids in token_rows]
    local = torch.cat(token_rows) - model.runtime.codec_audio_range[0]
    spans = torch.as_tensor(
        model.runtime.audio_tokenizer.frame_spans(local),
        device=local.device,
        dtype=torch.long,
    )
    if spans.shape != local.shape:
        raise ValueError("audio token frame spans must align with generated tokens.")
    return torch.stack([values.sum() for values in spans.split(token_lengths)])


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
            decoded = decode_generated_semantic(
                token_batch,
                codec=model.runtime.codec,
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
                codec=model.runtime.codec,
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
