from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from anydataset.types import Modality
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .decode import decode_generated_audio, decode_generated_semantic
from .protocol import AcousticFeatureGenerator, TokenGenerator
from .types import AcousticPrompt, AudioOutput, Request, Result


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
        task = request["task"]
        acoustic_prompt = request["acoustic_prompt"]
        if (
            acoustic_prompt is not None
            and task.source_modality is not Modality.AUDIO
        ):
            raise ValueError(
                f"{task.value} does not accept a source acoustic prompt."
            )
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
        features = None
        if modality is Modality.AUDIO and model.runtime.codec.acoustic_codebook_sizes:
            if not isinstance(model, AcousticFeatureGenerator):
                raise TypeError(
                    "a codec with acoustic codebooks requires an "
                    "AcousticFeatureGenerator."
                )
            sequence, features = model.generate_audio_features(
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

        for row, (result_index, _) in enumerate(group):
            token_ids = _response(sequence[row], prompt.size(1), stop_token_id)
            if modality is Modality.TEXT:
                results[result_index] = Result(response_ids=token_ids, audio=None)
                continue
            row_features = None if features is None else features[row]
            if row_features is None:
                waveform = decode_generated_semantic(
                    token_ids[None],
                    codec=model.runtime.codec,
                    audio_tokenizer=model.runtime.audio_tokenizer,
                    audio_token_range=model.runtime.codec_audio_range,
                )[0]
            else:
                row_features = row_features[: _frame_count(token_ids, model)]
                waveform = decode_generated_audio(
                    token_ids[None],
                    row_features[None],
                    codec=model.runtime.codec,
                    audio_tokenizer=model.runtime.audio_tokenizer,
                    audio_token_range=model.runtime.codec_audio_range,
                )[0]
            results[result_index] = Result(
                response_ids=token_ids,
                audio=AudioOutput(
                    features=row_features,
                    waveform=waveform,
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
        [value["codes"].to(device=device) for value in values], batch_first=True
    )
    token_positions = pad_sequence(
        [
            value["token_positions"].to(device=device)
            + width
            - prompts[row].numel()
            for row, value in enumerate(values)
        ],
        batch_first=True,
        padding_value=-1,
    )
    mask = token_positions.ge(0)
    return prompt, prompt_mask, codes, token_positions, mask


def _frame_count(token_ids: Tensor, model: TokenGenerator) -> int:
    local = token_ids - model.runtime.codec_audio_range[0]
    spans = model.runtime.audio_tokenizer.frame_spans(local)
    return int(torch.as_tensor(spans).sum().item())
