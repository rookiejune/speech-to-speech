from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import torch
from anydataset.types import Modality

from speech_to_speech.generation import Request, generate_responses
from speech_to_speech.generation.reporting import (
    allowed_values,
    hidden_last,
    hidden_layer_max_abs,
    selected_id,
    tensor_max_abs,
    top_logits,
)
from speech_to_speech.model import FlowModel


def run(
    model: FlowModel,
    request: Request,
    *,
    seed: int,
    max_new_tokens: int,
    use_cache: bool,
) -> dict[str, Any]:
    calls: list[dict[str, int | bool]] = []
    allowed_logits: list[torch.Tensor] = []
    original_step = model.generation_step

    def observed_step(input_ids, **kwargs):
        attention_mask = kwargs["attention_mask"]
        calls.append(
            {
                "input_tokens": int(input_ids.size(1)),
                "attention_tokens": int(attention_mask.size(1)),
                "has_past": kwargs.get("past_key_values") is not None,
                "has_acoustic_prompt": kwargs.get("acoustic_prompt_codes") is not None,
            }
        )
        output = original_step(input_ids, **kwargs)
        if output.logits is None:
            raise RuntimeError("generation step did not return logits.")
        ids = torch.as_tensor(
            model.runtime.audio_generation_allowed_ids,
            device=output.logits.device,
            dtype=torch.long,
        )
        requested_ids = kwargs.get("token_ids")
        modality = kwargs.get("modality")
        values = output.logits[0, -1]
        if requested_ids is None:
            if modality is not Modality.AUDIO:
                raise RuntimeError("generation smoke expected the audio output head.")
            audio_start, _ = model.layout.blocks[Modality.AUDIO.value]
            values = values.index_select(0, ids - audio_start)
        elif not torch.equal(requested_ids, ids):
            raise RuntimeError("generation used unexpected allowed token ids.")
        allowed_logits.append(values.detach().float().cpu())
        return output

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with patch.object(model, "generation_step", side_effect=observed_step):
        result = generate_responses(
            [request],
            model,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
        )[0]
    torch.cuda.synchronize()
    return {
        "result": result,
        "calls": calls,
        "allowed_logits": allowed_logits,
        "allowed_ids": model.runtime.audio_generation_allowed_ids,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }


@torch.no_grad()
def second_step(
    model: FlowModel,
    request: Request,
) -> dict[str, Any]:
    device = model.backbone.get_input_embeddings().weight.device
    prompt = request["prompt_ids"].to(device=device)[None]
    acoustic_prompt = request["acoustic_prompt"]
    if acoustic_prompt is None:
        raise RuntimeError("generation request has no source acoustic prompt.")
    acoustic_codes = acoustic_prompt["codes"].to(device=device)[None]
    acoustic_positions = acoustic_prompt["token_positions"].to(device=device)[None]

    def first_output():
        return model(
            prompt,
            attention_mask=torch.ones_like(prompt, dtype=torch.bool),
            acoustic_prompt_codes=acoustic_codes,
            acoustic_prompt_positions=acoustic_positions,
            output_hidden_states=True,
            use_cache=True,
        )

    first = first_output()
    allowed_ids = model.runtime.audio_generation_allowed_ids
    next_id = selected_id(first.logits[0, -1], allowed_ids)
    next_ids = torch.tensor([[next_id]], device=device)
    sequence = torch.cat((prompt, next_ids), dim=1)
    attention_mask = torch.ones_like(sequence, dtype=torch.bool)
    cache = first.past_key_values
    if cache is None:
        raise RuntimeError("backbone did not return a probe cache.")
    cache_before = int(cache.get_seq_length())

    cached_bool = model(
        next_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        output_hidden_states=True,
        use_cache=True,
    )
    long_cache = first_output().past_key_values
    if long_cache is None:
        raise RuntimeError("backbone did not return a long-mask probe cache.")
    cached_long = model(
        next_ids,
        attention_mask=attention_mask.long(),
        past_key_values=long_cache,
        output_hidden_states=True,
        use_cache=True,
    )
    no_mask_cache = first_output().past_key_values
    if no_mask_cache is None:
        raise RuntimeError("backbone did not return a no-mask probe cache.")
    cached_without_mask = model(
        next_ids,
        past_key_values=no_mask_cache,
        output_hidden_states=True,
        use_cache=True,
    )
    explicit_cache = first_output().past_key_values
    if explicit_cache is None:
        raise RuntimeError("backbone did not return an explicit-position probe cache.")
    position = torch.tensor([prompt.size(1)], device=device)
    cached_explicit_position = model(
        next_ids,
        attention_mask=attention_mask,
        past_key_values=explicit_cache,
        position_ids=position[None],
        cache_position=position,
        output_hidden_states=True,
        use_cache=True,
    )
    full_with_cache = model(
        sequence,
        attention_mask=attention_mask,
        acoustic_prompt_codes=acoustic_codes,
        acoustic_prompt_positions=acoustic_positions,
        output_hidden_states=True,
        use_cache=True,
    )
    full_without_cache = model(
        sequence,
        attention_mask=attention_mask,
        acoustic_prompt_codes=acoustic_codes,
        acoustic_prompt_positions=acoustic_positions,
        output_hidden_states=True,
        use_cache=False,
    )
    outputs = {
        "cached_bool_mask": cached_bool,
        "cached_long_mask": cached_long,
        "cached_without_mask": cached_without_mask,
        "cached_explicit_position": cached_explicit_position,
        "full_with_cache": full_with_cache,
        "full_without_cache": full_without_cache,
    }
    values = {
        name: allowed_values(output.logits[0, -1], allowed_ids)
        for name, output in outputs.items()
    }
    hidden = {
        name: hidden_last(output, name)[0, -1] for name, output in outputs.items()
    }
    return {
        "first_token_id": next_id,
        "cache_length_before": cache_before,
        "cache_length_after": int(cache.get_seq_length()),
        "top_logits": {
            name: top_logits(logits, allowed_ids) for name, logits in values.items()
        },
        "logit_max_abs": {
            name: tensor_max_abs(logits, values["full_without_cache"])
            for name, logits in values.items()
            if name != "full_without_cache"
        },
        "hidden_max_abs": {
            name: tensor_max_abs(state, hidden["full_without_cache"])
            for name, state in hidden.items()
            if name != "full_without_cache"
        },
        "hidden_layer_max_abs": {
            name: hidden_layer_max_abs(output, full_without_cache)
            for name, output in outputs.items()
            if name != "full_without_cache"
        },
        "full_with_vs_without_cache": {
            "logits": tensor_max_abs(
                values["full_with_cache"], values["full_without_cache"]
            ),
            "hidden": tensor_max_abs(
                hidden["full_with_cache"], hidden["full_without_cache"]
            ),
        },
    }
