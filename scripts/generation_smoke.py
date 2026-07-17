from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from anydataset.types import Modality

from speech_to_speech.datamodule import Collator
from speech_to_speech.generation import (
    AudioOutput,
    Request,
    Result,
    generate,
)
from speech_to_speech.generation.batch import requests_from_batch
from speech_to_speech.model import SpeechToSpeechFlowModel
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import init_runtime
from speech_to_speech.task import Task
from zhuyin.datasets.wmt19_tts import wmt19_tts_codec


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rt = init_runtime(
        RuntimeConfig(
            codec=args.codec,
            backbone=args.backbone,
            audio_tokenizer=args.audio_tokenizer,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
    )
    dataset = wmt19_tts_codec(codec=args.codec, split=args.split)
    raw = dataset[args.sample_index]
    batch = Collator(rt, {Task.S2ST: 1.0})([raw])
    request = requests_from_batch(batch)[0]

    model = SpeechToSpeechFlowModel(runtime=rt).eval()
    with torch.no_grad():
        model.acoustic_prompt_gate.fill_(args.acoustic_prompt_gate)

    second_step_probe = probe_second_step(model, request)

    cached = run(
        model,
        request,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )
    full = run(
        model,
        request,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        use_cache=False,
    )
    comparison = compare(cached, full)
    batch_sizes = _batch_sizes(args.batch_sizes)
    batch_requests = [
        requests_from_batch(Collator(rt, {Task.S2ST: 1.0})([dataset[index]]))[0]
        for index in range(args.sample_index, args.sample_index + max(batch_sizes))
    ]
    for prefix_length, batch_request in enumerate(batch_requests):
        if prefix_length == 0:
            continue
        prefix = batch_request["prompt_ids"].new_full(
            (prefix_length,), rt.bos_token_id
        )
        batch_request["prompt_ids"] = torch.cat(
            (prefix, batch_request["prompt_ids"])
        )
        acoustic_prompt = batch_request["acoustic_prompt"]
        if acoustic_prompt is not None:
            acoustic_prompt["token_positions"] = (
                acoustic_prompt["token_positions"] + prefix_length
            )
    batch_benchmark = [
        benchmark_batch(
            model,
            batch_requests[:batch_size],
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
        for batch_size in batch_sizes
    ]

    result = {
        "task": Task.S2ST.value,
        "sample_index": args.sample_index,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "prompt_tokens": int(request["prompt_ids"].numel()),
        "source_acoustic_frames": (
            0
            if request["acoustic_prompt"] is None
            else int(request["acoustic_prompt"]["codes"].size(0))
        ),
        "acoustic_prompt_gate": args.acoustic_prompt_gate,
        "second_step_probe": second_step_probe,
        "cached": summary(cached),
        "full_recompute": summary(full),
        "comparison": comparison,
        "batch_benchmark": batch_benchmark,
    }
    result_path = output_dir / "metrics.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not comparison["tokens_equal"]:
        raise RuntimeError("cached and full-recompute greedy tokens differ.")
    if not comparison["cached_finite"] or not comparison["full_finite"]:
        raise RuntimeError("generation produced non-finite acoustic output.")
    if not all(item["tokens_equal"] for item in batch_benchmark):
        raise RuntimeError("batch and per-request greedy tokens differ.")


def benchmark_batch(
    model: SpeechToSpeechFlowModel,
    requests: Sequence[Request],
    *,
    seed: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    batched = timed_generate(model, requests, seed, max_new_tokens)
    serial_started = time.perf_counter()
    serial_results = []
    serial_peak = 0
    for offset, request in enumerate(requests):
        output = timed_generate(model, [request], seed + offset, max_new_tokens)
        serial_results.extend(output["results"])
        serial_peak = max(serial_peak, output["peak_cuda_bytes"])
    serial_elapsed = time.perf_counter() - serial_started
    batch_results = batched["results"]
    token_count = sum(result["response_ids"].numel() for result in batch_results)
    finite = all(
        torch.isfinite(audio_output(result, "batch result")["waveform"]).all()
        for result in batch_results
    )
    return {
        "batch_size": len(requests),
        "prompt_tokens": [int(request["prompt_ids"].numel()) for request in requests],
        "source_acoustic_frames": [
            0
            if request["acoustic_prompt"] is None
            else int(request["acoustic_prompt"]["codes"].size(0))
            for request in requests
        ],
        "response_tokens": [
            int(result["response_ids"].numel()) for result in batch_results
        ],
        "batch_token_ids": [
            result["response_ids"].detach().cpu().tolist() for result in batch_results
        ],
        "serial_token_ids": [
            result["response_ids"].detach().cpu().tolist() for result in serial_results
        ],
        "tokens_equal": all(
            torch.equal(batch["response_ids"], serial["response_ids"])
            for batch, serial in zip(batch_results, serial_results)
        ),
        "finite": bool(finite),
        "batch_elapsed_seconds": batched["elapsed_seconds"],
        "serial_elapsed_seconds": serial_elapsed,
        "batch_tokens_per_second": token_count / batched["elapsed_seconds"],
        "serial_tokens_per_second": token_count / serial_elapsed,
        "batch_peak_cuda_bytes": batched["peak_cuda_bytes"],
        "serial_peak_cuda_bytes": serial_peak,
    }


def timed_generate(
    model: SpeechToSpeechFlowModel,
    requests: Sequence[Request],
    seed: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    results = generate(
        requests,
        model,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    torch.cuda.synchronize()
    return {
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }


def _batch_sizes(value: str) -> list[int]:
    sizes = [int(item) for item in value.split(",")]
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("batch sizes must be positive integers.")
    return sizes


def run(
    model: SpeechToSpeechFlowModel,
    request: Request,
    *,
    seed: int,
    max_new_tokens: int,
    use_cache: bool,
) -> dict[str, Any]:
    calls: list[dict[str, int | bool]] = []
    allowed_logits: list[torch.Tensor] = []

    def observe(module, args, kwargs) -> None:
        del module
        input_ids = args[0]
        attention_mask = kwargs["attention_mask"]
        calls.append(
            {
                "input_tokens": int(input_ids.size(1)),
                "attention_tokens": int(attention_mask.size(1)),
                "has_past": kwargs.get("past_key_values") is not None,
                "has_acoustic_prompt": kwargs.get("acoustic_prompt_codes") is not None,
            }
        )

    def observe_output(module, args, kwargs, output) -> None:
        del args
        ids = torch.as_tensor(
            module.runtime.audio_generation_allowed_ids,
            device=output.logits.device,
            dtype=torch.long,
        )
        requested_ids = kwargs.get("_generation_token_ids")
        generation_modality = kwargs.get("_generation_modality")
        values = output.logits[0, -1]
        if requested_ids is None:
            if generation_modality is not Modality.AUDIO:
                raise RuntimeError("generation smoke expected the audio output head.")
            audio_start, _ = module.layout.blocks[Modality.AUDIO.value]
            values = values.index_select(0, ids - audio_start)
        elif not torch.equal(requested_ids, ids):
            raise RuntimeError("generation used unexpected allowed token ids.")
        allowed_logits.append(
            values.detach().float().cpu()
        )

    pre_handle = model.register_forward_pre_hook(observe, with_kwargs=True)
    post_handle = model.register_forward_hook(observe_output, with_kwargs=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        result = generate(
            [request],
            model,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
        )[0]
    finally:
        pre_handle.remove()
        post_handle.remove()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "result": result,
        "calls": calls,
        "allowed_logits": allowed_logits,
        "allowed_ids": model.runtime.audio_generation_allowed_ids,
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }


@torch.no_grad()
def probe_second_step(
    model: SpeechToSpeechFlowModel,
    request: Request,
) -> dict[str, Any]:
    device = model.backbone.get_input_embeddings().weight.device
    prompt = request["prompt_ids"].to(device=device)[None]
    acoustic_prompt = request["acoustic_prompt"]
    if acoustic_prompt is None:
        raise RuntimeError("generation request has no source acoustic prompt.")
    acoustic_codes = acoustic_prompt["codes"].to(device=device)[None]
    acoustic_token_positions = acoustic_prompt["token_positions"].to(device=device)[None]

    def first_output():
        return model(
            prompt,
            attention_mask=torch.ones_like(prompt, dtype=torch.bool),
            acoustic_prompt_codes=acoustic_codes,
            acoustic_prompt_positions=acoustic_token_positions,
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
        acoustic_prompt_positions=acoustic_token_positions,
        output_hidden_states=True,
        use_cache=True,
    )
    full_without_cache = model(
        sequence,
        attention_mask=attention_mask,
        acoustic_prompt_codes=acoustic_codes,
        acoustic_prompt_positions=acoustic_token_positions,
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


def summary(run_output: dict[str, Any]) -> dict[str, Any]:
    result = run_output["result"]
    audio = audio_output(result, "generation result")
    features = audio["features"]
    waveform = audio["waveform"]
    if features is None:
        raise RuntimeError("generation smoke requires acoustic features.")
    return {
        "token_ids": result["response_ids"].detach().cpu().tolist(),
        "acoustic_shape": list(features.shape),
        "waveform_shape": list(waveform.shape),
        "finite": bool(
            torch.isfinite(features).all() and torch.isfinite(waveform).all()
        ),
        "calls": run_output["calls"],
        "top_logits": [
            top_logits(values, run_output["allowed_ids"])
            for values in run_output["allowed_logits"]
        ],
        "elapsed_seconds": run_output["elapsed_seconds"],
        "peak_cuda_bytes": run_output["peak_cuda_bytes"],
    }


def compare(cached_run: dict[str, Any], full_run: dict[str, Any]) -> dict[str, Any]:
    cached = cached_run["result"]
    full = full_run["result"]
    cached_audio = audio_output(cached, "cached result")
    full_audio = audio_output(full, "full-recompute result")
    cached_features = cached_audio["features"]
    full_features = full_audio["features"]
    cached_waveform = cached_audio["waveform"]
    full_waveform = full_audio["waveform"]
    cached_tokens = cached["response_ids"]
    full_tokens = full["response_ids"]
    if cached_features is None or full_features is None:
        raise RuntimeError("generation smoke requires acoustic features.")
    return {
        "tokens_equal": bool(
            torch.equal(cached["response_ids"], full["response_ids"])
        ),
        "first_token_difference": first_difference(cached_tokens, full_tokens),
        "logit_steps": compare_logits(
            cached_run["allowed_logits"], full_run["allowed_logits"]
        ),
        "acoustic_shapes_equal": cached_features.shape == full_features.shape,
        "waveform_shapes_equal": cached_waveform.shape == full_waveform.shape,
        "acoustic_max_abs": optional_max_abs(cached_features, full_features),
        "waveform_max_abs": optional_max_abs(cached_waveform, full_waveform),
        "cached_finite": bool(
            torch.isfinite(cached_features).all()
            and torch.isfinite(cached_waveform).all()
        ),
        "full_finite": bool(
            torch.isfinite(full_features).all() and torch.isfinite(full_waveform).all()
        ),
    }


def optional_max_abs(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.shape != right.shape:
        return None
    return float((left.float() - right.float()).abs().max())


def compare_logits(
    cached: list[torch.Tensor], full: list[torch.Tensor]
) -> list[dict[str, float | int]]:
    return [
        {
            "step": step,
            "max_abs": float((cached_values - full_values).abs().max()),
        }
        for step, (cached_values, full_values) in enumerate(zip(cached, full))
    ]


def allowed_values(logits: torch.Tensor, allowed_ids: Sequence[int]) -> torch.Tensor:
    ids = torch.as_tensor(allowed_ids, device=logits.device, dtype=torch.long)
    return logits.index_select(0, ids).detach().float().cpu()


def selected_id(logits: torch.Tensor, allowed_ids: Sequence[int]) -> int:
    values = allowed_values(logits, allowed_ids)
    return allowed_ids[int(values.argmax())]


def tensor_max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def hidden_last(output, name: str) -> torch.Tensor:
    if output.hidden_states is None:
        raise RuntimeError(f"generation did not return {name} hidden states.")
    return output.hidden_states[-1]


def hidden_layer_max_abs(output, reference) -> list[float]:
    if output.hidden_states is None or reference.hidden_states is None:
        raise RuntimeError("probe did not return layer hidden states.")
    return [
        tensor_max_abs(left[0, -1], right[0, -1])
        for left, right in zip(output.hidden_states, reference.hidden_states)
    ]


def top_logits(
    values: torch.Tensor, allowed_ids: Sequence[int], count: int = 5
) -> dict[str, Any]:
    top_values, local_ids = values.topk(min(count, values.numel()))
    margin = float(top_values[0] - top_values[1]) if top_values.numel() > 1 else None
    return {
        "token_ids": [allowed_ids[index] for index in local_ids.tolist()],
        "values": top_values.tolist(),
        "top1_margin": margin,
    }


def first_difference(left: torch.Tensor, right: torch.Tensor) -> int | None:
    shared = min(left.numel(), right.numel())
    difference = (left[:shared] != right[:shared]).nonzero()
    if difference.numel():
        return int(difference[0].item())
    if left.numel() != right.numel():
        return shared
    return None


def audio_output(result: Result, name: str) -> AudioOutput:
    audio = result["audio"]
    if audio is None:
        raise RuntimeError(f"{name} did not return audio output.")
    return audio


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio-tokenizer", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--acoustic-prompt-gate", type=float, default=1.0)
    parser.add_argument("--codec", default="longcat")
    parser.add_argument("--backbone", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser


if __name__ == "__main__":
    main()
