from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch

from speech_to_speech.generation import Request, Result, generate_responses
from speech_to_speech.generation.eval.reporting import audio_output
from speech_to_speech.model.acoustic import FlowModel


def benchmark_batch(
    model: FlowModel,
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
    batch_finite = _finite(batch_results, "batch result")
    serial_finite = _finite(serial_results, "serial result")
    return {
        "batch_size": len(requests),
        "prompt_tokens": [int(request["prompt_ids"].numel()) for request in requests],
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
        "finite": batch_finite and serial_finite,
        "batch_finite": batch_finite,
        "serial_finite": serial_finite,
        "batch_elapsed_seconds": batched["elapsed_seconds"],
        "serial_elapsed_seconds": serial_elapsed,
        "batch_tokens_per_second": token_count / batched["elapsed_seconds"],
        "serial_tokens_per_second": token_count / serial_elapsed,
        "batch_peak_cuda_bytes": batched["peak_cuda_bytes"],
        "serial_peak_cuda_bytes": serial_peak,
    }


def timed_generate(
    model: FlowModel,
    requests: Sequence[Request],
    seed: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    device = _model_device(model)
    torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    results = generate_responses(
        requests,
        model,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_bytes = torch.cuda.max_memory_allocated(device)
    else:
        peak_cuda_bytes = 0
    return {
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": peak_cuda_bytes,
    }


def _model_device(model: FlowModel) -> torch.device:
    return model.backbone.get_input_embeddings().weight.device


def _finite(results: Sequence[Result], name: str) -> bool:
    return all(
        bool(torch.isfinite(audio_output(result, name)["waveform"]).all())
        for result in results
    )
