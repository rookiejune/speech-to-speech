from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch

from speech_to_speech.generation import Request, generate_responses
from speech_to_speech.model import SpeechToSpeechFlowModel

if __package__:
    from ._generation_reporting import audio_output
else:
    from _generation_reporting import audio_output


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
    results = generate_responses(
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
