from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .types import AudioOutput, Result


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
        "finite": bool(torch.isfinite(features).all() and torch.isfinite(waveform).all()),
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
        "tokens_equal": bool(torch.equal(cached_tokens, full_tokens)),
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
    if len(cached) != len(full):
        raise ValueError("cached and full generation must contain the same logit steps.")
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


def hidden_last(output: Any, name: str) -> torch.Tensor:
    if output.hidden_states is None:
        raise RuntimeError(f"generation did not return {name} hidden states.")
    return output.hidden_states[-1]


def hidden_layer_max_abs(output: Any, reference: Any) -> list[float]:
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
