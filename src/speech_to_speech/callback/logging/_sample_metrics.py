from __future__ import annotations

import math
import unicodedata

import torch
from torch import Tensor


def text_metrics(reference: str, generated: str) -> dict[str, float]:
    expected = _text(reference)
    actual = _text(generated)
    return {
        "text/cer": _edit_distance(expected, actual) / max(1, len(expected)),
        "text/exact_match": float(actual == expected),
        "text/reference_chars": float(len(expected)),
        "text/generated_chars": float(len(actual)),
    }


def audio_metrics(
    waveform: Tensor,
    sample_rate: int,
    *,
    target_duration: float | None = None,
) -> dict[str, float]:
    if waveform.numel() == 0:
        raise ValueError("sample audio metrics require a non-empty waveform.")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("sample audio metric sample_rate must be an integer.")
    if sample_rate <= 0:
        raise ValueError("sample audio metric sample_rate must be positive.")
    value = waveform.detach().float()
    duration = value.size(-1) / sample_rate
    finite = bool(torch.isfinite(value).all().item())
    metrics = {
        "audio/duration_seconds": float(duration),
        "audio/finite": float(finite),
    }
    if finite:
        magnitude = value.abs()
        metrics.update(
            {
                "audio/rms": float(value.square().mean().sqrt().item()),
                "audio/peak": float(magnitude.max().item()),
                "audio/silence_ratio": float(magnitude.lt(1e-4).float().mean().item()),
                "audio/clipping_ratio": float(magnitude.ge(0.999).float().mean().item()),
            }
        )
    if target_duration is not None:
        if not math.isfinite(target_duration) or target_duration <= 0:
            raise ValueError("sample target duration must be finite and positive.")
        metrics["audio/duration_ratio"] = duration / target_duration
    return metrics


def _text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("sample text metrics require strings.")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _edit_distance(reference: str, generated: str) -> int:
    previous = list(range(len(generated) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(generated, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + int(expected != actual),
                )
            )
        previous = current
    return previous[-1]


__all__ = ["audio_metrics", "text_metrics"]
