from __future__ import annotations

import math


def seconds(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None.")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def from_frames(
    value: object,
    *,
    frames: int,
    frame_rate: float,
) -> float:
    duration = seconds(value, name="AudioMeta.DURATION")
    if duration is not None:
        return duration
    if isinstance(frames, bool) or not isinstance(frames, int):
        raise TypeError("audio frame count must be an integer.")
    if frames < 0:
        raise ValueError("audio frame count must be non-negative.")
    if isinstance(frame_rate, bool) or not isinstance(frame_rate, (int, float)):
        raise TypeError("codec frame_rate must be a number.")
    rate = float(frame_rate)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("codec frame_rate must be finite and positive.")
    return frames / rate


def from_samples(
    value: object,
    *,
    samples: int,
    sample_rate: int,
) -> float:
    duration = seconds(value, name="AudioMeta.DURATION")
    if duration is not None:
        return duration
    if isinstance(samples, bool) or not isinstance(samples, int):
        raise TypeError("audio sample count must be an integer.")
    if samples < 0:
        raise ValueError("audio sample count must be non-negative.")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise TypeError("audio sample_rate must be an integer.")
    if sample_rate <= 0:
        raise ValueError("audio sample_rate must be positive.")
    return samples / sample_rate
