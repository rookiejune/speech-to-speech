"""Structured timing events shared by streaming-synthesis producers."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager


_SCHEMA = "speech-to-speech-synthesis-event-v1"
EventSink = Callable[[Mapping[str, object]], None]


def emit_event(event: str, **values: object) -> None:
    """Write one timestamped JSON event to the producer's captured stdout."""

    payload = _event(event, values)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


@contextmanager
def stage(
    name: str,
    *,
    sample_count: int | None = None,
    device: str | None = None,
    gpu_ids: Sequence[int | str] | None = None,
    sink: EventSink | None = None,
) -> Iterator[None]:
    """Emit start/finish/failure events around one producer stage."""

    name = _string(name, "synthesis stage name")
    if sample_count is not None and (
        type(sample_count) is not int or sample_count < 0
    ):
        raise ValueError("synthesis stage sample_count must be non-negative.")
    if device is not None:
        device = _string(device, "synthesis stage device")
    normalized_gpu_ids = None if gpu_ids is None else [_gpu_id(value) for value in gpu_ids]
    output = _stdout_sink if sink is None else sink
    common: dict[str, object] = {"stage": name}
    if sample_count is not None:
        common["sample_count"] = sample_count
    if device is not None:
        common["device"] = device
    if normalized_gpu_ids is not None:
        common["gpu_ids"] = normalized_gpu_ids

    started = time.perf_counter()
    output(_event("stage_started", common))
    try:
        yield
    except BaseException as error:
        output(
            _event(
                "stage_failed",
                {
                    **common,
                    "elapsed_seconds": _elapsed(started),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        )
        raise
    output(
        _event(
            "stage_finished",
            {**common, "elapsed_seconds": _elapsed(started)},
        )
    )


def _event(event: str, values: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema": _SCHEMA,
        "event": _string(event, "synthesis event"),
        "timestamp_unix": time.time(),
        "monotonic_seconds": time.monotonic(),
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **values,
    }


def _stdout_sink(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _elapsed(started: float) -> float:
    elapsed = time.perf_counter() - started
    if elapsed < 0 or not math.isfinite(elapsed):
        raise RuntimeError("synthesis stage timer returned an invalid duration.")
    return elapsed


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _gpu_id(value: object) -> int | str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError("synthesis stage GPU ids must be integers or strings.")
    if isinstance(value, int) and value < 0:
        raise ValueError("synthesis stage GPU ids must be non-negative.")
    if isinstance(value, str) and not value:
        raise ValueError("synthesis stage GPU ids must be non-empty.")
    return value


__all__ = ["emit_event", "stage"]
