"""Structured timing events shared by streaming-synthesis producers."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from speech_to_speech.callback.gpu import GpuTelemetryMark, GpuTelemetrySampler


_SCHEMA = "speech-to-speech-synthesis-event-v1"
EventSink = Callable[[Mapping[str, object]], None]


class SynthesisTelemetry:
    """Durable producer events plus constant-memory GPU interval summaries."""

    def __init__(
        self,
        root: Path,
        *,
        gpu_sample_interval_seconds: float = 1.0,
    ) -> None:
        if isinstance(gpu_sample_interval_seconds, bool) or not isinstance(
            gpu_sample_interval_seconds,
            (int, float),
        ):
            raise TypeError("synthesis GPU sample interval must be numeric.")
        if gpu_sample_interval_seconds < 0:
            raise ValueError("synthesis GPU sample interval must be non-negative.")
        self.root = root.expanduser().resolve()
        self.gpu_sample_interval_seconds = float(gpu_sample_interval_seconds)
        self.events_path = self.root / "producer_telemetry.jsonl"
        self.gpu_path = self.root / "producer_gpu.csv"
        self.gpu_summary_path = self.root / "producer_gpu_summary.json"
        self._events: TextIO | None = None
        self._gpu: GpuTelemetrySampler | None = None
        self._lock = threading.Lock()

    def __enter__(self) -> SynthesisTelemetry:
        if self._events is not None:
            raise RuntimeError("synthesis telemetry session is already active.")
        self.root.mkdir(parents=True, exist_ok=True)
        self._events = self.events_path.open("a", encoding="utf-8")
        self._gpu = GpuTelemetrySampler(
            self.gpu_path,
            interval_seconds=self.gpu_sample_interval_seconds,
        )
        self._gpu.start()
        self.event(
            "producer_telemetry_started",
            gpu_sample_interval_seconds=self.gpu_sample_interval_seconds,
            gpu_path=str(self.gpu_path),
        )
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: object,
    ) -> None:
        del traceback
        gpu = self._gpu
        if gpu is None:
            raise RuntimeError("synthesis telemetry session was not active.")
        gpu.stop()
        summary = gpu.summary()
        self.gpu_summary_path.write_text(
            json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        values: dict[str, object] = {"gpu": summary}
        if error is not None:
            values.update(
                {
                    "error_type": error_type.__name__ if error_type is not None else "unknown",
                    "error": str(error),
                }
            )
        self.event(
            "producer_telemetry_failed" if error is not None else "producer_telemetry_finished",
            **values,
        )
        events = self._events
        if events is not None:
            events.close()
        self._events = None
        self._gpu = None

    def event(self, event: str, **values: object) -> None:
        """Append one event to the sidecar and mirror it to captured stdout."""

        self._sink(_event(event, values))

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        sample_count: int | None = None,
        device: str | None = None,
        gpu_ids: Sequence[int | str] | None = None,
    ) -> Iterator[None]:
        """Time one model stage and summarize GPU samples in the same interval."""

        normalized_gpu_ids = (
            None if gpu_ids is None else tuple(_gpu_id(value) for value in gpu_ids)
        )
        mark = self._mark()
        try:
            with stage(
                name,
                sample_count=sample_count,
                device=device,
                gpu_ids=gpu_ids,
                sink=self._sink,
            ):
                yield
        finally:
            self._interval_gpu(
                "stage_gpu_summary",
                "stage",
                name,
                mark,
                gpu_ids=normalized_gpu_ids,
            )

    @contextmanager
    def wait(
        self,
        name: str,
        *,
        sample_count: int | None = None,
    ) -> Iterator[None]:
        """Record queue/join/backpressure wait separately from model work."""

        name = _string(name, "synthesis wait name")
        if sample_count is not None and (
            type(sample_count) is not int or sample_count < 0
        ):
            raise ValueError("synthesis wait sample_count must be non-negative.")
        common: dict[str, object] = {"wait": name}
        if sample_count is not None:
            common["sample_count"] = sample_count
        mark = self._mark()
        started = time.perf_counter()
        self._sink(_event("wait_started", common))
        try:
            yield
        except BaseException as error:
            self._sink(
                _event(
                    "wait_failed",
                    {
                        **common,
                        "elapsed_seconds": _elapsed(started),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
            )
            raise
        else:
            self._sink(
                _event(
                    "wait_finished",
                    {**common, "elapsed_seconds": _elapsed(started)},
                )
            )
        finally:
            self._interval_gpu("wait_gpu_summary", "wait", name, mark)

    def _mark(self) -> GpuTelemetryMark:
        gpu = self._gpu
        if gpu is None:
            raise RuntimeError("synthesis telemetry session is not active.")
        return gpu.mark()

    def _interval_gpu(
        self,
        event: str,
        key: str,
        name: str,
        mark: GpuTelemetryMark,
        *,
        gpu_ids: Sequence[int | str] | None = None,
    ) -> None:
        gpu = self._gpu
        if gpu is None:
            raise RuntimeError("synthesis telemetry session is not active.")
        self._sink(
            _event(
                event,
                {key: name, **gpu.summary_since(mark, gpu_ids=gpu_ids)},
            )
        )

    def _sink(self, payload: Mapping[str, object]) -> None:
        events = self._events
        if events is None:
            raise RuntimeError("synthesis telemetry session is not active.")
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock:
            events.write(line + "\n")
            events.flush()
            print(line, flush=True)


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


__all__ = ["SynthesisTelemetry", "emit_event", "stage"]
