"""Best-effort background GPU telemetry for long-running training stages."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


_QUERY_FIELDS = (
    "timestamp",
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
)
_CSV_FIELDS = (
    "sample_time_unix",
    "sample_time_utc",
    "timestamp",
    "index",
    "name",
    "utilization_gpu_percent",
    "utilization_memory_percent",
    "memory_used_mb",
    "memory_total_mb",
    "power_draw_w",
    "power_limit_w",
)
_METRIC_FIELDS = (
    "utilization_gpu_percent",
    "utilization_memory_percent",
    "memory_used_mb",
    "memory_total_mb",
    "power_draw_w",
    "power_limit_w",
)


@dataclass(frozen=True)
class _GpuRow:
    timestamp: str
    index: str
    name: str
    utilization_gpu_percent: float | None
    utilization_memory_percent: float | None
    memory_used_mb: float | None
    memory_total_mb: float | None
    power_draw_w: float | None
    power_limit_w: float | None


@dataclass(frozen=True)
class GpuTelemetryMark:
    """Constant-size sampler state used to summarize an overlapping time span."""

    sampled_at_unix: float
    samples: int
    sums: tuple[float, ...]
    counts: tuple[int, ...]
    gpu_sums: tuple[tuple[str, tuple[float, ...]], ...]
    gpu_counts: tuple[tuple[str, tuple[int, ...]], ...]


class GpuTelemetrySampler:
    """Sample visible GPUs without making the training loop depend on NVML."""

    def __init__(self, path: Path, *, interval_seconds: float) -> None:
        if isinstance(interval_seconds, bool) or not isinstance(
            interval_seconds,
            (int, float),
        ):
            raise TypeError("GPU telemetry interval must be numeric.")
        if interval_seconds < 0:
            raise ValueError("GPU telemetry interval must be non-negative.")
        self.path = path.expanduser().resolve()
        self.interval_seconds = float(interval_seconds)
        self._binary: str | None = None
        self._file: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._visible = _visible_gpu_indices(os.environ.get("CUDA_VISIBLE_DEVICES"))
        self._available = False
        self._reason: str | None = None
        self._started_at: float | None = None
        self._ended_at: float | None = None
        self._sample_count = 0
        self._latest: dict[str, float] = {}
        self._sums = {field: 0.0 for field in _METRIC_FIELDS}
        self._counts = {field: 0 for field in _METRIC_FIELDS}
        self._gpu_sums: dict[str, dict[str, float]] = {}
        self._gpu_counts: dict[str, dict[str, int]] = {}

    def start(self) -> None:
        if self._started_at is not None:
            return
        self._started_at = time.time()
        if self.interval_seconds <= 0:
            self._reason = "disabled"
            self._ended_at = self._started_at
            return
        binary = shutil.which("nvidia-smi")
        if binary is None:
            self._reason = "nvidia-smi not found"
            self._ended_at = self._started_at
            return
        self._binary = binary
        self.path.parent.mkdir(parents=True, exist_ok=True)
        has_header = self.path.exists() and self.path.stat().st_size > 0
        self._file = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=_CSV_FIELDS)
        if not has_header:
            self._writer.writeheader()
            self._file.flush()
        if not self._poll_once():
            self._ended_at = time.time()
            self._close_file()
            return
        self._available = True
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="streaming-gpu-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._started_at is None:
            return
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, min(self.interval_seconds, 5.0)))
        self._ended_at = time.time()
        self._close_file()

    def latest(self) -> dict[str, float]:
        with self._lock:
            return dict(self._latest)

    def mark(self) -> GpuTelemetryMark:
        """Capture cumulative counters without retaining individual samples."""

        with self._lock:
            return GpuTelemetryMark(
                sampled_at_unix=time.time(),
                samples=self._sample_count,
                sums=tuple(self._sums[field] for field in _METRIC_FIELDS),
                counts=tuple(self._counts[field] for field in _METRIC_FIELDS),
                gpu_sums=tuple(
                    (
                        gpu_id,
                        tuple(values[field] for field in _METRIC_FIELDS),
                    )
                    for gpu_id, values in sorted(self._gpu_sums.items())
                ),
                gpu_counts=tuple(
                    (
                        gpu_id,
                        tuple(values[field] for field in _METRIC_FIELDS),
                    )
                    for gpu_id, values in sorted(self._gpu_counts.items())
                ),
            )

    def summary_since(
        self,
        mark: GpuTelemetryMark,
        *,
        gpu_ids: Iterable[int | str] | None = None,
    ) -> dict[str, object]:
        """Return mean GPU metrics collected since ``mark``.

        Cumulative counter deltas make this safe for overlapping producer
        stages without keeping a multi-month run in memory.
        """

        if not isinstance(mark, GpuTelemetryMark):
            raise TypeError("GPU telemetry summary mark must be a GpuTelemetryMark.")
        selected = None if gpu_ids is None else frozenset(str(value) for value in gpu_ids)
        with self._lock:
            if mark.samples > self._sample_count:
                raise ValueError("GPU telemetry mark is ahead of the sampler state.")
            means = self._means_since(mark, selected)
            samples = self._sample_count - mark.samples
        return {
            "scope": "time_span",
            "available": self._available,
            "reason": self._reason,
            "gpu_ids": None if selected is None else sorted(selected),
            "samples": samples,
            "started_at_unix": mark.sampled_at_unix,
            "ended_at_unix": time.time(),
            "duration_seconds": max(0.0, time.time() - mark.sampled_at_unix),
            **means,
        }

    def _means_since(
        self,
        mark: GpuTelemetryMark,
        selected: frozenset[str] | None,
    ) -> dict[str, float]:
        if selected is None:
            current_sums = tuple(self._sums[field] for field in _METRIC_FIELDS)
            current_counts = tuple(self._counts[field] for field in _METRIC_FIELDS)
            previous_sums = mark.sums
            previous_counts = mark.counts
        else:
            previous_gpu_sums = dict(mark.gpu_sums)
            previous_gpu_counts = dict(mark.gpu_counts)
            current_sums = tuple(
                sum(
                    self._gpu_sums.get(gpu_id, {}).get(field, 0.0)
                    for gpu_id in selected
                )
                for field in _METRIC_FIELDS
            )
            current_counts = tuple(
                sum(
                    self._gpu_counts.get(gpu_id, {}).get(field, 0)
                    for gpu_id in selected
                )
                for field in _METRIC_FIELDS
            )
            previous_sums = tuple(
                sum(
                    previous_gpu_sums.get(gpu_id, (0.0,) * len(_METRIC_FIELDS))[position]
                    for gpu_id in selected
                )
                for position in range(len(_METRIC_FIELDS))
            )
            previous_counts = tuple(
                sum(
                    previous_gpu_counts.get(gpu_id, (0,) * len(_METRIC_FIELDS))[position]
                    for gpu_id in selected
                )
                for position in range(len(_METRIC_FIELDS))
            )
        means: dict[str, float] = {}
        for position, field in enumerate(_METRIC_FIELDS):
            count = current_counts[position] - previous_counts[position]
            total = current_sums[position] - previous_sums[position]
            if count < 0 or total < 0:
                raise ValueError("GPU telemetry mark does not belong to this sampler.")
            if count:
                means[field] = total / count
        return means

    def summary(self) -> dict[str, object]:
        with self._lock:
            means = {
                field: self._sums[field] / self._counts[field]
                for field in _METRIC_FIELDS
                if self._counts[field]
            }
            sample_count = self._sample_count
        started = self._started_at
        ended = self._ended_at
        return {
            "scope": "current_process",
            "available": self._available,
            "reason": self._reason,
            "path": str(self.path),
            "interval_seconds": self.interval_seconds,
            "samples": sample_count,
            "started_at_unix": started,
            "ended_at_unix": ended,
            "duration_seconds": None
            if started is None or ended is None
            else max(0.0, ended - started),
            **means,
        }

    def _poll_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._poll_once()

    def _poll_once(self) -> bool:
        binary = self._binary
        if binary is None:
            return False
        try:
            completed = subprocess.run(
                [
                    binary,
                    f"--query-gpu={','.join(_QUERY_FIELDS)}",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            self._reason = str(error)
            return False
        if completed.returncode != 0:
            self._reason = completed.stderr.strip() or "nvidia-smi failed"
            return False
        rows = _filter_visible_rows(_parse_rows(completed.stdout), self._visible)
        if not rows:
            self._reason = "nvidia-smi returned no visible GPU rows"
            return False
        sample_time = time.time()
        with self._lock:
            self._sample_count += 1
            latest: dict[str, list[float]] = {field: [] for field in _METRIC_FIELDS}
            for row in rows:
                values = _csv_row(sample_time, row)
                if self._writer is not None:
                    self._writer.writerow(values)
                for field in _METRIC_FIELDS:
                    value = getattr(row, field)
                    if value is not None:
                        latest[field].append(value)
                        self._sums[field] += value
                        self._counts[field] += 1
                        gpu_sums = self._gpu_sums.setdefault(
                            row.index,
                            {name: 0.0 for name in _METRIC_FIELDS},
                        )
                        gpu_counts = self._gpu_counts.setdefault(
                            row.index,
                            {name: 0 for name in _METRIC_FIELDS},
                        )
                        gpu_sums[field] += value
                        gpu_counts[field] += 1
            if self._file is not None:
                self._file.flush()
            self._latest = {
                field: sum(values) / len(values)
                for field, values in latest.items()
                if values
            }
        return True

    def _close_file(self) -> None:
        file = self._file
        if file is None:
            return
        file.close()
        self._file = None
        self._writer = None


def _parse_rows(output: str) -> list[_GpuRow]:
    rows: list[_GpuRow] = []
    for raw in csv.reader(output.splitlines()):
        if len(raw) != len(_QUERY_FIELDS):
            continue
        rows.append(
            _GpuRow(
                timestamp=raw[0].strip(),
                index=raw[1].strip(),
                name=raw[2].strip(),
                utilization_gpu_percent=_float(raw[3]),
                utilization_memory_percent=_float(raw[4]),
                memory_used_mb=_float(raw[5]),
                memory_total_mb=_float(raw[6]),
                power_draw_w=_float(raw[7]),
                power_limit_w=_float(raw[8]),
            )
        )
    return rows


def _filter_visible_rows(
    rows: Iterable[_GpuRow],
    visible: frozenset[str] | None,
) -> list[_GpuRow]:
    if visible is None:
        return list(rows)
    return [row for row in rows if row.index in visible]


def _csv_row(sample_time: float, row: _GpuRow) -> dict[str, object]:
    return {
        "sample_time_unix": sample_time,
        "sample_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(sample_time)),
        "timestamp": row.timestamp,
        "index": row.index,
        "name": row.name,
        **{
            field: getattr(row, field)
            for field in _METRIC_FIELDS
        },
    }


def _float(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() == "N/A":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _visible_gpu_indices(value: str | None) -> frozenset[str] | None:
    if value is None or not value.strip() or value.strip() in {"-1", "none", "None"}:
        return None if value is None or not value.strip() else frozenset()
    indexes = [item.strip() for item in value.split(",") if item.strip().isdecimal()]
    return None if not indexes else frozenset(str(int(item)) for item in indexes)


__all__ = ["GpuTelemetryMark", "GpuTelemetrySampler"]
