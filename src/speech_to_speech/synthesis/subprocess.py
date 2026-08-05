"""Rank-zero subprocess controller for a streaming-synthesis producer.

The child receives stream identity through explicit environment variables and
publishes snapshots independently.  This module only owns the child lifecycle;
it deliberately does not implement synthesis work itself.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from speech_to_speech.datamodule.streaming import SynthesisRequest


_PID_SCHEMA = "speech-to-speech-stream-producer-v1"
_FAILURE_SCHEMA = "speech-to-speech-stream-failure-v1"
_SEAL_SCHEMA = "speech-to-speech-stream-seal-v1"


def subprocess_controller(request: SynthesisRequest) -> SubprocessController:
    """Build the standard ``module:attribute`` synthesis producer factory."""

    return SubprocessController(request)


class SubprocessController:
    """Start or reuse exactly one producer process for one stream root."""

    def __init__(self, request: SynthesisRequest) -> None:
        self.request = request
        self.root = request.root.expanduser().resolve()
        self._command: tuple[str, ...] | None = None
        self._process: subprocess.Popen[Any] | None = None
        self._pid: int | None = None
        self._owns_process = False
        self._monitor: threading.Thread | None = None
        self._stop = threading.Event()
        self._closed = False

    def start(self) -> None:
        """Start the configured non-shell command on rank zero, if needed."""

        if not _rank_zero():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with _ProducerLock(self.root / ".producer.lock"):
            self._start_locked()

    def _start_locked(self) -> None:
        if _sealed(self.root, self.request):
            return
        if self._closed:
            raise RuntimeError("a closed synthesis controller cannot be restarted.")
        _validate_options(self.request.options)
        retry = _bool_option(self.request.options, "retry", default=False)
        failure = self.root / "failed.json"
        if failure.exists():
            if not retry:
                raise RuntimeError(
                    "streaming synthesis already failed; set producer option retry=true "
                    "to explicitly start a new producer."
                )
            failure.unlink()
        command = _command(self.request.options)
        environment_overrides = _option_environment(self.request.options)
        self._command = command
        metadata = self.root / "producer.json"
        if metadata.exists():
            pid = _read_pid(
                metadata,
                self.request,
                command,
                environment_overrides,
            )
            if _alive(pid):
                self._pid = pid
                self._start_monitor()
                return
            _write_failure(
                self.root,
                self.request,
                f"recorded streaming synthesis producer pid {pid} is not running.",
                None,
            )
            if not retry:
                raise RuntimeError("recorded streaming synthesis producer is not running.")
            metadata.unlink()
            (self.root / "failed.json").unlink(missing_ok=True)
        environment = os.environ.copy()
        environment.update(environment_overrides)
        environment.update(_environment(self.request, self.root))
        cwd = _cwd(self.request.options)
        log = (self.root / "producer.log").open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
        except BaseException:
            log.close()
            raise
        log.close()
        self._process = process
        self._pid = process.pid
        self._owns_process = True
        _write_json(
            metadata,
            {
                "schema": _PID_SCHEMA,
                "stream_id": self.request.stream_id,
                "expected_samples": self.request.expected_samples,
                "input_codec": self.request.resolved_input_codec,
                "codec": self.request.codec,
                "split": self.request.split,
                "pid": process.pid,
                "command": list(command),
                "environment": environment_overrides,
            },
        )
        self._start_monitor()

    def check(self) -> None:
        """Raise promptly if an unsealed producer has exited."""

        if not _rank_zero() or self._pid is None or _sealed(self.root, self.request):
            return
        exit_code = self._exit_code()
        if exit_code is None:
            return
        self._record_failure(exit_code)
        raise RuntimeError(
            "streaming synthesis producer exited before the stream was sealed "
            f"(exit code {exit_code})."
        )

    def close(self) -> None:
        """Stop a child started by this controller without touching a reused one."""

        self._closed = True
        self._stop.set()
        if self._owns_process:
            with _ProducerLock(self.root / ".producer.lock"):
                self._close_owned_process()
        monitor = self._monitor
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=5.0)

    def _close_owned_process(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            _terminate_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                _terminate_group(process, signal.SIGKILL)
                process.wait(timeout=5.0)
        if self._owns_process and self._pid is not None:
            metadata = self.root / "producer.json"
            command = self._command
            if command is None:
                raise RuntimeError("owned streaming producer command was not initialized.")
            environment = _option_environment(self.request.options)
            if (
                metadata.exists()
                and _read_pid(metadata, self.request, command, environment) == self._pid
            ):
                metadata.unlink()

    def _start_monitor(self) -> None:
        if self._monitor is not None:
            return
        monitor = threading.Thread(
            target=self._monitor_process,
            name="streaming-synthesis-monitor",
            daemon=True,
        )
        self._monitor = monitor
        monitor.start()

    def _monitor_process(self) -> None:
        seconds = _positive_seconds(self.request.options)
        while not self._stop.is_set():
            exit_code = self._exit_code()
            if exit_code is not None:
                if not _sealed(self.root, self.request) and not self._closed:
                    self._record_failure(exit_code)
                return
            self._stop.wait(seconds)

    def _exit_code(self) -> int | None:
        if self._pid is None:
            return None
        if self._process is not None:
            return self._process.poll()
        return None if _alive(self._pid) else -1

    def _record_failure(self, exit_code: int) -> None:
        _write_failure(
            self.root,
            self.request,
            "streaming synthesis producer exited before the stream was sealed.",
            exit_code,
        )


def _rank_zero() -> bool:
    try:
        import torch.distributed as distributed
    except ImportError:
        return True
    if not distributed.is_available() or not distributed.is_initialized():
        return True
    return distributed.get_rank() == 0


def _command(options: Mapping[str, object]) -> tuple[str, ...]:
    value = options.get("command")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                "streaming subprocess producer command JSON must be an array."
            ) from error
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(
            "streaming subprocess producer option command must be a string list "
            "or a JSON string array."
        )
    command = tuple(value)
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("streaming subprocess producer command must be non-empty strings.")
    return command


def _validate_options(options: Mapping[str, object]) -> None:
    unknown = set(options) - {
        "command",
        "cwd",
        "environment",
        "monitor_seconds",
        "retry",
    }
    if unknown:
        raise ValueError(
            "unknown streaming subprocess producer options: "
            + ", ".join(sorted(unknown))
        )


def _cwd(options: Mapping[str, object]) -> str | None:
    value = options.get("cwd")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError("streaming subprocess producer cwd must be a non-empty string.")
    return value


def _option_environment(options: Mapping[str, object]) -> dict[str, str]:
    value = options.get("environment", {})
    if not isinstance(value, Mapping):
        raise TypeError("streaming subprocess producer environment must be a mapping.")
    result: dict[str, str] = {}
    reserved = set(_environment_names())
    for name, item in value.items():
        if not isinstance(name, str) or not name:
            raise TypeError(
                "streaming subprocess producer environment keys must be non-empty strings."
            )
        if name in reserved:
            raise ValueError(
                f"streaming subprocess producer environment cannot override {name}."
            )
        if not isinstance(item, str):
            raise TypeError(
                "streaming subprocess producer environment values must be strings."
            )
        if item:
            result[name] = item
    return result


def _bool_option(options: Mapping[str, object], name: str, *, default: bool) -> bool:
    value = options.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"streaming subprocess producer {name} must be a boolean.")
    return value


def _positive_seconds(options: Mapping[str, object]) -> float:
    value = options.get("monitor_seconds", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("streaming subprocess producer monitor_seconds must be numeric.")
    if value <= 0:
        raise ValueError("streaming subprocess producer monitor_seconds must be positive.")
    return float(value)


def _environment(request: SynthesisRequest, root: Path) -> dict[str, str]:
    return {
        "S2S_SYNTHESIS_STREAM_ID": request.stream_id,
        "S2S_SYNTHESIS_ROOT": str(root),
        "S2S_SYNTHESIS_EXPECTED_SAMPLES": str(request.expected_samples),
        "S2S_SYNTHESIS_INPUT_CODEC": request.resolved_input_codec,
        "S2S_SYNTHESIS_CODEC": request.codec,
        "S2S_SYNTHESIS_OUTPUT_CODEC": request.codec,
        "S2S_SYNTHESIS_SPLIT": request.split,
    }


def _environment_names() -> tuple[str, ...]:
    return (
        "S2S_SYNTHESIS_STREAM_ID",
        "S2S_SYNTHESIS_ROOT",
        "S2S_SYNTHESIS_EXPECTED_SAMPLES",
        "S2S_SYNTHESIS_INPUT_CODEC",
        "S2S_SYNTHESIS_CODEC",
        "S2S_SYNTHESIS_OUTPUT_CODEC",
        "S2S_SYNTHESIS_SPLIT",
    )


def _sealed(root: Path, request: SynthesisRequest) -> bool:
    path = root / "sealed.json"
    if not path.is_file():
        return False
    value = _read_json(path, "streaming synthesis seal")
    expected: dict[str, object] = {
        "schema": _SEAL_SCHEMA,
        "stream_id": request.stream_id,
        "expected_samples": request.expected_samples,
        "codec": request.codec,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(
                f"streaming synthesis seal {name} does not match the request."
            )
    if _input_codec(value, request) != request.resolved_input_codec:
        raise ValueError(
            "streaming synthesis seal input_codec does not match the request."
        )
    return True


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_group(process: subprocess.Popen[Any], signal_number: int) -> None:
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        return


def _read_pid(
    path: Path,
    request: SynthesisRequest,
    command: Sequence[str],
    environment: Mapping[str, str],
) -> int:
    value = _read_json(path, "streaming producer metadata")
    expected: dict[str, object] = {
        "schema": _PID_SCHEMA,
        "stream_id": request.stream_id,
        "expected_samples": request.expected_samples,
        "codec": request.codec,
        "split": request.split,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(f"streaming producer metadata {name} does not match the request.")
    if _input_codec(value, request) != request.resolved_input_codec:
        raise ValueError(
            "streaming producer metadata input_codec does not match the request."
        )
    if value.get("command") != list(command):
        raise ValueError("streaming producer metadata command does not match the request.")
    recorded_environment = value.get("environment", {})
    if recorded_environment != dict(environment):
        raise ValueError(
            "streaming producer metadata environment does not match the request."
        )
    pid = value.get("pid")
    if type(pid) is not int or pid <= 0:
        raise ValueError(f"streaming producer metadata has an invalid pid: {path}.")
    return pid


def _write_failure(
    root: Path,
    request: SynthesisRequest,
    error: str,
    exit_code: int | None,
) -> None:
    payload: dict[str, object] = {
        "schema": _FAILURE_SCHEMA,
        "stream_id": request.stream_id,
        "expected_samples": request.expected_samples,
        "input_codec": request.resolved_input_codec,
        "codec": request.codec,
        "error": error,
    }
    if exit_code is not None:
        payload["exit_code"] = exit_code
    _write_json(root / "failed.json", payload)


def _input_codec(
    value: Mapping[str, object],
    request: SynthesisRequest,
) -> object:
    if "input_codec" not in value and request.resolved_input_codec == request.codec:
        return request.codec
    return value.get("input_codec")


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON: {path}.") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object: {path}.")
    return value


class _ProducerLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: TextIO | None = None

    def __enter__(self) -> _ProducerLock:
        import fcntl

        self._file = self.path.open("a", encoding="utf-8")
        fcntl.flock(self._file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        import fcntl

        if self._file is None:
            return
        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()
        self._file = None
