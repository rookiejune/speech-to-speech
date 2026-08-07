"""Resolve a workspace data source before CUDA or model initialization."""

from __future__ import annotations

import importlib
import json
import logging
import os
import signal
import subprocess
import warnings
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Protocol, cast

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch.utils.data import Dataset

from .._compat import StrEnum, auto
from ..audio import AudioStream
from ..datamodule.config import (
    StreamingConfig,
    WorkspaceSourceConfig,
)


_CHILD_PLAN_ENV = "SPEECH_TO_SPEECH_SOURCE_CHILD_PLAN"
_SUBPROCESS_FACTORY = "speech_to_speech.synthesis.process:controller"
_LEGACY_SOURCE_FACTORY = "zhuyin.datasets.wmt19.streaming_s2st:source"
_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from anytrain.lightning import ManagedService


class _LiveDataset(Protocol):
    lineage_id: str
    snapshot_id: str | None
    sample_count: int
    sealed: bool

    def acknowledge(self) -> None: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, value: Mapping[str, object]) -> None: ...

    def set_stop_requested(self, predicate: Any | None) -> None: ...

    def close(self) -> None: ...


class SourceRoute(StrEnum):
    ACCESS = auto()
    GENERATE = auto()
    TOY = auto()
    LEGACY = auto()


@dataclass(frozen=True)
class DevicePlan:
    assigned: tuple[str, ...] | None
    training: tuple[str, ...] | None
    generation: tuple[str, ...]
    training_count: int | None
    recipe: str | None = None
    stage_devices: Mapping[str, str] | None = None
    factories: Mapping[str, tuple[str, ...]] | None = None

    def as_dict(self) -> dict[str, object]:
        if self.factories is not None:
            return {
                "visible": None if self.assigned is None else list(self.assigned),
                "factories": {
                    name: list(devices)
                    for name, devices in self.factories.items()
                },
                "remaining": (
                    None if self.training is None else list(self.training)
                ),
            }
        return {
            "assigned": None if self.assigned is None else list(self.assigned),
            "training": None if self.training is None else list(self.training),
            "training_count": self.training_count,
            "generation": list(self.generation),
            "recipe": self.recipe,
            "stages": (
                None if self.stage_devices is None else dict(self.stage_devices)
            ),
        }


@dataclass(frozen=True)
class DataPlan:
    route: SourceRoute
    formal_training: bool
    reason: str
    source_factory: str | None
    logical_id: str | None
    stream_id: str | None
    expected_samples: int | None
    access_state: str | None
    access_layout: str | None
    access_detail: str | None
    devices: DevicePlan
    generation_profile: str | None = None
    generation_entrypoint: str | None = None
    generation_log: str | None = None
    generation_telemetry: str | None = None
    lineage_id: str | None = None
    snapshot_id: str | None = None
    sample_count: int | None = None
    sealed: bool | None = None
    generation_logs: Mapping[str, str] | None = None

    def as_dict(self) -> dict[str, object]:
        if self.lineage_id is not None:
            return {
                "event": "data.plan",
                "route": self.route.value,
                "formal_training": self.formal_training,
                "reason": self.reason,
                "source": {
                    "factory": self.source_factory,
                    "lineage_id": self.lineage_id,
                },
                "access": {
                    "state": self.access_state,
                    "detail": self.access_detail,
                    "snapshot_id": self.snapshot_id,
                    "sample_count": self.sample_count,
                    "sealed": self.sealed,
                },
                "devices": self.devices.as_dict(),
                "generation": (
                    None
                    if self.generation_logs is None
                    else {
                        "factories": {
                            name: {"log": path}
                            for name, path in self.generation_logs.items()
                        }
                    }
                ),
            }
        return {
            "event": "data.plan",
            "route": self.route.value,
            "formal_training": self.formal_training,
            "reason": self.reason,
            "source": {
                "factory": self.source_factory,
                "logical_id": self.logical_id,
                "stream_id": self.stream_id,
                "expected_samples": self.expected_samples,
            },
            "access": {
                "state": self.access_state,
                "layout": self.access_layout,
                "detail": self.access_detail,
            },
            "devices": self.devices.as_dict(),
            "generation": (
                None
                if self.generation_profile is None
                else {
                    "profile": self.generation_profile,
                    "entrypoint": self.generation_entrypoint,
                    "producer_log": self.generation_log,
                    "telemetry": self.generation_telemetry,
                    "stage_seconds": "stage_finished.elapsed_seconds",
                    "wait_seconds": "wait_finished.elapsed_seconds",
                }
            ),
        }


@dataclass(frozen=True)
class SourceResolution:
    config: Any
    plan: DataPlan | None
    training_datasets: Mapping[str, Dataset[Any]]
    service: ManagedService | None = None
    live_dataset: _LiveDataset | None = None


@dataclass(frozen=True)
class _FactoryProcess:
    name: str
    command: tuple[str, ...]
    environment: Mapping[str, str]
    devices: tuple[str, ...]
    log: Path


class GenerationService:
    """Own the workspace generation factory processes on global rank zero."""

    def __init__(self, factories: Sequence[_FactoryProcess]) -> None:
        self.factories = tuple(factories)
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._logs: dict[str, IO[str]] = {}

    def start(self, *, owner: bool) -> None:
        if not owner or self._processes:
            return
        try:
            for factory in self.factories:
                factory.log.parent.mkdir(parents=True, exist_ok=True)
                log = factory.log.open("a", encoding="utf-8")
                environment = dict(os.environ)
                environment.update(factory.environment)
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(factory.devices)
                try:
                    process = subprocess.Popen(
                        factory.command,
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        text=True,
                    )
                except Exception:
                    log.close()
                    raise
                self._logs[factory.name] = log
                self._processes[factory.name] = process
                _LOGGER.info(
                    "data.generation.started %s",
                    json.dumps(
                        {
                            "factory": factory.name,
                            "devices": list(factory.devices),
                            "log": str(factory.log),
                            "pid": process.pid,
                        },
                        sort_keys=True,
                    ),
                )
        except Exception:
            self.close(owner=True)
            raise

    def check(self, *, owner: bool) -> None:
        if not owner:
            return
        for name, process in tuple(self._processes.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            self._finish(name)
            if returncode != 0:
                factory = next(item for item in self.factories if item.name == name)
                raise RuntimeError(
                    f"workspace generation factory {name!r} exited with status "
                    f"{returncode}; see {factory.log}."
                )
            _LOGGER.info(
                "data.generation.finished %s",
                json.dumps({"factory": name, "status": 0}, sort_keys=True),
            )

    def close(self, *, owner: bool) -> None:
        if not owner:
            return
        for process in self._processes.values():
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
        for name, process in tuple(self._processes.items()):
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            self._finish(name)

    def _finish(self, name: str) -> None:
        self._processes.pop(name, None)
        log = self._logs.pop(name, None)
        if log is not None:
            log.close()


@dataclass(frozen=True)
class GenerationServiceProvider:
    service: ManagedService

    def __call__(self, trainer: Trainer) -> ManagedService:
        del trainer
        return self.service


class LiveS2STCursor(Callback):
    """Checkpoint and acknowledge one injected live S2ST dataset."""

    def __init__(
        self,
        dataset: _LiveDataset,
        service: ManagedService | None = None,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.service = service
        self._detached = False
        self._global_step: int | None = None

    @property
    def state_key(self) -> str:
        return self._generate_state_key(lineage_id=self.dataset.lineage_id)

    def state_dict(self) -> dict[str, object]:
        return self.dataset.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self.dataset.load_state_dict(state_dict)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module

        def stop_requested() -> bool:
            if self.service is not None:
                self.service.check(owner=bool(trainer.is_global_zero))
            return bool(trainer.received_sigterm)

        self.dataset.set_stop_requested(stop_requested)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        self._global_step = int(trainer.global_step)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        current = int(trainer.global_step)
        previous = self._global_step
        if previous is None:
            previous = current
        if current <= previous:
            self._global_step = previous
            return
        self.dataset.acknowledge()
        self._global_step = current

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer, pl_module
        self._detach()

    def on_exception(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        exception: BaseException,
    ) -> None:
        del trainer, pl_module, exception
        self._detach()

    def _detach(self) -> None:
        if self._detached:
            return
        self.dataset.set_stop_requested(None)
        self._detached = True


def resolve_workspace_source(
    config: Any,
    *,
    environment: Mapping[str, str] | None = None,
) -> SourceResolution:
    """Resolve the current S2ST source, preserving the old stream for one cycle."""

    source_config = config.datamodule.source
    if not source_config.enabled:
        return _resolve_legacy_workspace_source(config, environment=environment)
    if source_config.factory == _LEGACY_SOURCE_FACTORY:
        warnings.warn(
            "the WMT19-specific workspace source is deprecated; migrate to "
            "zhuyin.datasets.s2st:source within one compatibility cycle.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _resolve_legacy_workspace_source(config, environment=environment)

    source = _load_s2st_source(source_config)
    if not hasattr(source, "lineage_id") or not callable(
        getattr(source, "generate", None)
    ):
        warnings.warn(
            "the configured workspace source uses the legacy generation contract; "
            "migrate it to the S2ST lineage/factory contract.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _resolve_legacy_workspace_source(config, environment=environment)
    _validate_s2st_source(source, factory_path=cast(str, source_config.factory))
    return _resolve_s2st_source(
        config,
        source,
        source_config,
        environment=environment,
    )


def _resolve_s2st_source(
    config: Any,
    source: Any,
    source_config: WorkspaceSourceConfig,
    *,
    environment: Mapping[str, str] | None,
) -> SourceResolution:
    values = os.environ if environment is None else environment
    lineage_id = _string(source.lineage_id, "workspace lineage_id")
    child = _s2st_child_plan(
        values,
        source_factory=cast(str, source_config.factory),
        lineage_id=lineage_id,
    )
    access = source.access()
    state = _enum_value(access.state, "workspace access state").lower()
    detail = _string(access.detail, "workspace access detail")
    if state == "invalid":
        raise RuntimeError(f"workspace source access is invalid: {detail}")
    if state not in {"ready", "missing"}:
        raise ValueError(f"unsupported workspace access state: {state!r}.")
    if not isinstance(access.sealed, bool):
        raise TypeError("workspace access sealed must be a boolean.")
    dataset = _s2st_dataset(access, lineage_id=lineage_id)
    snapshot_id = _optional_string(dataset.snapshot_id, "workspace snapshot_id")
    _non_negative_int(
        dataset.sample_count,
        "workspace sample_count",
    )
    sealed = _boolean(dataset.sealed, "workspace dataset sealed")
    if access.sealed != sealed:
        raise RuntimeError(
            "workspace access and live dataset disagree about whether the catalog "
            "is sealed."
        )
    if state == "ready" and snapshot_id is None:
        raise RuntimeError(
            "workspace access reported ready without a published snapshot."
        )
    if state == "missing" and snapshot_id is not None:
        raise RuntimeError(
            "workspace access reported missing with a published snapshot."
        )
    if sealed and snapshot_id is None:
        raise RuntimeError("workspace access is sealed without a published snapshot.")

    mode = source_config.mode
    if child is not None:
        mode = SourceRoute.GENERATE.value
    if mode == SourceRoute.TOY.value:
        return _s2st_toy_resolution(
            config,
            source,
            dataset,
            state=state,
            detail=detail,
            environment=values,
        )
    if sealed:
        return _s2st_access_resolution(
            config,
            dataset,
            source_config,
            state=state,
            detail=detail,
            environment=values,
            reason="workspace snapshot is sealed",
        )
    if mode == SourceRoute.ACCESS.value:
        if snapshot_id is None:
            raise RuntimeError(f"workspace access was explicitly required: {detail}")
        return _s2st_access_resolution(
            config,
            dataset,
            source_config,
            state=state,
            detail=detail,
            environment=values,
            reason=(
                "workspace snapshot is sealed"
                if sealed
                else "workspace snapshot is available; generation was not requested"
            ),
        )
    if mode == SourceRoute.GENERATE.value:
        return _s2st_generation_resolution(
            config,
            source,
            dataset,
            source_config,
            state=state,
            detail=detail,
            environment=values,
            child=child,
        )
    if mode != "auto":
        raise ValueError(f"unsupported workspace source mode: {mode!r}.")

    device_count = _known_device_count(config, values)
    if device_count == 0:
        raise RuntimeError("workspace source routing has no assigned training device.")
    if snapshot_id is not None and device_count == 1:
        return _s2st_access_resolution(
            config,
            dataset,
            source_config,
            state=state,
            detail=detail,
            environment=values,
            reason=(
                "workspace snapshot is available; one device keeps training on the "
                "current prefix without generation"
            ),
        )
    if snapshot_id is None and device_count == 1:
        return _s2st_toy_resolution(
            config,
            source,
            dataset,
            state=state,
            detail=detail,
            environment=values,
        )
    if device_count is None:
        condition = (
            "an unsealed workspace snapshot"
            if snapshot_id is not None
            else "a missing initial workspace snapshot"
        )
        raise RuntimeError(
            f"{condition} requires explicit CUDA_VISIBLE_DEVICES so preflight can "
            "decide whether generation has a device separate from training."
        )
    return _s2st_generation_resolution(
        config,
        source,
        dataset,
        source_config,
        state=state,
        detail=detail,
        environment=values,
        child=child,
    )


def _s2st_access_resolution(
    config: Any,
    dataset: _LiveDataset,
    source_config: WorkspaceSourceConfig,
    *,
    state: str,
    detail: str,
    environment: Mapping[str, str],
    reason: str,
) -> SourceResolution:
    assigned = _assigned_devices(config, environment)
    count = _training_device_count(config, assigned, required=True)
    assert count is not None
    resolved = _live_config(config, training_count=count)
    plan = _s2st_plan(
        route=SourceRoute.ACCESS,
        formal_training=True,
        reason=reason,
        source_factory=source_config.factory,
        dataset=dataset,
        state=state,
        detail=detail,
        devices=DevicePlan(
            assigned=assigned,
            training=assigned,
            generation=(),
            training_count=count,
            factories={},
        ),
    )
    return SourceResolution(
        resolved,
        plan,
        {_single_loader_name(config): cast(Dataset[Any], dataset)},
        live_dataset=dataset,
    )


def _s2st_toy_resolution(
    config: Any,
    source: Any,
    live_dataset: _LiveDataset,
    *,
    state: str,
    detail: str,
    environment: Mapping[str, str],
) -> SourceResolution:
    lineage_id = live_dataset.lineage_id
    snapshot_id = live_dataset.snapshot_id
    sample_count = live_dataset.sample_count
    sealed = live_dataset.sealed
    live_dataset.close()
    toy = source.toy()
    load = getattr(toy, "load", None)
    if not callable(load):
        raise TypeError("workspace source toy is missing load().")
    dataset = load()
    if not isinstance(dataset, Dataset):
        raise TypeError("workspace source toy load() must return a Dataset.")
    toy_config = config.datamodule.source.toy
    max_steps = toy_config.warmup_steps + toy_config.measure_window_steps
    output_subdir = f"{config.output_subdir.rstrip('/')}/toy-perf"
    output_dir = str(Path(config.repo_output_root).expanduser() / output_subdir)
    callbacks = replace(
        config.callbacks,
        task_sample=replace(config.callbacks.task_sample, enabled=False),
        synthesis_sample=replace(config.callbacks.synthesis_sample, enabled=False),
        text_retention=replace(config.callbacks.text_retention, enabled=False),
        gradient_probe=replace(config.callbacks.gradient_probe, enabled=False),
        performance=replace(
            config.callbacks.performance,
            enabled=True,
            log_every_n_steps=toy_config.log_every_n_steps,
            warmup_steps=toy_config.warmup_steps,
            measure_window_steps=toy_config.measure_window_steps,
            stop_after_measurement=True,
        ),
    )
    resolved = replace(
        config,
        output_subdir=output_subdir,
        output_dir=output_dir,
        logging=_toy_logging(config.logging, output_dir, output_subdir),
        datamodule=replace(
            config.datamodule,
            dataloader=_live_dataloader(config.datamodule.dataloader),
            encode_missing_codes=True,
            source=WorkspaceSourceConfig(),
            streaming=StreamingConfig(),
        ),
        train=replace(
            config.train,
            max_steps=max_steps,
            ckpt_path=None,
            auto_resume=False,
        ),
        validation=replace(config.validation, enabled=False),
        trainer=replace(
            config.trainer,
            devices=1,
            strategy="auto",
            max_epochs=-1,
            enable_checkpointing=False,
            log_every_n_steps=toy_config.log_every_n_steps,
        ),
        callbacks=callbacks,
    )
    assigned = _assigned_devices(config, environment)
    if assigned == () and config.trainer.accelerator != "cpu":
        raise RuntimeError("toy performance routing requires one assigned device.")
    remaining = None if assigned is None else assigned[:1]
    plan = DataPlan(
        route=SourceRoute.TOY,
        formal_training=False,
        reason="bounded real-model toy performance probe; formal training did not start",
        source_factory=config.datamodule.source.factory,
        logical_id=None,
        stream_id=None,
        expected_samples=None,
        access_state=state,
        access_layout=None,
        access_detail=detail,
        devices=DevicePlan(
            assigned=assigned,
            training=remaining,
            generation=(),
            training_count=1,
            factories={},
        ),
        lineage_id=lineage_id,
        snapshot_id=snapshot_id,
        sample_count=sample_count,
        sealed=sealed,
    )
    return SourceResolution(
        resolved,
        plan,
        {_single_loader_name(config): dataset},
    )


def _s2st_generation_resolution(
    config: Any,
    source: Any,
    dataset: _LiveDataset,
    source_config: WorkspaceSourceConfig,
    *,
    state: str,
    detail: str,
    environment: Mapping[str, str],
    child: Mapping[str, object] | None,
) -> SourceResolution:
    if _externally_distributed(environment) and child is None:
        raise RuntimeError(
            "workspace generation device allocation must run before distributed "
            "launch; start scripts/train.py directly with CUDA_VISIBLE_DEVICES."
        )
    assigned = _s2st_visible_devices(config, environment, child=child)
    if assigned is None:
        raise RuntimeError(
            f"workspace access is not sufficient ({detail}); generation requires "
            "explicit CUDA_VISIBLE_DEVICES so factory and training devices can be "
            "separated before CUDA initialization."
        )
    generation = source.generate()
    lineage_id = _string(generation.lineage_id, "generation lineage_id")
    if lineage_id != dataset.lineage_id:
        raise RuntimeError(
            "workspace generation lineage does not match the accessed live dataset."
        )
    factories = _generation_factories(generation.factories)
    try:
        allocations, remaining = _factory_allocations(
            config.devices,
            factories,
            assigned,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            f"workspace access is not sufficient ({detail}); generation device "
            f"configuration is invalid: {error}"
        ) from error
    if child is not None:
        _validate_s2st_child_partition(
            child,
            allocations=allocations,
            remaining=remaining,
            current=_assigned_devices(config, environment),
        )
    log_root = Path(config.output_dir).expanduser() / "generation"
    processes = tuple(
        _FactoryProcess(
            name=name,
            command=_factory_command(factory, name=name),
            environment=_factory_environment(factory, name=name),
            devices=allocations[name],
            log=log_root / f"{name}.log",
        )
        for name, factory in factories.items()
    )
    service = GenerationService(processes)
    resolved = _live_config(config, training_count=len(remaining))
    marker = {
        "source_factory": source_config.factory,
        "lineage_id": lineage_id,
        "route": SourceRoute.GENERATE.value,
        "visible": list(assigned),
        "remaining": list(remaining),
        "factories": {
            name: list(devices) for name, devices in allocations.items()
        },
    }
    if environment is os.environ:
        os.environ[_CHILD_PLAN_ENV] = json.dumps(
            marker,
            separators=(",", ":"),
            sort_keys=True,
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(remaining)
    plan = _s2st_plan(
        route=SourceRoute.GENERATE,
        formal_training=True,
        reason=(
            "workspace snapshot is available and unsealed; generation resumes while "
            "training reads the current prefix"
            if dataset.snapshot_id is not None
            else "workspace initial snapshot is missing; generation starts before training waits"
        ),
        source_factory=source_config.factory,
        dataset=dataset,
        state=state,
        detail=detail,
        devices=DevicePlan(
            assigned=assigned,
            training=remaining,
            generation=tuple(
                device for devices in allocations.values() for device in devices
            ),
            training_count=len(remaining),
            factories=allocations,
        ),
        generation_logs={item.name: str(item.log) for item in processes},
    )
    return SourceResolution(
        resolved,
        plan,
        {_single_loader_name(config): cast(Dataset[Any], dataset)},
        service=service,
        live_dataset=dataset,
    )


def _live_config(config: Any, *, training_count: int) -> Any:
    callbacks = replace(
        config.callbacks,
        synthesis_sample=replace(config.callbacks.synthesis_sample, enabled=False),
    )
    return replace(
        config,
        datamodule=replace(
            config.datamodule,
            dataloader=_live_dataloader(config.datamodule.dataloader),
            encode_missing_codes=True,
            source=WorkspaceSourceConfig(),
            streaming=StreamingConfig(),
        ),
        trainer=replace(
            config.trainer,
            devices=training_count,
            strategy=("auto" if training_count == 1 else config.trainer.strategy),
        ),
        callbacks=callbacks,
    )


def _live_dataloader(config: Any) -> Any:
    return replace(
        config,
        num_workers=0,
        persistent_workers=False,
        costs=replace(config.costs, enabled=False),
    )


def _s2st_plan(
    *,
    route: SourceRoute,
    formal_training: bool,
    reason: str,
    source_factory: str | None,
    dataset: _LiveDataset,
    state: str,
    detail: str,
    devices: DevicePlan,
    generation_logs: Mapping[str, str] | None = None,
) -> DataPlan:
    return DataPlan(
        route=route,
        formal_training=formal_training,
        reason=reason,
        source_factory=source_factory,
        logical_id=None,
        stream_id=None,
        expected_samples=None,
        access_state=state,
        access_layout=None,
        access_detail=detail,
        devices=devices,
        lineage_id=dataset.lineage_id,
        snapshot_id=dataset.snapshot_id,
        sample_count=dataset.sample_count,
        sealed=dataset.sealed,
        generation_logs=generation_logs,
    )


def _s2st_dataset(access: Any, *, lineage_id: str) -> _LiveDataset:
    load = getattr(access, "load", None)
    if not callable(load):
        raise TypeError("workspace access is missing load().")
    dataset = load()
    if not isinstance(dataset, Dataset):
        raise TypeError("workspace access load() must return a Dataset.")
    actual_lineage = _string(
        getattr(dataset, "lineage_id", None),
        "workspace live dataset lineage_id",
    )
    if actual_lineage != lineage_id:
        raise RuntimeError(
            "workspace source and live dataset lineage identities do not match."
        )
    for name in (
        "acknowledge",
        "state_dict",
        "load_state_dict",
        "set_stop_requested",
        "close",
    ):
        if not callable(getattr(dataset, name, None)):
            raise TypeError(f"workspace live dataset is missing {name}().")
    _optional_string(getattr(dataset, "snapshot_id", None), "workspace snapshot_id")
    _non_negative_int(getattr(dataset, "sample_count", None), "workspace sample_count")
    _boolean(getattr(dataset, "sealed", None), "workspace dataset sealed")
    return cast(_LiveDataset, dataset)


def _generation_factories(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError("workspace generation factories must be a non-empty mapping.")
    result: dict[str, Any] = {}
    for name, factory in value.items():
        key = _string(name, "workspace generation factory name")
        result[key] = factory
    return result


def _factory_allocations(
    configured: object,
    factories: Mapping[str, Any],
    visible: tuple[str, ...],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    if not visible:
        raise RuntimeError("no visible CUDA device is available.")
    if not isinstance(configured, Mapping):
        raise TypeError("top-level devices must be a mapping.")
    names = set(factories)
    configured_names = set(configured)
    if unknown := configured_names - names:
        raise ValueError(
            "unknown generation factories in devices: " + ", ".join(sorted(unknown))
        )
    if missing := names - configured_names:
        raise ValueError(
            "devices is missing generation factories: " + ", ".join(sorted(missing))
        )
    used: set[int] = set()
    allocations: dict[str, tuple[str, ...]] = {}
    for name in factories:
        indices = configured[name]
        if not isinstance(indices, list):
            raise TypeError(f"devices.{name} must be a list of relative device ids.")
        if not indices:
            raise ValueError(f"devices.{name} must not be empty.")
        selected: list[str] = []
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(f"devices.{name} must contain integer device ids.")
            if index < 0 or index >= len(visible):
                raise ValueError(
                    f"devices.{name} contains out-of-range id {index}; "
                    f"CUDA_VISIBLE_DEVICES has {len(visible)} entries."
                )
            if index in used:
                raise ValueError(f"relative device id {index} is assigned more than once.")
            used.add(index)
            selected.append(visible[index])
        allocations[name] = tuple(selected)
    remaining = tuple(
        device for index, device in enumerate(visible) if index not in used
    )
    if not remaining:
        raise RuntimeError(
            "generation factories use every visible device; at least one must remain."
        )
    return allocations, remaining


def _factory_command(factory: Any, *, name: str) -> tuple[str, ...]:
    value = factory.command
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"generation factory {name!r} command must be a string sequence.")
    command = tuple(
        _string(part, f"generation factory {name!r} command item") for part in value
    )
    if not command:
        raise ValueError(f"generation factory {name!r} command must not be empty.")
    return command


def _factory_environment(factory: Any, *, name: str) -> dict[str, str]:
    value = factory.environment
    if not isinstance(value, Mapping):
        raise TypeError(f"generation factory {name!r} environment must be a mapping.")
    result: dict[str, str] = {}
    for key, item in value.items():
        resolved_key = _string(key, f"generation factory {name!r} environment key")
        resolved_value = _string(
            item,
            f"generation factory {name!r} environment value",
        )
        result[resolved_key] = resolved_value
    return result


def _known_device_count(config: Any, environment: Mapping[str, str]) -> int | None:
    assigned = _assigned_devices(config, environment)
    if assigned is not None:
        return len(assigned)
    if config.trainer.accelerator == "cpu":
        return 1
    devices = config.trainer.devices
    return devices if type(devices) is int and devices > 0 else None


def _s2st_visible_devices(
    config: Any,
    environment: Mapping[str, str],
    *,
    child: Mapping[str, object] | None,
) -> tuple[str, ...] | None:
    if child is None:
        return _assigned_devices(config, environment)
    visible = child.get("visible")
    if not isinstance(visible, list):
        raise TypeError("workspace child device plan visible must be a list.")
    return _device_tokens(visible)


def _s2st_child_plan(
    environment: Mapping[str, str],
    *,
    source_factory: str,
    lineage_id: str,
) -> Mapping[str, object] | None:
    if not _externally_distributed(environment):
        return None
    raw = environment.get(_CHILD_PLAN_ENV)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("workspace source child plan is invalid JSON.") from error
    if not isinstance(value, Mapping):
        raise TypeError("workspace source child plan must be an object.")
    plan = cast(Mapping[str, object], value)
    if plan.get("source_factory") != source_factory:
        raise RuntimeError("workspace source child plan factory does not match config.")
    if plan.get("lineage_id") != lineage_id:
        raise RuntimeError("workspace source child plan lineage does not match access.")
    if plan.get("route") != SourceRoute.GENERATE.value:
        raise RuntimeError("workspace source child plan route is unsupported.")
    return plan


def _validate_s2st_child_partition(
    child: Mapping[str, object],
    *,
    allocations: Mapping[str, tuple[str, ...]],
    remaining: tuple[str, ...],
    current: tuple[str, ...] | None,
) -> None:
    expected_factories = {
        name: list(devices) for name, devices in allocations.items()
    }
    if child.get("factories") != expected_factories:
        raise RuntimeError(
            "workspace generation factory devices changed across distributed ranks."
        )
    if child.get("remaining") != list(remaining):
        raise RuntimeError(
            "workspace remaining devices changed across distributed ranks."
        )
    if current != remaining:
        raise RuntimeError(
            "distributed child CUDA_VISIBLE_DEVICES does not match the parent remainder."
        )


def _resolve_legacy_workspace_source(
    config: Any,
    *,
    environment: Mapping[str, str] | None = None,
) -> SourceResolution:
    """Resolve access/generation/toy and apply the route to a parsed config."""

    source_config = config.datamodule.source
    if not source_config.enabled:
        if not config.datamodule.streaming.enabled:
            return SourceResolution(config, None, {})
        warnings.warn(
            "explicit datamodule.streaming configuration is deprecated; migrate "
            "to datamodule.source within one compatibility cycle.",
            DeprecationWarning,
            stacklevel=2,
        )
        return SourceResolution(
            config,
            DataPlan(
                route=SourceRoute.LEGACY,
                formal_training=True,
                reason="explicit legacy streaming configuration",
                source_factory=None,
                logical_id=None,
                stream_id=config.datamodule.streaming.stream_id,
                expected_samples=config.datamodule.streaming.expected_samples,
                access_state=None,
                access_layout=None,
                access_detail=None,
                devices=_unpartitioned_devices(config, environment),
            ),
            {},
        )

    values = os.environ if environment is None else environment
    source = _load_legacy_source(config, source_config)
    identity = _identity(source)
    access = source.access()
    state = _enum_value(access.state, "workspace access state")
    detail = _string(access.detail, "workspace access detail")
    if state == "invalid":
        raise RuntimeError(f"workspace source access is invalid: {detail}")
    if state not in {"ready", "missing"}:
        raise ValueError(f"unsupported workspace access state: {state!r}.")

    mode = source_config.mode
    child = _child_plan(values, source_factory=cast(str, source_config.factory))
    if child is not None:
        mode = SourceRoute.GENERATE.value
    if mode == SourceRoute.TOY.value:
        return _toy_resolution(config, source, identity, access, values)
    if mode == SourceRoute.ACCESS.value:
        if state != "ready":
            raise RuntimeError(f"workspace access was explicitly required: {detail}")
        return _access_resolution(
            config,
            identity,
            access,
            source_config,
            values,
        )
    if mode == SourceRoute.GENERATE.value:
        return _generation_resolution(
            config,
            source,
            identity,
            access,
            values,
            child=child,
        )
    if mode != "auto":
        raise ValueError(f"unsupported workspace source mode: {mode!r}.")
    if state == "ready":
        return _access_resolution(
            config,
            identity,
            access,
            source_config,
            values,
        )

    assigned = _assigned_devices(config, values)
    if assigned is not None and len(assigned) == 1:
        return _toy_resolution(config, source, identity, access, values)
    if assigned is None and _single_implicit_device(config):
        return _toy_resolution(config, source, identity, access, values)
    return _generation_resolution(
        config,
        source,
        identity,
        access,
        values,
        child=None,
    )


def emit_data_plan(plan: DataPlan | None, output_dir: Path) -> None:
    """Print and persist the rank-zero startup route record."""

    if plan is None or not _preflight_owner(os.environ):
        return
    payload = plan.as_dict()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_plan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _load_s2st_source(source_config: WorkspaceSourceConfig) -> Any:
    factory_path = cast(str, source_config.factory)
    factory = _source_factory(factory_path)
    return factory(**dict(source_config.options))


def _validate_s2st_source(source: Any, *, factory_path: str) -> None:
    for name in ("access", "toy"):
        if not callable(getattr(source, name, None)):
            raise TypeError(
                f"workspace source from {factory_path!r} is missing {name}()."
            )
    if not callable(getattr(source, "generate", None)):
        raise TypeError(
            f"workspace source from {factory_path!r} is missing generate()."
        )
    _string(getattr(source, "lineage_id", None), "workspace lineage_id")


def _source_factory(factory_path: str) -> Any:
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("workspace source factory must use 'module:attribute' syntax.")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"workspace source factory {factory_path!r} must be callable.")
    return factory


def _load_legacy_source(config: Any, source_config: WorkspaceSourceConfig) -> Any:
    factory_path = cast(str, source_config.factory)
    factory = _source_factory(factory_path)
    options = dict(source_config.options)
    output_codec = config.runtime.audio_output.tokenizer
    input_codec = _input_codec(config.runtime)
    _source_option(options, "split", config.datamodule.dataset.split)
    root = config.datamodule.dataset.root
    if root is not None:
        _source_option(options, "root", root)
    _runtime_option(options, "codec", output_codec)
    _runtime_option(options, "input_codec", input_codec)
    source = factory(**options)
    for name in ("access", "toy", "generation"):
        if not callable(getattr(source, name, None)):
            raise TypeError(
                f"workspace source from {factory_path!r} is missing {name}()."
            )
    for name, expected in (("codec", output_codec), ("input_codec", input_codec)):
        actual = getattr(source, name, None)
        if actual != expected:
            raise ValueError(
                f"workspace source {name} does not match the training runtime: "
                f"{actual!r} != {expected!r}."
            )
    return source


def _access_resolution(
    config: Any,
    identity: Any,
    access: Any,
    source_config: WorkspaceSourceConfig,
    environment: Mapping[str, str],
) -> SourceResolution:
    layout = _enum_value(access.layout, "workspace access layout")
    expected = _positive_int(access.sample_count, "workspace access sample_count")
    assigned = _assigned_devices(config, environment)
    training_count = _training_device_count(
        config,
        assigned,
        required=True,
    )
    assert training_count is not None
    if layout == "store":
        dataset = access.load()
        datamodule = replace(
            config.datamodule,
            source=WorkspaceSourceConfig(),
            streaming=StreamingConfig(),
        )
        training_datasets = {_single_loader_name(config): dataset}
        callbacks = replace(
            config.callbacks,
            synthesis_sample=replace(
                config.callbacks.synthesis_sample,
                enabled=False,
            ),
        )
    elif layout == "stream":
        _validate_world_size(expected, training_count)
        previous = config.datamodule.streaming
        streaming = StreamingConfig(
            enabled=True,
            root=str(access.root),
            stream_id=_string(identity.stream_id, "workspace stream_id"),
            expected_samples=expected,
            poll_seconds=previous.poll_seconds,
            status_seconds=previous.status_seconds,
            telemetry=replace(previous.telemetry, enabled=False),
        )
        datamodule = replace(
            config.datamodule,
            source=WorkspaceSourceConfig(),
            streaming=streaming,
        )
        training_datasets = {}
        callbacks = replace(
            config.callbacks,
            synthesis_sample=replace(
                config.callbacks.synthesis_sample,
                enabled=False,
            ),
        )
    else:
        raise ValueError(f"unsupported ready workspace access layout: {layout!r}.")
    trainer = replace(
        config.trainer,
        devices=training_count,
        strategy=("auto" if training_count == 1 else config.trainer.strategy),
    )
    resolved = replace(
        config,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
    )
    return SourceResolution(
        resolved,
        DataPlan(
            route=SourceRoute.ACCESS,
            formal_training=True,
            reason=f"workspace access is ready ({layout})",
            source_factory=source_config.factory,
            logical_id=_string(identity.logical_id, "workspace logical_id"),
            stream_id=_string(identity.stream_id, "workspace stream_id"),
            expected_samples=expected,
            access_state="ready",
            access_layout=layout,
            access_detail=_string(access.detail, "workspace access detail"),
            devices=DevicePlan(
                assigned=assigned,
                training=assigned,
                generation=(),
                training_count=training_count,
            ),
        ),
        training_datasets,
    )


def _toy_resolution(
    config: Any,
    source: Any,
    identity: Any,
    access: Any,
    environment: Mapping[str, str],
) -> SourceResolution:
    toy = source.toy()
    dataset = toy.load()
    toy_config = config.datamodule.source.toy
    max_steps = toy_config.warmup_steps + toy_config.measure_window_steps
    output_subdir = f"{config.output_subdir.rstrip('/')}/toy-perf"
    output_dir = str(Path(config.repo_output_root).expanduser() / output_subdir)
    callbacks = replace(
        config.callbacks,
        task_sample=replace(config.callbacks.task_sample, enabled=False),
        synthesis_sample=replace(config.callbacks.synthesis_sample, enabled=False),
        text_retention=replace(config.callbacks.text_retention, enabled=False),
        gradient_probe=replace(config.callbacks.gradient_probe, enabled=False),
        performance=replace(
            config.callbacks.performance,
            enabled=True,
            log_every_n_steps=toy_config.log_every_n_steps,
            warmup_steps=toy_config.warmup_steps,
            measure_window_steps=toy_config.measure_window_steps,
            stop_after_measurement=True,
        ),
    )
    resolved = replace(
        config,
        output_subdir=output_subdir,
        output_dir=output_dir,
        logging=_toy_logging(config.logging, output_dir, output_subdir),
        datamodule=replace(
            config.datamodule,
            source=WorkspaceSourceConfig(),
            streaming=StreamingConfig(),
        ),
        train=replace(
            config.train,
            max_steps=max_steps,
            ckpt_path=None,
            auto_resume=False,
        ),
        validation=replace(config.validation, enabled=False),
        trainer=replace(
            config.trainer,
            devices=1,
            strategy="auto",
            max_epochs=-1,
            enable_checkpointing=False,
            log_every_n_steps=toy_config.log_every_n_steps,
        ),
        callbacks=callbacks,
    )
    return SourceResolution(
        resolved,
        DataPlan(
            route=SourceRoute.TOY,
            formal_training=False,
            reason=(
                "bounded real-model toy performance probe; formal training did not start"
            ),
            source_factory=config.datamodule.source.factory,
            logical_id=_string(identity.logical_id, "workspace logical_id"),
            stream_id=_string(identity.stream_id, "workspace stream_id"),
            expected_samples=len(dataset),
            access_state=_enum_value(access.state, "workspace access state"),
            access_layout=_optional_enum_value(
                access.layout,
                "workspace access layout",
            ),
            access_detail=_string(access.detail, "workspace access detail"),
            devices=_toy_devices(config, environment),
        ),
        {_single_loader_name(config): dataset},
    )


def _generation_resolution(
    config: Any,
    source: Any,
    identity: Any,
    access: Any,
    environment: Mapping[str, str],
    *,
    child: Mapping[str, object] | None,
) -> SourceResolution:
    logical_id = _string(identity.logical_id, "workspace logical_id")
    if child is not None and child.get("logical_id") != logical_id:
        raise RuntimeError(
            "workspace generation child plan logical identity changed across ranks."
        )
    if _externally_distributed(environment) and child is None:
        raise RuntimeError(
            "workspace generation device partitioning must run before distributed "
            "launch; start scripts/train.py directly with CUDA_VISIBLE_DEVICES."
        )
    assigned = _assigned_devices(config, environment, child=child)
    if assigned is None:
        raise RuntimeError(
            "workspace generation requires an explicit CUDA_VISIBLE_DEVICES assignment "
            "so physical generation and training devices can be separated before CUDA "
            "initialization."
        )
    generation = source.generation()
    if not bool(generation.available):
        reason = getattr(generation, "unavailable_reason", None)
        raise RuntimeError(
            "workspace generation is unavailable for this source"
            + (f": {reason}" if reason else ".")
        )
    recipe = _placement_recipe(generation.placement_recipes, len(assigned))
    generation_count = _positive_int(
        recipe.generation_devices,
        "generation recipe devices",
    )
    training_count = len(assigned) - generation_count
    training_devices = assigned[:training_count]
    generation_devices = assigned[training_count:]
    if not training_devices:
        raise RuntimeError("generation placement left no device for training.")
    if child is not None:
        current_devices = _assigned_devices(config, environment)
        if current_devices != training_devices:
            raise RuntimeError(
                "workspace generation child CUDA_VISIBLE_DEVICES does not match "
                "the parent training partition."
            )
    expected = _positive_int(
        (
            child.get("expected_samples")
            if child is not None
            else generation.resolve_expected_samples()
        ),
        "workspace generation expected_samples",
    )
    _validate_world_size(expected, training_count)
    if child is not None:
        if child.get("recipe") != recipe.name:
            raise RuntimeError(
                "workspace generation child placement recipe changed across ranks."
            )
        if child.get("training") != list(training_devices) or child.get(
            "generation"
        ) != list(generation_devices):
            raise RuntimeError(
                "workspace generation child device partition changed across ranks."
            )
    producer_environment = dict(generation.environment(recipe))
    producer_environment["CUDA_VISIBLE_DEVICES"] = ",".join(generation_devices)
    previous = config.datamodule.streaming
    streaming = StreamingConfig(
        enabled=True,
        root=str(source.root),
        stream_id=_string(identity.stream_id, "workspace stream_id"),
        expected_samples=expected,
        poll_seconds=previous.poll_seconds,
        status_seconds=previous.status_seconds,
        producer_factory=_SUBPROCESS_FACTORY,
        producer_options={
            "command": list(generation.command),
            "environment": producer_environment,
        },
        telemetry=previous.telemetry,
    )
    trainer = replace(
        config.trainer,
        devices=len(training_devices),
        strategy=("auto" if len(training_devices) == 1 else config.trainer.strategy),
    )
    resolved = replace(
        config,
        datamodule=replace(
            config.datamodule,
            source=WorkspaceSourceConfig(),
            streaming=streaming,
        ),
        trainer=trainer,
    )
    stage_devices = {
        stage: generation_devices[index]
        for stage, index in recipe.stage_devices.items()
    }
    device_plan = DevicePlan(
        assigned=assigned,
        training=training_devices,
        generation=generation_devices,
        training_count=training_count,
        recipe=_string(recipe.name, "generation recipe name"),
        stage_devices=stage_devices,
    )
    marker = {
        "source_factory": config.datamodule.source.factory,
        "logical_id": logical_id,
        "route": SourceRoute.GENERATE.value,
        "assigned": list(assigned),
        "training": list(training_devices),
        "generation": list(generation_devices),
        "recipe": recipe.name,
        "expected_samples": expected,
    }
    if environment is os.environ:
        os.environ[_CHILD_PLAN_ENV] = json.dumps(
            marker,
            separators=(",", ":"),
            sort_keys=True,
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(training_devices)
    root = Path(source.root)
    return SourceResolution(
        resolved,
        DataPlan(
            route=SourceRoute.GENERATE,
            formal_training=True,
            reason="workspace access is missing; generation and training are isolated",
            source_factory=config.datamodule.source.factory,
            logical_id=logical_id,
            stream_id=_string(identity.stream_id, "workspace stream_id"),
            expected_samples=expected,
            access_state=_enum_value(access.state, "workspace access state"),
            access_layout=_optional_enum_value(
                access.layout,
                "workspace access layout",
            ),
            access_detail=_string(access.detail, "workspace access detail"),
            devices=device_plan,
            generation_profile=_string(
                generation.profile,
                "generation profile",
            ),
            generation_entrypoint=_string(
                generation.entrypoint,
                "generation entrypoint",
            ),
            generation_log=str(root / "producer.log"),
            generation_telemetry=str(root / "producer_telemetry.jsonl"),
        ),
        {},
    )


def _identity(source: Any) -> Any:
    identity = source.identity
    for name in ("logical_id", "stream_id"):
        _string(getattr(identity, name, None), f"workspace source identity {name}")
    return identity


def _runtime_option(options: dict[str, object], name: str, expected: str) -> None:
    configured = options.get(name)
    if configured is not None and configured != expected:
        raise ValueError(
            f"workspace source option {name} conflicts with the training runtime: "
            f"{configured!r} != {expected!r}."
        )
    options[name] = expected


def _source_option(options: dict[str, object], name: str, expected: str) -> None:
    configured = options.get(name)
    if configured is not None and configured != expected:
        raise ValueError(
            f"workspace source option {name} conflicts with the data config: "
            f"{configured!r} != {expected!r}."
        )
    options[name] = expected


def _input_codec(runtime: Any) -> str:
    output = _string(runtime.audio_output.tokenizer, "runtime output codec")
    input_audio = runtime.audio_input
    if input_audio is None:
        return output
    if input_audio.composed:
        return _string(
            input_audio.stream(AudioStream.SEMANTIC).tokenizer,
            "runtime semantic input codec",
        )
    return output if input_audio.tokenizer is None else _string(
        input_audio.tokenizer,
        "runtime input codec",
    )


def _placement_recipe(recipes: Sequence[Any], total_devices: int) -> Any:
    if total_devices < 2:
        raise RuntimeError(
            "workspace generation requires disjoint generation and training devices; "
            f"only {total_devices} assigned device was provided."
        )
    for recipe in recipes:
        generation = _positive_int(
            recipe.generation_devices,
            "generation recipe devices",
        )
        training = _positive_int(
            recipe.minimum_training_devices,
            "generation recipe minimum training devices",
        )
        if generation + training <= total_devices:
            return recipe
    summary = ", ".join(
        f"{recipe.name}={recipe.generation_devices}+{recipe.minimum_training_devices}"
        for recipe in recipes
    )
    raise RuntimeError(
        "no workspace generation placement recipe fits the assigned devices "
        f"({total_devices}); available recipes: {summary or 'none'}."
    )


def _assigned_devices(
    config: Any,
    environment: Mapping[str, str],
    *,
    child: Mapping[str, object] | None = None,
) -> tuple[str, ...] | None:
    if child is not None:
        value = child.get("assigned")
        if not isinstance(value, list):
            raise TypeError("workspace child device plan assigned must be a list.")
        return _device_tokens(value)
    raw = environment.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    if not raw.strip() or raw.strip() == "-1":
        return ()
    return _device_tokens(raw.split(","))


def _device_tokens(values: Sequence[object]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("CUDA_VISIBLE_DEVICES must contain non-empty tokens.")
        result.append(value.strip())
    if len(set(result)) != len(result):
        raise ValueError("CUDA_VISIBLE_DEVICES must not contain duplicate devices.")
    return tuple(result)


def _unpartitioned_devices(
    config: Any,
    environment: Mapping[str, str] | None,
) -> DevicePlan:
    values = os.environ if environment is None else environment
    assigned = _assigned_devices(config, values)
    return DevicePlan(
        assigned=assigned,
        training=assigned,
        generation=(),
        training_count=_training_device_count(
            config,
            assigned,
            required=False,
        ),
    )


def _toy_devices(config: Any, environment: Mapping[str, str]) -> DevicePlan:
    assigned = _assigned_devices(config, environment)
    if assigned == () and config.trainer.accelerator != "cpu":
        raise RuntimeError(
            "toy performance routing requires one assigned training device."
        )
    training = None if assigned is None else assigned[:1]
    return DevicePlan(
        assigned=assigned,
        training=training,
        generation=(),
        training_count=1,
    )


def _toy_logging(logging: Any, output_dir: str, output_subdir: str) -> Any:
    if logging.name == "csv":
        return replace(logging, save_dir=output_dir)
    return replace(logging, run_name=output_subdir)


def _training_device_count(
    config: Any,
    assigned: tuple[str, ...] | None,
    *,
    required: bool,
) -> int | None:
    if assigned is not None:
        if not assigned:
            if required:
                raise RuntimeError(
                    "workspace source routing has no assigned training device."
                )
            return 0
        return len(assigned)
    devices = config.trainer.devices
    if type(devices) is int and devices > 0:
        return devices
    if config.trainer.accelerator == "cpu" and devices == "auto":
        return 1
    if required:
        raise RuntimeError(
            "workspace access requires CUDA_VISIBLE_DEVICES or an explicit positive "
            "trainer.devices count so the training world size is known before model "
            "initialization."
        )
    return None


def _validate_world_size(expected_samples: int, world_size: int) -> None:
    if expected_samples % world_size != 0:
        raise ValueError(
            "workspace source expected_samples must be divisible by the training "
            f"world size: {expected_samples} % {world_size} != 0."
        )


def _single_implicit_device(config: Any) -> bool:
    if config.trainer.accelerator == "cpu":
        return True
    devices = config.trainer.devices
    return type(devices) is int and devices == 1


def _single_loader_name(config: Any) -> str:
    names = tuple(config.loader_plan.loaders)
    if len(names) != 1:
        raise RuntimeError("workspace source routing requires exactly one loader.")
    return names[0]


def _child_plan(
    environment: Mapping[str, str],
    *,
    source_factory: str,
) -> Mapping[str, object] | None:
    if not _externally_distributed(environment):
        return None
    raw = environment.get(_CHILD_PLAN_ENV)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("workspace source child plan is invalid JSON.") from error
    if not isinstance(value, Mapping):
        raise TypeError("workspace source child plan must be an object.")
    plan = cast(Mapping[str, object], value)
    if plan.get("source_factory") != source_factory:
        raise RuntimeError("workspace source child plan factory does not match config.")
    if plan.get("route") != SourceRoute.GENERATE.value:
        raise RuntimeError("workspace source child plan route is unsupported.")
    return plan


def _externally_distributed(environment: Mapping[str, str]) -> bool:
    return (
        "LOCAL_RANK" in environment
        or _environment_int(environment, "WORLD_SIZE", default=1) > 1
    )


def _preflight_owner(environment: Mapping[str, str]) -> bool:
    return _environment_int(environment, "LOCAL_RANK", default=0) == 0


def _environment_int(
    environment: Mapping[str, str],
    name: str,
    *,
    default: int,
) -> int:
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error


def _enum_value(value: object, name: str) -> str:
    resolved = getattr(value, "value", value)
    return _string(resolved, name)


def _optional_enum_value(value: object, name: str) -> str | None:
    return None if value is None else _enum_value(value, name)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def _optional_string(value: object, name: str) -> str | None:
    return None if value is None else _string(value, name)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


__all__ = [
    "DataPlan",
    "DevicePlan",
    "GenerationServiceProvider",
    "LiveS2STCursor",
    "SourceResolution",
    "SourceRoute",
    "emit_data_plan",
    "resolve_workspace_source",
]
