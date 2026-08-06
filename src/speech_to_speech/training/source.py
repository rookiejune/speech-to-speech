"""Resolve a workspace data source before CUDA or model initialization."""

from __future__ import annotations

import importlib
import json
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from torch.utils.data import Dataset

from .._compat import StrEnum, auto
from ..audio import AudioStream
from ..datamodule.config import (
    StreamingConfig,
    WorkspaceSourceConfig,
)


_CHILD_PLAN_ENV = "SPEECH_TO_SPEECH_SOURCE_CHILD_PLAN"
_SUBPROCESS_FACTORY = "speech_to_speech.synthesis.process:controller"


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

    def as_dict(self) -> dict[str, object]:
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

    def as_dict(self) -> dict[str, object]:
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


def resolve_workspace_source(
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
    source = _load_source(config, source_config)
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


def _load_source(config: Any, source_config: WorkspaceSourceConfig) -> Any:
    factory_path = cast(str, source_config.factory)
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("workspace source factory must use 'module:attribute' syntax.")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError(f"workspace source factory {factory_path!r} must be callable.")
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


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


__all__ = [
    "DataPlan",
    "DevicePlan",
    "SourceResolution",
    "SourceRoute",
    "emit_data_plan",
    "resolve_workspace_source",
]
