from __future__ import annotations

import json
import os
import unittest
from collections.abc import Iterator, Mapping
from dataclasses import replace
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

from anytrain.lightning import ManagedServiceCallback
from torch.utils.data import Dataset, IterableDataset

from _config_helpers import _train
from scripts import train as train_script
from speech_to_speech.datamodule.config import StreamingConfig
from speech_to_speech.training.source import (
    LiveS2STCursor,
    SourceRoute,
    resolve_workspace_source,
)


_CHILD_PLAN_ENV = "SPEECH_TO_SPEECH_SOURCE_CHILD_PLAN"
_FACTORY = "fake.workspace:source"


class _Dataset(Dataset[object]):
    def __init__(self, count: int = 8) -> None:
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> object:
        if index < 0 or index >= self.count:
            raise IndexError(index)
        return {"index": index}


class _Live(IterableDataset[object]):
    def __init__(
        self,
        *,
        snapshot_id: str | None,
        sample_count: int,
        sealed: bool,
        lineage_id: str = "lineage-v1",
    ) -> None:
        super().__init__()
        self.lineage_id = lineage_id
        self.snapshot_id = snapshot_id
        self.sample_count = sample_count
        self.sealed = sealed
        self.acknowledged = 0
        self.loaded_states: list[Mapping[str, object]] = []
        self.stop_requested: Any | None = None
        self.closed = 0

    def __iter__(self) -> Iterator[object]:
        yield from ({"index": index} for index in range(self.sample_count))

    def acknowledge(self) -> None:
        self.acknowledged += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "lineage_id": self.lineage_id,
            "snapshot_id": self.snapshot_id,
            "pair_cursor": self.acknowledged,
        }

    def load_state_dict(self, value: Mapping[str, object]) -> None:
        self.loaded_states.append(value)

    def set_stop_requested(self, predicate: Any | None) -> None:
        self.stop_requested = predicate

    def close(self) -> None:
        self.closed += 1


class _Access:
    def __init__(
        self,
        state: str,
        dataset: _Live,
        *,
        detail: str | None = None,
        sealed: bool | None = None,
    ) -> None:
        self.state = state
        self.detail = detail or f"access is {state}"
        self.dataset = dataset
        self.sealed = dataset.sealed if sealed is None else sealed
        self.load = Mock(return_value=dataset)


class _Generation:
    def __init__(
        self,
        *,
        lineage_id: str = "lineage-v1",
        factories: Mapping[str, object] | None = None,
    ) -> None:
        self.lineage_id = lineage_id
        self.factories = factories or {
            "translation": SimpleNamespace(
                command=("/tmp/translation-producer",),
                environment={"FACTORY_KIND": "translation"},
            ),
            "tts": SimpleNamespace(
                command=("/tmp/tts-producer",),
                environment={"FACTORY_KIND": "tts"},
            ),
        }


class _Source:
    lineage_id = "lineage-v1"

    def __init__(
        self,
        access: _Access,
        *,
        generation: _Generation | None = None,
        toy_count: int = 64,
    ) -> None:
        self._access = access
        self._generation = generation or _Generation()
        self._toy_dataset = _Dataset(toy_count)
        self.access = Mock(return_value=access)
        self.generate = Mock(return_value=self._generation)
        self.toy = Mock(
            return_value=SimpleNamespace(load=Mock(return_value=self._toy_dataset))
        )


class WorkspaceSourceRoutingTest(unittest.TestCase):
    def test_sealed_snapshot_is_pure_access_and_uses_every_visible_device(self) -> None:
        config = _config(devices={"translation": [0], "tts": [1]})
        live = _Live(snapshot_id="snapshot-3", sample_count=12, sealed=True)
        source = _Source(_Access("ready", live))

        resolution, factory = _resolve(
            config,
            source,
            environment={"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b,GPU-c"},
        )

        self.assertIs(resolution.plan.route, SourceRoute.ACCESS)
        self.assertIs(resolution.training_datasets["s2st"], live)
        self.assertIs(resolution.live_dataset, live)
        self.assertIsNone(resolution.service)
        self.assertEqual(resolution.config.trainer.devices, 3)
        self.assertEqual(resolution.plan.devices.factories, {})
        self.assertEqual(
            resolution.plan.devices.training,
            ("GPU-a", "GPU-b", "GPU-c"),
        )
        self.assertFalse(resolution.config.datamodule.source.enabled)
        self.assertFalse(resolution.config.datamodule.streaming.enabled)
        self.assertTrue(resolution.config.datamodule.encode_missing_codes)
        self.assertFalse(resolution.config.callbacks.synthesis_sample.enabled)
        source.generate.assert_not_called()
        factory.assert_called_once_with()

    def test_unsealed_snapshot_single_device_trains_current_prefix_only(self) -> None:
        config = _config()
        live = _Live(snapshot_id="snapshot-1", sample_count=8, sealed=False)
        source = _Source(_Access("ready", live))

        resolution, _ = _resolve(
            config,
            source,
            environment={"CUDA_VISIBLE_DEVICES": "GPU-only"},
        )

        self.assertIs(resolution.plan.route, SourceRoute.ACCESS)
        self.assertEqual(resolution.config.trainer.devices, 1)
        self.assertIs(resolution.training_datasets["s2st"], live)
        source.generate.assert_not_called()

    def test_unsealed_snapshot_multi_device_resumes_factories_and_trains_now(self) -> None:
        config = _config(devices={"translation": [0], "tts": [1, 2]})
        live = _Live(snapshot_id="snapshot-4", sample_count=32, sealed=False)
        source = _Source(_Access("ready", live))
        environment = {
            "CUDA_VISIBLE_DEVICES": "gen-t,gen-v0,gen-v1,train-0,train-1"
        }

        resolution, _ = _resolve(config, source, environment=environment)

        self.assertIs(resolution.plan.route, SourceRoute.GENERATE)
        self.assertEqual(resolution.config.trainer.devices, 2)
        self.assertEqual(
            resolution.plan.devices.factories,
            {"translation": ("gen-t",), "tts": ("gen-v0", "gen-v1")},
        )
        self.assertEqual(resolution.plan.devices.training, ("train-0", "train-1"))
        self.assertIs(resolution.training_datasets["s2st"], live)
        self.assertIsNotNone(resolution.service)
        assert resolution.service is not None
        self.assertEqual(
            [(item.name, item.devices) for item in resolution.service.factories],
            [
                ("translation", ("gen-t",)),
                ("tts", ("gen-v0", "gen-v1")),
            ],
        )
        self.assertEqual(
            resolution.service.factories[0].environment,
            {"FACTORY_KIND": "translation"},
        )
        self.assertIn("current prefix", resolution.plan.reason)
        self.assertEqual(
            environment["CUDA_VISIBLE_DEVICES"],
            "gen-t,gen-v0,gen-v1,train-0,train-1",
        )

    def test_missing_single_device_uses_bounded_real_model_toy_probe(self) -> None:
        config = _config()
        live = _Live(snapshot_id=None, sample_count=0, sealed=False)
        source = _Source(_Access("missing", live), toy_count=64)

        resolution, _ = _resolve(
            config,
            source,
            environment={"CUDA_VISIBLE_DEVICES": "GPU-only"},
        )

        resolved = resolution.config
        self.assertIs(resolution.plan.route, SourceRoute.TOY)
        self.assertFalse(resolution.plan.formal_training)
        self.assertEqual(resolved.trainer.devices, 1)
        self.assertEqual(resolved.trainer.strategy, "auto")
        self.assertFalse(resolved.trainer.enable_checkpointing)
        self.assertFalse(resolved.validation.enabled)
        self.assertTrue(resolved.datamodule.encode_missing_codes)
        self.assertTrue(resolved.callbacks.performance.enabled)
        self.assertTrue(resolved.callbacks.performance.stop_after_measurement)
        self.assertEqual(
            resolved.train.max_steps,
            resolved.callbacks.performance.warmup_steps
            + resolved.callbacks.performance.measure_window_steps,
        )
        self.assertTrue(resolved.output_subdir.endswith("/toy-perf"))
        self.assertIs(resolution.training_datasets["s2st"], source._toy_dataset)
        self.assertEqual(live.closed, 1)
        source.generate.assert_not_called()

    def test_preflight_forces_live_and_toy_dataloaders_to_one_process(self) -> None:
        base = _config()
        dataloader = replace(
            base.datamodule.dataloader,
            num_workers=4,
            persistent_workers=True,
            costs=replace(base.datamodule.dataloader.costs, enabled=True),
        )
        config = replace(
            base,
            datamodule=replace(base.datamodule, dataloader=dataloader),
        )
        ready = _Source(
            _Access(
                "ready",
                _Live(snapshot_id="snapshot-1", sample_count=8, sealed=True),
            )
        )

        accessed, _ = _resolve(
            config,
            ready,
            environment={"CUDA_VISIBLE_DEVICES": "0"},
        )
        self.assertEqual(accessed.config.datamodule.dataloader.num_workers, 0)
        self.assertFalse(accessed.config.datamodule.dataloader.persistent_workers)
        self.assertFalse(accessed.config.datamodule.dataloader.costs.enabled)

        missing = _Source(
            _Access("missing", _Live(snapshot_id=None, sample_count=0, sealed=False))
        )
        toy, _ = _resolve(
            config,
            missing,
            environment={"CUDA_VISIBLE_DEVICES": "0"},
        )
        self.assertEqual(toy.config.datamodule.dataloader.num_workers, 0)
        self.assertFalse(toy.config.datamodule.dataloader.persistent_workers)
        self.assertFalse(toy.config.datamodule.dataloader.costs.enabled)

    def test_missing_multi_device_starts_factories_and_keeps_live_facade(self) -> None:
        config = _config(devices={"translation": [0], "tts": [1]})
        live = _Live(snapshot_id=None, sample_count=0, sealed=False)
        source = _Source(_Access("missing", live, detail="catalog is absent"))

        resolution, _ = _resolve(
            config,
            source,
            environment={"CUDA_VISIBLE_DEVICES": "translate,voice,train"},
        )

        self.assertIs(resolution.plan.route, SourceRoute.GENERATE)
        self.assertEqual(resolution.config.trainer.devices, 1)
        self.assertIs(resolution.training_datasets["s2st"], live)
        self.assertIn("initial snapshot", resolution.plan.reason)
        self.assertEqual(
            resolution.plan.devices.factories,
            {"translation": ("translate",), "tts": ("voice",)},
        )

    def test_invalid_access_never_falls_back(self) -> None:
        for mode in ("auto", "access", "generate", "toy"):
            with self.subTest(mode=mode):
                config = _config(mode=mode)
                live = _Live(snapshot_id=None, sample_count=0, sealed=False)
                access = _Access("invalid", live, detail="catalog digest mismatch")
                source = _Source(access)

                with self.assertRaisesRegex(RuntimeError, "catalog digest mismatch"):
                    _resolve(
                        config,
                        source,
                        environment={"CUDA_VISIBLE_DEVICES": "0,1,2"},
                    )

                access.load.assert_not_called()
                source.toy.assert_not_called()
                source.generate.assert_not_called()

    def test_access_exception_is_not_converted_to_generation_or_toy(self) -> None:
        config = _config()
        source = _Source(
            _Access("missing", _Live(snapshot_id=None, sample_count=0, sealed=False))
        )
        source.access.side_effect = RuntimeError("invalid final catalog")

        with self.assertRaisesRegex(RuntimeError, "invalid final catalog"):
            _resolve(
                config,
                source,
                environment={"CUDA_VISIBLE_DEVICES": "0,1,2"},
            )

        source.generate.assert_not_called()
        source.toy.assert_not_called()

    def test_access_state_seal_and_snapshot_must_match_live_facade(self) -> None:
        cases = (
            (
                _Access(
                    "ready",
                    _Live(snapshot_id=None, sample_count=0, sealed=False),
                ),
                "ready without",
            ),
            (
                _Access(
                    "missing",
                    _Live(snapshot_id="snapshot-1", sample_count=8, sealed=False),
                ),
                "missing with",
            ),
            (
                _Access(
                    "ready",
                    _Live(snapshot_id="snapshot-1", sample_count=8, sealed=True),
                    sealed=False,
                ),
                "disagree",
            ),
        )
        for access, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                RuntimeError,
                message,
            ):
                _resolve(
                    _config(),
                    _Source(access),
                    environment={"CUDA_VISIBLE_DEVICES": "0"},
                )

    def test_generation_device_errors_include_access_detail(self) -> None:
        cases = (
            ({"translation": [0], "tts": [1], "codec": [2]}, "unknown"),
            ({"translation": [0]}, "missing"),
            ({"translation": [0], "tts": []}, "must not be empty"),
            ({"translation": [0], "tts": [0]}, "more than once"),
            ({"translation": [0], "tts": [3]}, "out-of-range"),
            ({"translation": [0], "tts": [1, 2]}, "at least one must remain"),
        )
        for devices, message in cases:
            with self.subTest(devices=devices):
                config = _config(devices=devices)
                source = _Source(
                    _Access(
                        "missing",
                        _Live(snapshot_id=None, sample_count=0, sealed=False),
                        detail="no final snapshot yet",
                    )
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"no final snapshot yet.*{message}",
                ):
                    _resolve(
                        config,
                        source,
                        environment={"CUDA_VISIBLE_DEVICES": "0,1,2"},
                    )

    def test_generation_requires_explicit_visibility_when_device_count_is_unknown(self) -> None:
        config = _config(devices={"translation": [0], "tts": [1]})
        source = _Source(
            _Access("missing", _Live(snapshot_id=None, sample_count=0, sealed=False))
        )

        with self.assertRaisesRegex(RuntimeError, "explicit CUDA_VISIBLE_DEVICES"):
            _resolve(config, source, environment={})

    def test_source_factory_receives_only_declared_workspace_options(self) -> None:
        config = _config(options={"languages": ["zh", "en"], "seed": 7})
        source = _Source(
            _Access(
                "ready",
                _Live(snapshot_id="snapshot-1", sample_count=8, sealed=True),
            )
        )

        _, factory = _resolve(
            config,
            source,
            environment={"CUDA_VISIBLE_DEVICES": "0"},
        )

        factory.assert_called_once_with(languages=["zh", "en"], seed=7)

    def test_generation_lineage_must_match_live_dataset(self) -> None:
        config = _config(devices={"translation": [0], "tts": [1]})
        source = _Source(
            _Access("missing", _Live(snapshot_id=None, sample_count=0, sealed=False)),
            generation=_Generation(lineage_id="other-lineage"),
        )

        with self.assertRaisesRegex(RuntimeError, "lineage"):
            _resolve(
                config,
                source,
                environment={"CUDA_VISIBLE_DEVICES": "0,1,2"},
            )

    def test_sealed_snapshot_remains_access_in_explicit_generate_mode(self) -> None:
        config = _config(
            mode="generate",
            devices={"translation": [0], "tts": [1]},
        )
        source = _Source(
            _Access(
                "ready",
                _Live(snapshot_id="snapshot-9", sample_count=20, sealed=True),
            )
        )

        resolution, _ = _resolve(
            config,
            source,
            environment={"CUDA_VISIBLE_DEVICES": "0,1,2"},
        )

        self.assertIs(resolution.plan.route, SourceRoute.ACCESS)
        source.generate.assert_not_called()

    def test_explicit_access_and_toy_modes_override_unsealed_auto_route(self) -> None:
        live = _Live(snapshot_id="snapshot-2", sample_count=10, sealed=False)
        access_source = _Source(_Access("ready", live))
        accessed, _ = _resolve(
            _config(mode="access"),
            access_source,
            environment={"CUDA_VISIBLE_DEVICES": "0,1,2"},
        )
        self.assertIs(accessed.plan.route, SourceRoute.ACCESS)
        access_source.generate.assert_not_called()

        toy_source = _Source(_Access("ready", live))
        toy, _ = _resolve(
            _config(mode="toy"),
            toy_source,
            environment={"CUDA_VISIBLE_DEVICES": "0,1,2"},
        )
        self.assertIs(toy.plan.route, SourceRoute.TOY)

        missing = _Source(
            _Access("missing", _Live(snapshot_id=None, sample_count=0, sealed=False))
        )
        with self.assertRaisesRegex(RuntimeError, "explicitly required"):
            _resolve(
                _config(mode="access"),
                missing,
                environment={"CUDA_VISIBLE_DEVICES": "0"},
            )

    def test_parent_factory_partition_is_reused_by_distributed_children(self) -> None:
        config = _config(devices={"translation": [0], "tts": [1, 2]})
        source = _Source(
            _Access("missing", _Live(snapshot_id=None, sample_count=0, sealed=False))
        )
        with patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "g0,g1,g2,t0,t1"},
            clear=True,
        ):
            parent, _ = _resolve(config, source)
            marker = os.environ[_CHILD_PLAN_ENV]
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "t0,t1")
            payload = json.loads(marker)
            self.assertEqual(
                payload["factories"],
                {"translation": ["g0"], "tts": ["g1", "g2"]},
            )
            self.assertEqual(payload["remaining"], ["t0", "t1"])
            self.assertNotIn("recipe", payload)
            self.assertNotIn("expected_samples", payload)

            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "2"
            child, _ = _resolve(config, source)

            self.assertEqual(child.plan.devices, parent.plan.devices)
            self.assertEqual(child.config.trainer.devices, 2)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "t0,t1")

    def test_distributed_child_does_not_resume_factories_after_catalog_seals(self) -> None:
        config = _config(devices={"translation": [0], "tts": [1]})
        source = _Source(
            _Access(
                "ready",
                _Live(snapshot_id="snapshot-final", sample_count=12, sealed=True),
            )
        )
        marker = json.dumps(
            {
                "source_factory": _FACTORY,
                "lineage_id": "lineage-v1",
                "route": "generate",
                "visible": ["g0", "g1", "t0"],
                "remaining": ["t0"],
                "factories": {"translation": ["g0"], "tts": ["g1"]},
            }
        )

        resolution, _ = _resolve(
            config,
            source,
            environment={
                "CUDA_VISIBLE_DEVICES": "t0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                _CHILD_PLAN_ENV: marker,
            },
        )

        self.assertIs(resolution.plan.route, SourceRoute.ACCESS)
        self.assertIsNone(resolution.service)
        source.generate.assert_not_called()

    def test_live_cursor_commits_only_when_optimizer_step_advances(self) -> None:
        live = _Live(snapshot_id="snapshot-1", sample_count=8, sealed=False)
        service = Mock()
        callback = LiveS2STCursor(live, service)
        trainer = SimpleNamespace(
            global_step=4,
            received_sigterm=False,
            is_global_zero=True,
        )

        callback.on_fit_start(cast(Any, trainer), cast(Any, object()))
        stop_requested = live.stop_requested
        assert callable(stop_requested)
        self.assertFalse(stop_requested())
        service.check.assert_called_once_with(owner=True)
        callback.on_train_start(cast(Any, trainer), cast(Any, object()))
        callback.on_train_batch_end(
            cast(Any, trainer), cast(Any, object()), None, None, 0
        )
        self.assertEqual(live.acknowledged, 0)

        trainer.global_step = 5
        callback.on_train_batch_end(
            cast(Any, trainer), cast(Any, object()), None, None, 1
        )
        self.assertEqual(live.acknowledged, 1)
        state = callback.state_dict()
        callback.load_state_dict(state)
        self.assertEqual(live.loaded_states, [state])
        callback.on_fit_end(cast(Any, trainer), cast(Any, object()))
        callback.on_exception(
            cast(Any, trainer),
            cast(Any, object()),
            RuntimeError("ignored"),
        )
        self.assertIsNone(live.stop_requested)
        self.assertEqual(live.closed, 0)
        service.close.assert_not_called()

    def test_new_lifecycle_uses_generation_service_and_live_cursor(self) -> None:
        config = _config(devices={"translation": [0], "tts": [1]})
        source = _Source(
            _Access("missing", _Live(snapshot_id=None, sample_count=0, sealed=False))
        )
        resolution, _ = _resolve(
            config,
            source,
            environment={"CUDA_VISIBLE_DEVICES": "0,1,2"},
        )

        callbacks = train_script._lifecycle_callbacks(
            resolution.config,
            resolution,
        )

        self.assertEqual(
            [type(callback) for callback in callbacks],
            [ManagedServiceCallback, LiveS2STCursor],
        )
        managed = callbacks[0]
        assert isinstance(managed, ManagedServiceCallback)
        self.assertEqual(managed.label, "S2ST generation")

    def test_real_workspace_s2st_contract_resolves_without_training_injections(self) -> None:
        with TemporaryDirectory() as directory:
            config = _config(
                devices={"translation": [0], "tts": [1]},
                options={"root": directory},
            )
            config = replace(
                config,
                datamodule=replace(
                    config.datamodule,
                    source=replace(
                        config.datamodule.source,
                        factory="zhuyin.datasets.s2st:source",
                    ),
                ),
            )
            resolution = resolve_workspace_source(
                config,
                environment={"CUDA_VISIBLE_DEVICES": "0,1,2"},
            )

        assert resolution.plan is not None
        self.assertIs(resolution.plan.route, SourceRoute.GENERATE)
        self.assertEqual(
            resolution.plan.devices.factories,
            {"translation": ("0",), "tts": ("1",)},
        )
        self.assertIsNotNone(resolution.service)
        self.assertIsNotNone(resolution.live_dataset)
        assert resolution.live_dataset is not None
        resolution.live_dataset.close()

    def test_explicit_legacy_streaming_is_still_available_for_one_cycle(self) -> None:
        base = _config()
        config = replace(
            base,
            datamodule=replace(
                base.datamodule,
                encode_missing_codes=False,
                source=replace(base.datamodule.source, factory=None),
                streaming=StreamingConfig(
                    enabled=True,
                    root="/tmp/workspace-source",
                    stream_id="legacy-stream",
                    expected_samples=8,
                ),
            ),
        )
        source = _Source(
            _Access("missing", _Live(snapshot_id=None, sample_count=0, sealed=False))
        )

        with self.assertWarns(DeprecationWarning):
            resolution, factory = _resolve(
                config,
                source,
                environment={"CUDA_VISIBLE_DEVICES": "GPU-only"},
            )

        self.assertIs(resolution.plan.route, SourceRoute.LEGACY)
        factory.assert_not_called()


def _config(
    *,
    mode: str = "auto",
    devices: dict[str, list[int]] | None = None,
    options: dict[str, object] | None = None,
) -> Any:
    with patch.dict(os.environ, {"DYNAMIC_HOME": "/tmp/dynamic"}):
        base = _train("experiment=train/streaming_s2st")
    return replace(
        base,
        devices={} if devices is None else devices,
        datamodule=replace(
            base.datamodule,
            source=replace(
                base.datamodule.source,
                factory=_FACTORY,
                mode=mode,
                options={} if options is None else options,
            ),
        ),
    )


def _resolve(
    config: Any,
    source: _Source,
    *,
    environment: dict[str, str] | None = None,
) -> tuple[Any, Mock]:
    factory = Mock(return_value=source)
    module = SimpleNamespace(source=factory)
    with patch(
        "speech_to_speech.training.source.importlib.import_module",
        return_value=module,
    ):
        resolution = resolve_workspace_source(config, environment=environment)
    return resolution, factory


if __name__ == "__main__":
    unittest.main()
