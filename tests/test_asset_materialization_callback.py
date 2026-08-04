from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

from _config_helpers import _train
from scripts import train as train_script
from speech_to_speech.callback import AssetMaterialization
from speech_to_speech.training.composition import create_trainer


_BROADCAST_INPUT = object()


class _DataModule:
    materialization_enabled = True
    has_pending_assets = True

    def __init__(
        self,
        events: list[tuple[object, ...]],
        *,
        start_error: Exception | None = None,
        refresh_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.start_error = start_error
        self.refresh_error = refresh_error

    def start_asset_materialization(self, *, owner: bool) -> None:
        self.events.append(("start", owner))
        if self.start_error is not None:
            raise self.start_error

    def finish_asset_materialization(self, *, owner: bool) -> None:
        self.events.append(("finish", owner))

    def refresh_materialized_assets(self) -> None:
        self.events.append(("refresh",))
        if self.refresh_error is not None:
            raise self.refresh_error

    def close_asset_materialization(self) -> None:
        self.events.append(("close",))


class _Strategy:
    def __init__(
        self,
        events: list[tuple[object, ...]],
        *,
        broadcast_result: object = _BROADCAST_INPUT,
        broadcast_results: list[object] | None = None,
    ) -> None:
        self.events = events
        self.broadcast_result = broadcast_result
        self.broadcast_results = broadcast_results

    def broadcast(self, value: object, src: int = 0) -> object:
        self.events.append(("broadcast", value, src))
        if self.broadcast_results is not None:
            return self.broadcast_results.pop(0)
        if self.broadcast_result is _BROADCAST_INPUT:
            return value
        return self.broadcast_result

    def barrier(self, name: str | None = None) -> None:
        self.events.append(("barrier", name))


@patch.dict(
    "os.environ",
    {
        "DYNAMIC_HOME": "/tmp/dynamic",
        "SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer",
    },
)
class AssetMaterializationCallbackTest(unittest.TestCase):
    def test_global_zero_finishes_then_barriers_and_refreshes(self) -> None:
        events: list[tuple[object, ...]] = []
        datamodule = _DataModule(events)
        trainer = SimpleNamespace(
            datamodule=datamodule,
            is_global_zero=True,
            strategy=_Strategy(events),
        )
        callback = AssetMaterialization()

        callback.on_train_start(cast(Any, trainer), cast(Any, object()))
        callback.on_train_epoch_end(cast(Any, trainer), cast(Any, object()))

        self.assertEqual(
            events,
            [
                ("start", True),
                ("finish", True),
                ("broadcast", None, 0),
                ("barrier", "asset_materialization_finished"),
                ("finish", False),
                ("refresh",),
            ],
        )

    def test_non_owner_raises_broadcast_owner_error_without_refresh(self) -> None:
        events: list[tuple[object, ...]] = []
        trainer = SimpleNamespace(
            datamodule=_DataModule(events),
            is_global_zero=False,
            global_rank=1,
            world_size=2,
            strategy=_Strategy(
                events,
                broadcast_results=[
                    None,
                    None,
                    True,
                    True,
                    None,
                    None,
                    ("ValueError", "provider failed"),
                ],
            ),
        )
        callback = AssetMaterialization()

        callback.on_train_start(cast(Any, trainer), cast(Any, object()))

        with self.assertRaisesRegex(
            RuntimeError,
            "global owner: ValueError: provider failed",
        ):
            callback.on_train_epoch_end(
                cast(Any, trainer),
                cast(Any, object()),
            )

        self.assertIn(("start", False), events)
        self.assertNotIn(("refresh",), events)

    def test_start_error_is_synchronized_from_global_owner(self) -> None:
        events: list[tuple[object, ...]] = []
        trainer = SimpleNamespace(
            datamodule=_DataModule(events),
            is_global_zero=False,
            global_rank=1,
            world_size=2,
            strategy=_Strategy(
                events,
                broadcast_results=[
                    None,
                    None,
                    True,
                    True,
                    ("RuntimeError", "spawn failed"),
                    None,
                ],
            ),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "asset materialization start failed on rank 0: RuntimeError: spawn failed",
        ):
            AssetMaterialization().on_train_start(
                cast(Any, trainer),
                cast(Any, object()),
            )

        self.assertIn(("start", False), events)

    def test_global_owner_start_error_preserves_local_cause(self) -> None:
        events: list[tuple[object, ...]] = []
        start_error = OSError("spawn failed")
        trainer = SimpleNamespace(
            datamodule=_DataModule(events, start_error=start_error),
            is_global_zero=True,
            global_rank=0,
            world_size=2,
            strategy=_Strategy(
                events,
                broadcast_results=[
                    None,
                    None,
                    True,
                    True,
                    ("OSError", "spawn failed"),
                    None,
                ],
            ),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "asset materialization start failed on rank 0: OSError: spawn failed",
        ) as caught:
            AssetMaterialization().on_train_start(
                cast(Any, trainer),
                cast(Any, object()),
            )

        self.assertIs(caught.exception.__cause__, start_error)
        self.assertIn(("start", True), events)

    def test_refresh_error_is_synchronized_from_non_owner_rank(self) -> None:
        events: list[tuple[object, ...]] = []
        datamodule = _DataModule(
            events,
            refresh_error=OSError("store is not visible"),
        )
        trainer = SimpleNamespace(
            datamodule=datamodule,
            is_global_zero=False,
            global_rank=1,
            world_size=2,
            strategy=_Strategy(
                events,
                broadcast_results=[
                    None,
                    None,
                    True,
                    True,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    ("OSError", "store is not visible"),
                ],
            ),
        )
        callback = AssetMaterialization()

        callback.on_train_start(cast(Any, trainer), cast(Any, object()))

        with self.assertRaisesRegex(
            RuntimeError,
            "asset materialization refresh failed on rank 1: OSError: store is not visible",
        ):
            callback.on_train_epoch_end(
                cast(Any, trainer),
                cast(Any, object()),
            )

        self.assertIn(("start", False), events)
        self.assertIn(("barrier", "asset_materialization_finished"), events)
        self.assertIn(("refresh",), events)

    def test_enabled_materialization_forwards_reload_to_lightning(self) -> None:
        config = _materialization_config()
        with (
            patch("scripts.train.create_trainer") as entry,
            patch("scripts.train.build_logger"),
        ):
            train_script.build_trainer(config, Path("/tmp/output"), [])

        self.assertEqual(
            entry.call_args.kwargs["reload_dataloaders_every_n_epochs"],
            1,
        )

        factory = Mock(return_value=object())
        built = create_trainer(
            config,
            Path("/tmp/output"),
            [],
            logger=None,
            factory=factory,
            reload_dataloaders_every_n_epochs=1,
        )

        self.assertIs(built, factory.return_value)
        self.assertEqual(
            factory.call_args.kwargs["reload_dataloaders_every_n_epochs"],
            1,
        )

    def test_training_callbacks_install_materialization_only_when_enabled(self) -> None:
        enabled = _materialization_config(
            "trainer.enable_checkpointing=false",
            "callbacks.text_retention.enabled=false",
        )
        disabled = _train(
            "trainer.enable_checkpointing=false",
            "callbacks.text_retention.enabled=false",
        )
        schedule_runtime = Mock()
        schedule_runtime.callbacks.return_value = []

        enabled_callbacks = train_script.training_callbacks(
            enabled,
            Path("/tmp/output"),
            Mock(),
            schedule_runtime=schedule_runtime,
        )
        disabled_callbacks = train_script.training_callbacks(
            disabled,
            Path("/tmp/output"),
            Mock(),
            schedule_runtime=schedule_runtime,
        )

        self.assertEqual(
            sum(isinstance(callback, AssetMaterialization) for callback in enabled_callbacks),
            1,
        )
        self.assertFalse(
            any(isinstance(callback, AssetMaterialization) for callback in disabled_callbacks)
        )


def _materialization_config(*overrides: str):
    return _train(
        "+datamodule.encode_missing_codes=true",
        "datamodule.materialization.enabled=true",
        "datamodule.materialization.output_root=/tmp/materialized-assets",
        "datamodule.materialization.device=cpu",
        "datamodule.materialization.provider_id=test-provider-v1",
        *overrides,
    )


if __name__ == "__main__":
    unittest.main()
