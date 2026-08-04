from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from anytrain.lightning import Materialization, MaterializationCallback
from lightning import Trainer


class _DataModule(Protocol):
    @property
    def materialization_enabled(self) -> bool: ...

    @property
    def has_pending_assets(self) -> bool: ...

    def start_asset_materialization(self, *, owner: bool) -> None: ...

    def finish_asset_materialization(self, *, owner: bool) -> None: ...

    def refresh_materialized_assets(self) -> None: ...

    def close_asset_materialization(self) -> None: ...


@dataclass(frozen=True)
class _AssetController:
    datamodule: _DataModule

    @property
    def pending(self) -> bool:
        return self.datamodule.has_pending_assets

    def start(self, *, owner: bool) -> None:
        self.datamodule.start_asset_materialization(owner=owner)

    def finish(self, *, owner: bool) -> None:
        self.datamodule.finish_asset_materialization(owner=owner)

    def refresh(self) -> None:
        self.datamodule.refresh_materialized_assets()

    def close(self) -> None:
        self.datamodule.close_asset_materialization()


def _asset_materialization(trainer: Trainer) -> Materialization | None:
    datamodule = _datamodule(trainer)
    if not datamodule.materialization_enabled:
        return None
    return _AssetController(datamodule)


class AssetMaterialization(MaterializationCallback):
    """Adapt the workspace asset controller to anytrain's generic lifecycle."""

    def __init__(self) -> None:
        super().__init__(
            _asset_materialization,
            barrier_name="asset_materialization_finished",
            label="asset materialization",
        )


def _datamodule(trainer: Trainer) -> _DataModule:
    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is None:
        raise RuntimeError("asset materialization requires Trainer.fit(..., datamodule=...).")
    return cast(_DataModule, cast(object, datamodule))


__all__ = ["AssetMaterialization"]
