from __future__ import annotations

from typing import Protocol, cast

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class _DataModule(Protocol):
    @property
    def materialization_enabled(self) -> bool: ...

    @property
    def has_pending_assets(self) -> bool: ...

    def start_asset_materialization(self, *, owner: bool) -> None: ...

    def finish_asset_materialization(self, *, owner: bool) -> None: ...

    def refresh_materialized_assets(self) -> None: ...


class AssetMaterialization(Callback):
    """Finalize background assets at an epoch boundary and reload them on all ranks."""

    def on_train_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        datamodule = _datamodule(trainer)
        if not datamodule.materialization_enabled:
            return
        if not datamodule.has_pending_assets:
            return
        start_error: Exception | None = None
        try:
            datamodule.start_asset_materialization(owner=bool(trainer.is_global_zero))
        except Exception as error:
            start_error = error
        start_failure = _sync_rank_errors(trainer, start_error)
        if start_failure is not None:
            rank, error_type, message = start_failure
            failure = RuntimeError(
                f"asset materialization start failed on rank {rank}: {error_type}: {message}"
            )
            if start_error is not None and rank == _global_rank(trainer):
                raise failure from start_error
            raise failure

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        del pl_module
        datamodule = _datamodule(trainer)
        if not datamodule.materialization_enabled:
            return
        if not datamodule.has_pending_assets:
            return

        owner_error: Exception | None = None
        if trainer.is_global_zero:
            try:
                datamodule.finish_asset_materialization(owner=True)
            except Exception as error:
                owner_error = error

        payload = _broadcast_error(trainer, owner_error)
        if payload is not None:
            error_type, message = payload
            failure = RuntimeError(
                f"asset materialization failed on the global owner: {error_type}: {message}"
            )
            if owner_error is not None:
                raise failure from owner_error
            raise failure

        trainer.strategy.barrier("asset_materialization_finished")
        refresh_error: Exception | None = None
        try:
            datamodule.refresh_materialized_assets()
        except Exception as error:
            refresh_error = error
        refresh_failure = _sync_rank_errors(trainer, refresh_error)
        if refresh_failure is not None:
            rank, error_type, message = refresh_failure
            failure = RuntimeError(f"asset refresh failed on rank {rank}: {error_type}: {message}")
            if refresh_error is not None and rank == _global_rank(trainer):
                raise failure from refresh_error
            raise failure


def _datamodule(trainer: Trainer) -> _DataModule:
    datamodule = getattr(trainer, "datamodule", None)
    if datamodule is None:
        raise RuntimeError("asset materialization requires Trainer.fit(..., datamodule=...).")
    return cast(_DataModule, cast(object, datamodule))


def _broadcast_error(
    trainer: Trainer,
    error: Exception | None,
) -> tuple[str, str] | None:
    payload = None if error is None else (type(error).__name__, str(error))
    received = trainer.strategy.broadcast(payload, src=0)
    if received is None:
        return None
    if (
        not isinstance(received, tuple)
        or len(received) != 2
        or any(not isinstance(value, str) for value in received)
    ):
        raise TypeError("materialization error broadcast returned an invalid payload.")
    return cast(tuple[str, str], received)


def _sync_rank_errors(
    trainer: Trainer,
    error: Exception | None,
) -> tuple[int, str, str] | None:
    rank = _global_rank(trainer)
    world_size = int(getattr(trainer, "world_size", 1))
    if world_size < 1:
        raise ValueError("trainer world_size must be positive.")
    if rank >= world_size:
        raise ValueError("trainer global_rank must be smaller than world_size.")
    payload = None if error is None else (type(error).__name__, str(error))
    if world_size == 1:
        if payload is None:
            return None
        return (rank, *payload)
    failures: list[tuple[int, str, str]] = []
    for source in range(world_size):
        value = payload if source == rank else None
        received = trainer.strategy.broadcast(value, src=source)
        if received is None:
            continue
        if (
            not isinstance(received, tuple)
            or len(received) != 2
            or any(not isinstance(entry, str) for entry in received)
        ):
            raise TypeError("materialization rank error broadcast was invalid.")
        error_type, message = cast(tuple[str, str], received)
        failures.append((source, error_type, message))
    if not failures:
        return None
    return min(failures, key=lambda failure: failure[0])


def _global_rank(trainer: Trainer) -> int:
    value = getattr(trainer, "global_rank", None)
    if value is None:
        return 0 if trainer.is_global_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("trainer global_rank must be a non-negative integer.")
    return value


__all__ = ["AssetMaterialization"]
