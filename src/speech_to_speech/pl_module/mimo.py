"""Standalone Lightning wrapper for aligned dual-stream pretraining."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, Generic, Protocol, TypeVar, TypedDict, Union, runtime_checkable

import torch
from anytrain import observation
from anytrain.lightning.schedule import ScheduleRuntime
from anytrain.optim.llm import create_optimizer
from lightning.pytorch import LightningModule
from torch import Tensor, nn

from ..mimo import MimoBatch
from ..loss.mimo import MimoObjective
from ..model.checkpoint_contract import (
    ModelCheckpointContract,
    state_dict_contract,
    validate_checkpoint_contract,
)
from ..model.mimo import MimoModel as AlignedMimoModel
from ..model.mimo import MimoModelConfig
from .optim import Config as OptimConfig
from ..runtime.backbone.mimo import (
    DualStreamHiddenStates,
    DualStreamLogits,
    DualStreamOutput,
)


DualHidden = Union[DualStreamHiddenStates, DualStreamOutput]


@runtime_checkable
class MimoModel(Protocol):
    """Model surface required by :class:`MimoModule`."""

    def dual_hidden_states(self, batch: MimoBatch) -> DualHidden: ...

    def dual_logits(
        self,
        hidden_states: DualHidden,
    ) -> DualStreamLogits | tuple[Tensor, Tensor]: ...


class MimoStepOutput(TypedDict):
    loss: Tensor
    mimo: Tensor


ModelT = TypeVar("ModelT", bound=nn.Module)
_MIMO_SCHEMA_KEY = "speech_to_speech_mimo_schema"
_MIMO_SCHEMA = "mimo-v1"
_MIMO_CONTRACT_KEY = "speech_to_speech_mimo_contract"


class MimoModule(LightningModule, Generic[ModelT]):
    """Train a dual-stream model without coupling it to the single-stream model.

    The wrapped model must be an ``nn.Module`` so Lightning owns its parameters,
    and it must satisfy :class:`MimoModel` for the objective forward.  The
    objective performs route-specific token normalization; this wrapper reduces
    it to the scalar loss consumed by automatic optimization.
    """

    def __init__(
        self,
        *,
        model: ModelT,
        objective: MimoObjective | None = None,
        optim: OptimConfig | None = None,
        schedule_runtime: ScheduleRuntime | None = None,
        checkpoint_metadata: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("MimoModule model must be a torch.nn.Module.")
        if not isinstance(model, MimoModel):
            raise TypeError("MimoModule model must expose dual_hidden_states and dual_logits.")
        if objective is not None and not isinstance(objective, MimoObjective):
            raise TypeError("MimoModule objective must be a MimoObjective or None.")
        if optim is not None and not isinstance(optim, OptimConfig):
            raise TypeError("MimoModule optim must be an OptimConfig or None.")
        if checkpoint_metadata is not None and not isinstance(checkpoint_metadata, Mapping):
            raise TypeError("MimoModule checkpoint_metadata must be a mapping or None.")

        self.model = model
        self.objective = MimoObjective() if objective is None else objective
        self.optim = OptimConfig() if optim is None else optim
        self.schedule_runtime = schedule_runtime
        self.checkpoint_contract = _checkpoint_contract(
            model,
            self.objective,
            {} if checkpoint_metadata is None else checkpoint_metadata,
        )

    def training_step(
        self,
        batch: MimoBatch,
        batch_idx: int = 0,
    ) -> MimoStepOutput:
        del batch_idx
        output = self._step(batch)
        self.log(
            "loss",
            output["loss"],
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            batch_size=batch.batch_size,
        )
        return output

    def validation_step(
        self,
        batch: MimoBatch,
        batch_idx: int = 0,
    ) -> MimoStepOutput:
        del batch_idx
        output = self._step(batch)
        self.log(
            "val/loss",
            output["loss"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            batch_size=batch.batch_size,
        )
        return output

    def on_train_epoch_start(self) -> None:
        """Advance deterministic task datasets with the Lightning epoch.

        ``MimoTaskDataset`` deliberately keeps its sampling state outside the
        worker RNG.  Lightning may receive either a datamodule or a direct
        dataloader, so discover the dataset from the trainer rather than
        relying on one particular entry path.
        """

        loader = getattr(self.trainer, "train_dataloader", None)
        _set_dataset_epoch(loader, self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        loader = getattr(self.trainer, "val_dataloaders", None)
        _set_dataset_epoch(loader, self.current_epoch)

    def _step(self, batch: MimoBatch) -> MimoStepOutput:
        if not isinstance(batch, MimoBatch):
            raise TypeError("MimoModule requires MimoBatch inputs.")
        # MimoBatch validates masks on CPU before transfer; trusted loss avoids
        # repeating device-to-host predicates in every training step.
        with observation.context(_observation_prefix(batch)):
            loss = self.objective.from_batch(
                batch,
                self.model,
                validate=False,
                distributed=True,
            )
        return {"loss": loss, "mimo": loss}

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        if isinstance(batch, MimoBatch):
            return batch.to(device, non_blocking=True)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def configure_optimizers(self):
        optimizer = create_optimizer(
            self.model,
            preset="pretrain",
            optimizer=self.optim.name,
            lr=self.optim.learning_rate,
            weight_decay=self.optim.weight_decay,
        )
        if self.schedule_runtime is None:
            return optimizer
        return self.schedule_runtime.configure_optimizers(optimizer)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint[_MIMO_SCHEMA_KEY] = _MIMO_SCHEMA
        checkpoint[_MIMO_CONTRACT_KEY] = self.checkpoint_contract.checkpoint_payload()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        actual_schema = checkpoint.get(_MIMO_SCHEMA_KEY)
        if actual_schema != _MIMO_SCHEMA:
            raise ValueError(
                "checkpoint MIMO schema is incompatible: "
                f"expected {_MIMO_SCHEMA!r}, got {actual_schema!r}."
            )
        if _MIMO_CONTRACT_KEY not in checkpoint:
            raise ValueError("checkpoint is missing the MIMO model contract.")
        validate_checkpoint_contract(
            checkpoint[_MIMO_CONTRACT_KEY],
            self.checkpoint_contract,
        )


MimoPretrainingModule = MimoModule


def _observation_prefix(batch: MimoBatch) -> str:
    tasks = set(batch.task_ids or ())
    task = next(iter(tasks)) if len(tasks) == 1 else "mixed"
    return f"task/{task}"


def _set_dataset_epoch(value: object, epoch: int) -> None:
    """Set an epoch on a loader/dataset, including Lightning wrappers."""

    if value is None:
        return
    if isinstance(value, Mapping):
        for loader in value.values():
            _set_dataset_epoch(loader, epoch)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for loader in value:
            _set_dataset_epoch(loader, epoch)
        return
    setter = getattr(value, "set_epoch", None)
    if callable(setter):
        setter(epoch)
        return
    dataset = getattr(value, "dataset", None)
    if dataset is not None and dataset is not value:
        _set_dataset_epoch(dataset, epoch)
    loaders = getattr(value, "loaders", None)
    if loaders is not None and loaders is not value:
        if isinstance(loaders, dict):
            for loader in loaders.values():
                _set_dataset_epoch(loader, epoch)
        else:
            _set_dataset_epoch(loaders, epoch)


def _checkpoint_contract(
    model: nn.Module,
    objective: MimoObjective,
    metadata: Mapping[str, object],
) -> ModelCheckpointContract:
    return ModelCheckpointContract.from_components(
        {
            "mimo_schema": _MIMO_SCHEMA,
            "model": _model_contract(model),
            "objective": {
                "text_weight": objective.text_weight,
                "audio_weight": objective.audio_weight,
                "ignore_index": objective.ignore_index,
            },
            "metadata": metadata,
        }
    )


def _model_contract(model: nn.Module) -> dict[str, object]:
    config = getattr(model, "config", None)
    return {
        "class": _class_name(model),
        "state_dict": state_dict_contract(model),
        "embeddings": {
            "text": _embedding_contract(getattr(model, "text_embedding", None)),
            "audio": _embedding_contract(getattr(model, "audio_embedding", None)),
        },
        "readouts": {
            "text": _readout_path(getattr(model, "text_readout", None)),
            "audio": _readout_path(getattr(model, "audio_readout", None)),
        },
        "heads": {
            "text": _head_contract(model, "text"),
            "audio": _head_contract(model, "audio"),
        },
        "audio_feature_projection": _module_contract(
            getattr(model, "audio_feature_projection", None)
        ),
        "config": asdict(config) if isinstance(config, MimoModelConfig) else None,
    }


def _embedding_contract(value: object) -> dict[str, object] | None:
    if not isinstance(value, nn.Embedding):
        return None
    return {
        "class": _class_name(value),
        "num_embeddings": value.num_embeddings,
        "embedding_dim": value.embedding_dim,
    }


def _readout_path(value: object) -> str | None:
    path = getattr(value, "path", None)
    return path if isinstance(path, str) else None


def _head_contract(model: nn.Module, route: str) -> dict[str, object] | None:
    value = getattr(model, f"{route}_head", None)
    if value is None and isinstance(model, AlignedMimoModel):
        return {"class": "embedding_tied"}
    return _module_contract(value)


def _module_contract(value: object) -> dict[str, object] | None:
    if not isinstance(value, nn.Module):
        return None
    result: dict[str, object] = {"class": _class_name(value)}
    for name in (
        "in_features",
        "out_features",
        "num_embeddings",
        "embedding_dim",
        "vocab_size",
    ):
        field = getattr(value, name, None)
        if isinstance(field, int) and not isinstance(field, bool):
            result[name] = field
    bias = getattr(value, "bias", None)
    if bias is not None:
        result["bias"] = isinstance(bias, Tensor)
    return result


def _class_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


__all__ = [
    "DualHidden",
    "MimoModel",
    "MimoModule",
    "MimoPretrainingModule",
    "MimoStepOutput",
]
