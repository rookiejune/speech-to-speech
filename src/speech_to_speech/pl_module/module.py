from __future__ import annotations

from dataclasses import asdict
from typing import TypedDict

import torch
from anytrain.optim import create_llm_lightning_optimizers
from lightning.pytorch import LightningModule
from torch import Tensor, device as TorchDevice
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..config import TrainConfig
from ..model.acoustic import AcousticFlowLossStats, acoustic_features_from_batch_side
from ..model.orchestrator import Orchestrator
from ..types import CausalLMBatch, IGNORE_INDEX, LongCatBatchSide, TaskFamily


ACOUSTIC_T_BINS = (
    ("0_025", 0.0, 0.25),
    ("025_050", 0.25, 0.5),
    ("050_075", 0.5, 0.75),
    ("075_100", 0.75, 1.0),
)


class _LRSchedulerConfig(TypedDict):
    scheduler: LRScheduler
    interval: str


class _LightningOptimizerConfig(TypedDict):
    optimizer: Optimizer
    lr_scheduler: _LRSchedulerConfig


class SpeechToSpeechModule(LightningModule):
    """Adapts the speech-to-speech orchestrator to Lightning training."""

    def __init__(
        self,
        model: Orchestrator,
        train: TrainConfig | None = None,
        *,
        bpe: object | None = None,
        acoustic_feature_extractor: object | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.train_config = train or TrainConfig()
        self.bpe = bpe
        self.__dict__["_acoustic_feature_extractor"] = acoustic_feature_extractor
        self.save_hyperparameters({"train": asdict(self.train_config)})

    @property
    def acoustic_feature_extractor(self) -> object | None:
        return self.__dict__.get("_acoustic_feature_extractor")

    def forward(self, batch: CausalLMBatch) -> CausalLMOutputWithPast:
        return self.model(batch)

    def training_step(self, batch: CausalLMBatch, batch_idx: int) -> Tensor:
        del batch_idx
        output = self.model(batch)
        row_loss = self._semantic_row_loss(batch, output)
        token_counts = _loss_token_counts(batch, dtype=row_loss.dtype)
        loss = self._loss(batch, row_loss, token_counts, stage=None)
        self.log(
            "loss",
            loss,
            batch_size=batch.input_ids.size(0),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self._log_semantic_accuracy(batch, output, token_counts, stage=None)
        self._log_task_losses(batch, row_loss, token_counts, stage=None)
        self._log_family_group_losses(batch, row_loss, token_counts, stage=None)
        self.log(
            "supervised_tokens",
            token_counts.sum(),
            batch_size=batch.input_ids.size(0),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        self._log_acoustic_frame_count(batch, stage=None)
        return loss

    def validation_step(self, batch: CausalLMBatch, batch_idx: int) -> Tensor:
        del batch_idx
        output = self.model(batch)
        row_loss = self._semantic_row_loss(batch, output)
        token_counts = _loss_token_counts(batch, dtype=row_loss.dtype)
        loss = self._loss(batch, row_loss, token_counts, stage="val")
        self.log(
            "val/loss",
            loss,
            batch_size=batch.input_ids.size(0),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self._log_semantic_accuracy(batch, output, token_counts, stage="val")
        self._log_task_losses(batch, row_loss, token_counts, stage="val")
        self._log_family_group_losses(batch, row_loss, token_counts, stage="val")
        self._log_acoustic_frame_count(batch, stage="val")
        return loss

    def transfer_batch_to_device(
        self,
        batch: CausalLMBatch,
        device: TorchDevice,
        dataloader_idx: int,
    ) -> CausalLMBatch:
        del dataloader_idx
        logits_to_keep = batch.logits_to_keep
        if isinstance(logits_to_keep, Tensor):
            logits_to_keep = logits_to_keep.to(device=device)
        task_family = batch.task_family
        if task_family is not None:
            task_family = task_family.to(device=device)
        return CausalLMBatch(
            input_ids=batch.input_ids.to(device=device),
            attention_mask=batch.attention_mask.to(device=device),
            labels=batch.labels.to(device=device),
            logits_to_keep=logits_to_keep,
            source_audio=_move_longcat_side(batch.source_audio, device),
            target_audio=_move_longcat_side(batch.target_audio, device),
            task_family=task_family,
        )

    def configure_optimizers(self) -> _LightningOptimizerConfig:
        train = self.train_config
        return create_llm_lightning_optimizers(
            self.model,
            preset=train.optimizer_preset,
            optimizer=train.optimizer,
            lr=_adamw_learning_rate(train),
            weight_decay=train.weight_decay,
            muon_lr=_muon_learning_rate(train),
            schedule=train.schedule,
            warmup_steps=train.warmup_steps,
            total_steps=_scheduler_total_steps(train),
            stable_steps=train.stable_steps,
            decay_steps=train.decay_steps,
            min_lr_ratio=train.min_lr_ratio,
        )

    def _loss(
        self,
        batch: CausalLMBatch,
        row_loss: Tensor | None = None,
        token_counts: Tensor | None = None,
        *,
        stage: str | None = None,
    ) -> Tensor:
        if row_loss is None:
            row_loss = self._semantic_row_loss(batch)
        if token_counts is None:
            token_counts = _loss_token_counts(batch, dtype=row_loss.dtype)
        loss = _weighted_mean(row_loss, token_counts)
        acoustic_weight = self.train_config.acoustic_loss_weight
        if acoustic_weight <= 0.0:
            return loss
        acoustic = self._acoustic_loss(batch)
        acoustic_loss = acoustic.loss if isinstance(acoustic, AcousticFlowLossStats) else acoustic
        self.log(
            _log_name("loss/acoustic", stage=stage),
            acoustic_loss,
            batch_size=batch.input_ids.size(0),
            on_step=stage is None,
            on_epoch=True,
            sync_dist=True,
        )
        if isinstance(acoustic, AcousticFlowLossStats):
            self._log_acoustic_t_bin_losses(acoustic, stage=stage)
        return loss + acoustic_weight * acoustic_loss

    def _semantic_row_loss(
        self,
        batch: CausalLMBatch,
        output: CausalLMOutputWithPast | None = None,
    ) -> Tensor:
        if output is None:
            output = self.model(batch)
        loss = output.loss
        if loss is None:
            raise RuntimeError("model output must include loss.")
        if loss.dim() == 0:
            return loss.unsqueeze(0).expand(batch.input_ids.size(0))
        if loss.dim() != 1 or loss.size(0) != batch.input_ids.size(0):
            raise RuntimeError("model loss must be scalar or one value per batch row.")
        return loss

    def _log_semantic_accuracy(
        self,
        batch: CausalLMBatch,
        output: CausalLMOutputWithPast,
        token_counts: Tensor,
        *,
        stage: str | None,
    ) -> None:
        self.log(
            _log_name("accuracy", stage=stage),
            self._semantic_accuracy(batch, output),
            batch_size=int(token_counts.sum().detach().item()),
            on_step=stage is None,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def _semantic_accuracy(
        self,
        batch: CausalLMBatch,
        output: CausalLMOutputWithPast,
    ) -> Tensor:
        semantic_accuracy = getattr(self.model, "semantic_accuracy", None)
        if not callable(semantic_accuracy):
            raise TypeError("model must provide semantic_accuracy().")
        accuracy = semantic_accuracy(batch, output)
        if not isinstance(accuracy, Tensor):
            raise TypeError("model semantic_accuracy() must return a Tensor.")
        return accuracy

    def _log_task_losses(
        self,
        batch: CausalLMBatch,
        row_loss: Tensor,
        token_counts: Tensor,
        *,
        stage: str | None,
    ) -> None:
        if batch.task_family is None:
            return
        row_loss = row_loss.detach()
        for family in TaskFamily:
            mask = batch.task_family.eq(family.id)
            if not bool(mask.any()):
                continue
            tokens = token_counts[mask].sum()
            self.log(
                _log_name(f"loss/{family.value}", stage=stage),
                _weighted_mean(row_loss[mask], token_counts[mask]),
                batch_size=int(mask.sum().item()),
                on_step=stage is None,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                _log_name(f"tokens/{family.value}", stage=stage),
                tokens,
                batch_size=int(mask.sum().item()),
                on_step=stage is None,
                on_epoch=False,
                sync_dist=True,
            )

    def _log_family_group_losses(
        self,
        batch: CausalLMBatch,
        row_loss: Tensor,
        token_counts: Tensor,
        *,
        stage: str | None,
    ) -> None:
        if batch.task_family is None:
            return
        row_loss = row_loss.detach()
        groups = {
            "semantic_ar": (
                TaskFamily.SOURCE_AR,
                TaskFamily.TARGET_AR,
            ),
            "translation": (
                TaskFamily.SOURCE_TO_TARGET,
                TaskFamily.TARGET_TO_SOURCE,
            ),
        }
        for name, families in groups.items():
            mask = torch.zeros_like(batch.task_family, dtype=torch.bool)
            for family in families:
                mask |= batch.task_family.eq(family.id)
            if not bool(mask.any()):
                continue
            self.log(
                _log_name(f"loss/{name}", stage=stage),
                _weighted_mean(row_loss[mask], token_counts[mask]),
                batch_size=int(mask.sum().item()),
                on_step=stage is None,
                on_epoch=True,
                sync_dist=True,
            )

    def _log_acoustic_frame_count(self, batch: CausalLMBatch, *, stage: str | None) -> None:
        if batch.target_audio is None:
            return
        frames = batch.target_audio.acoustic_mask.sum()
        self.log(
            _log_name("acoustic_frames", stage=stage),
            frames,
            batch_size=batch.input_ids.size(0),
            on_step=stage is None,
            on_epoch=stage is not None,
            sync_dist=True,
        )

    def _acoustic_loss(self, batch: CausalLMBatch) -> Tensor | AcousticFlowLossStats:
        if self.bpe is None:
            raise RuntimeError("acoustic loss requires a LongCat BPE tokenizer.")
        if self.acoustic_feature_extractor is None:
            raise RuntimeError("acoustic loss requires an acoustic feature extractor.")
        if batch.target_audio is None:
            raise RuntimeError("acoustic loss requires target_audio in the batch.")
        feature_extractor = _feature_extractor_to_device(
            self.acoustic_feature_extractor,
            batch.target_audio.acoustic_ids.device,
        )
        target_features, target_mask = acoustic_features_from_batch_side(
            batch.target_audio,
            feature_extractor=feature_extractor,
        )
        acoustic_flow_loss_stats = getattr(self.model, "acoustic_flow_loss_stats", None)
        if callable(acoustic_flow_loss_stats):
            return acoustic_flow_loss_stats(
                batch,
                self.bpe,
                target_features,
                target_mask=target_mask,
                source_feature_extractor=feature_extractor,
            )
        return self.model.acoustic_flow_loss(
            batch,
            self.bpe,
            target_features,
            target_mask=target_mask,
            source_feature_extractor=feature_extractor,
        )

    def _log_acoustic_t_bin_losses(
        self,
        stats: AcousticFlowLossStats,
        *,
        stage: str | None,
    ) -> None:
        timesteps = stats.timesteps.detach()
        row_loss = stats.row_loss.detach()
        row_weight = stats.row_weight.detach()
        for name, start, end in ACOUSTIC_T_BINS:
            if end >= 1.0:
                mask = timesteps.ge(start) & timesteps.le(end)
            else:
                mask = timesteps.ge(start) & timesteps.lt(end)
            if not bool(mask.any()):
                continue
            weight = row_weight[mask]
            if not bool(weight.gt(0).any()):
                continue
            loss = (row_loss[mask] * weight).sum() / weight.sum()
            batch_size = int(weight.sum().detach().item())
            self.log(
                _log_name(f"loss/acoustic_t/{name}", stage=stage),
                loss,
                batch_size=batch_size,
                on_step=stage is None,
                on_epoch=True,
                sync_dist=True,
            )


def _scheduler_total_steps(train: TrainConfig) -> int | None:
    if train.schedule == "constant":
        return None
    return train.max_steps


def _adamw_learning_rate(train: TrainConfig) -> float:
    if train.adamw_learning_rate is None:
        return train.learning_rate
    return train.adamw_learning_rate


def _muon_learning_rate(train: TrainConfig) -> float | None:
    if train.optimizer != "muon" and train.muon_learning_rate is not None:
        raise ValueError("train.muon_learning_rate requires train.optimizer='muon'.")
    return train.muon_learning_rate


def _loss_token_counts(batch: CausalLMBatch, *, dtype: torch.dtype) -> Tensor:
    positions = _loss_positions(batch)
    counts = torch.zeros(batch.input_ids.size(0), device=batch.labels.device)
    counts.scatter_add_(
        0,
        positions[:, 0],
        torch.ones_like(positions[:, 0], dtype=counts.dtype),
    )
    return counts.to(dtype=dtype)


def _loss_positions(batch: CausalLMBatch) -> Tensor:
    if isinstance(batch.logits_to_keep, Tensor):
        positions = batch.logits_to_keep.to(device=batch.labels.device, dtype=torch.long)
        if positions.dim() != 2 or positions.size(-1) != 2:
            raise ValueError("logits_to_keep tensor must have shape (n, 2).")
        return positions
    if batch.logits_to_keep <= 0:
        raise ValueError("logits_to_keep must be positive.")
    mask = batch.labels.ne(IGNORE_INDEX)
    if not bool(mask.any()):
        raise ValueError("labels must contain at least one supervised token.")
    if batch.logits_to_keep >= mask.size(1):
        return mask.nonzero(as_tuple=False)
    keep = mask.cumsum(dim=1) > (mask.sum(dim=1, keepdim=True) - batch.logits_to_keep).clamp_min(0)
    return (mask & keep).nonzero(as_tuple=False)


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    total = weights.sum()
    if not bool(total.gt(0)):
        raise ValueError("weighted mean requires a positive weight sum.")
    return (values * weights).sum() / total


def _log_name(name: str, *, stage: str | None) -> str:
    if stage is None:
        return name
    return f"{stage}/{name}"


def _feature_extractor_to_device(extractor: object, device: TorchDevice) -> object:
    move = getattr(extractor, "to", None)
    if callable(move):
        moved = move(device)
        if moved is not None:
            extractor = moved
    if hasattr(extractor, "device"):
        setattr(extractor, "device", device)
    return extractor


def _move_longcat_side(
    side: LongCatBatchSide | None,
    device: TorchDevice,
) -> LongCatBatchSide | None:
    if side is None:
        return None
    return LongCatBatchSide(
        semantic_ids=side.semantic_ids.to(device=device),
        semantic_mask=side.semantic_mask.to(device=device),
        acoustic_ids=side.acoustic_ids.to(device=device),
        acoustic_mask=side.acoustic_mask.to(device=device),
    )
