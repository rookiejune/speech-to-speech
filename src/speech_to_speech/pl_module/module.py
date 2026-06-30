from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypedDict

import torch
from anytrain.optim import create_llm_lightning_optimizers
from lightning.pytorch import LightningModule
from torch import Tensor
from torch import device as TorchDevice
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..config import TrainConfig
from ..model.acoustic import AcousticFlowLossStats, acoustic_features_from_batch_side
from ..model.orchestrator import (
    AcousticFlowInputs,
    DiTConditionTensors,
    Orchestrator,
)
from ..types import IGNORE_INDEX, CausalLMBatch, LongCatBatchSide, TaskFamily
from .metrics import (
    ReducedMean,
    log_reduced_mean,
    log_reduced_sum,
    log_reduced_value,
    reduced_weighted_mean,
    scaled_loss,
)

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
        output = self.model(
            batch,
            return_hidden_states=self.train_config.acoustic_loss_weight > 0.0,
        )
        row_loss = self._semantic_row_loss(batch, output)
        token_counts = _loss_token_counts(batch, dtype=row_loss.dtype)
        loss = self._loss(
            batch,
            row_loss,
            token_counts,
            stage=None,
            hidden_states=_last_hidden_state(output),
        )
        self._log_semantic_accuracy(batch, output, token_counts, stage=None)
        self._log_task_losses(batch, row_loss, token_counts, stage=None)
        self._log_family_group_losses(batch, row_loss, token_counts, stage=None)
        log_reduced_sum(
            self,
            "supervised_tokens",
            token_counts.sum(),
            on_step=True,
            on_epoch=False,
        )
        self._log_acoustic_frame_count(batch, stage=None)
        return loss

    def validation_step(self, batch: CausalLMBatch, batch_idx: int) -> Tensor:
        del batch_idx
        output = self.model(
            batch,
            return_hidden_states=self.train_config.acoustic_loss_weight > 0.0,
        )
        row_loss = self._semantic_row_loss(batch, output)
        token_counts = _loss_token_counts(batch, dtype=row_loss.dtype)
        loss = self._loss(
            batch,
            row_loss,
            token_counts,
            stage="val",
            hidden_states=_last_hidden_state(output),
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
        _validate_scheduler(train)
        return create_llm_lightning_optimizers(
            self.model,
            preset=train.optimizer_preset,
            optimizer=train.optimizer,
            lr=_adamw_learning_rate(train),
            weight_decay=train.weight_decay,
            muon_lr=_muon_learning_rate(train),
            schedule=train.schedule,
            warmup_steps=_scheduler_warmup_steps(train),
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
        hidden_states: Tensor | None = None,
    ) -> Tensor:
        if row_loss is None:
            row_loss = self._semantic_row_loss(batch)
        if token_counts is None:
            token_counts = _loss_token_counts(batch, dtype=row_loss.dtype)
        semantic_mean = reduced_weighted_mean(row_loss, token_counts)
        loss = scaled_loss(semantic_mean)
        logged_loss = semantic_mean.value
        acoustic_weight = self.train_config.acoustic_loss_weight
        if acoustic_weight > 0.0:
            acoustic = self._acoustic_loss(
                batch,
                hidden_states=hidden_states,
                stage=stage,
            )
            acoustic_mean = _acoustic_reduced_mean(batch, acoustic)
            acoustic_loss = scaled_loss(acoustic_mean)
            loss = loss + acoustic_weight * acoustic_loss
            logged_loss = logged_loss + acoustic_weight * acoustic_mean.value
            log_reduced_mean(
                self,
                _log_name("loss/acoustic", stage=stage),
                acoustic_mean,
                on_step=stage is None,
                on_epoch=True,
            )
            if isinstance(acoustic, AcousticFlowLossStats):
                self._log_acoustic_t_bin_losses(acoustic, stage=stage)
        log_reduced_value(
            self,
            _log_name("loss", stage=stage),
            logged_loss,
            semantic_mean.weight,
            on_step=stage is None,
            on_epoch=True,
            prog_bar=True,
        )
        if stage is None:
            return loss
        return logged_loss

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
        accuracy = self._semantic_accuracy(batch, output).reshape(())
        mean = reduced_weighted_mean(
            accuracy.detach(),
            token_counts.sum().detach().reshape(()),
        )
        log_reduced_mean(
            self,
            _log_name("accuracy", stage=stage),
            mean,
            on_step=stage is None,
            on_epoch=True,
            prog_bar=True,
        )

    def _semantic_accuracy(
        self,
        batch: CausalLMBatch,
        output: CausalLMOutputWithPast,
    ) -> Tensor:
        accuracy = self.model.semantic_accuracy(batch, output)
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
            mean = reduced_weighted_mean(row_loss[mask], token_counts[mask])
            tokens = token_counts[mask].sum()
            log_reduced_mean(
                self,
                _log_name(f"loss/{family.value}", stage=stage),
                mean,
                on_step=stage is None,
                on_epoch=True,
            )
            log_reduced_sum(
                self,
                _log_name(f"tokens/{family.value}", stage=stage),
                tokens,
                on_step=stage is None,
                on_epoch=False,
                skip_zero=True,
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
            log_reduced_mean(
                self,
                _log_name(f"loss/{name}", stage=stage),
                reduced_weighted_mean(row_loss[mask], token_counts[mask]),
                on_step=stage is None,
                on_epoch=True,
            )

    def _log_acoustic_frame_count(self, batch: CausalLMBatch, *, stage: str | None) -> None:
        if batch.target_audio is None:
            return
        frames = batch.target_audio.acoustic_mask.sum().float()
        log_reduced_sum(
            self,
            _log_name("acoustic_frames", stage=stage),
            frames,
            on_step=stage is None,
            on_epoch=stage is not None,
        )

    def _acoustic_loss(
        self,
        batch: CausalLMBatch,
        *,
        hidden_states: Tensor | None = None,
        stage: str | None = None,
    ) -> Tensor | AcousticFlowLossStats:
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
        inputs = self.model.acoustic_flow_inputs(
            batch,
            self.bpe,
            target_features,
            hidden_states=hidden_states,
            target_mask=target_mask,
            noise=None,
            acoustic_condition=None,
            source_feature_extractor=feature_extractor,
        )
        if not isinstance(inputs, AcousticFlowInputs):
            raise TypeError("model acoustic_flow_inputs() must return AcousticFlowInputs.")
        stats = self.model.acoustic_flow_loss_stats_from_inputs(inputs, timesteps=None)
        if not isinstance(stats, AcousticFlowLossStats):
            raise TypeError(
                "model acoustic_flow_loss_stats_from_inputs() must return "
                "AcousticFlowLossStats."
            )
        self._log_acoustic_condition_stats(
            inputs,
            timesteps=stats.timesteps,
            stage=stage,
        )
        return stats

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
            log_reduced_mean(
                self,
                _log_name(f"loss/acoustic_t/{name}", stage=stage),
                reduced_weighted_mean(row_loss[mask], row_weight[mask]),
                on_step=stage is None,
                on_epoch=True,
            )

    def _log_acoustic_condition_stats(
        self,
        inputs: AcousticFlowInputs,
        *,
        timesteps: Tensor,
        stage: str | None,
    ) -> None:
        tensors = self.model.acoustic_condition_tensors(inputs, timesteps=timesteps)
        if not isinstance(tensors, DiTConditionTensors):
            raise TypeError("model acoustic_condition_tensors() must return DiTConditionTensors.")
        mask = inputs.mask.to(device=inputs.last_hidden_state.device, dtype=torch.bool)
        frame_weights = mask.to(dtype=inputs.last_hidden_state.dtype)
        batch_weights = frame_weights.sum(dim=1).gt(0).to(dtype=inputs.last_hidden_state.dtype)
        self._log_tensor_stats(
            _log_name("condition/hidden", stage=stage),
            tensors.hidden.detach(),
            frame_weights,
            on_step=stage is None,
            on_epoch=True,
        )
        self._log_tensor_stats(
            _log_name("condition/time", stage=stage),
            tensors.time.detach().squeeze(1),
            batch_weights,
            on_step=stage is None,
            on_epoch=True,
        )
        self._log_tensor_stats(
            _log_name("condition/acoustic", stage=stage),
            tensors.acoustic.detach().squeeze(1),
            batch_weights,
            on_step=stage is None,
            on_epoch=True,
        )

    def _log_tensor_stats(
        self,
        prefix: str,
        values: Tensor,
        weights: Tensor,
        *,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        stats = _weighted_tensor_stats(values, weights)
        log_reduced_value(
            self,
            f"{prefix}_mean",
            stats.mean,
            stats.weight,
            on_step=on_step,
            on_epoch=on_epoch,
        )
        log_reduced_value(
            self,
            f"{prefix}_std",
            stats.std,
            stats.weight,
            on_step=on_step,
            on_epoch=on_epoch,
        )


def _validate_scheduler(train: TrainConfig) -> None:
    if train.schedule != "warmup_cosine":
        raise ValueError("train.schedule must be 'warmup_cosine'.")


def _scheduler_total_steps(train: TrainConfig) -> int:
    return train.max_steps


def _scheduler_warmup_steps(train: TrainConfig) -> int:
    ratio = train.warmup_ratio
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError("train.warmup_ratio must be greater than or equal to 0 and less than 1.")
    return round(train.max_steps * ratio)


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


def _acoustic_reduced_mean(
    batch: CausalLMBatch,
    acoustic: Tensor | AcousticFlowLossStats,
) -> ReducedMean:
    if isinstance(acoustic, AcousticFlowLossStats):
        return reduced_weighted_mean(acoustic.row_loss, acoustic.row_weight)
    if batch.target_audio is None:
        raise RuntimeError("acoustic loss requires target_audio in the batch.")
    weight = batch.target_audio.acoustic_mask.sum().to(device=acoustic.device, dtype=acoustic.dtype)
    return reduced_weighted_mean(acoustic.reshape(()), weight.reshape(()))


@dataclass(frozen=True)
class _TensorStats:
    mean: Tensor
    std: Tensor
    weight: Tensor


def _weighted_tensor_stats(values: Tensor, weights: Tensor) -> _TensorStats:
    if values.dim() < 1:
        raise ValueError("condition stats values must have at least one dimension.")
    if weights.shape != values.shape[:-1]:
        raise ValueError("condition stats weights must match values except feature dimension.")
    weights = weights.to(device=values.device, dtype=values.dtype)
    expanded_weights = weights.unsqueeze(-1).expand_as(values)
    mean = reduced_weighted_mean(values.reshape(-1), expanded_weights.reshape(-1))
    centered = values - mean.value.to(device=values.device, dtype=values.dtype)
    variance = reduced_weighted_mean(
        centered.square().reshape(-1),
        expanded_weights.reshape(-1),
    )
    return _TensorStats(
        mean=mean.value,
        std=variance.value.clamp_min(0.0).sqrt(),
        weight=mean.weight,
    )


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


def _last_hidden_state(output: CausalLMOutputWithPast) -> Tensor | None:
    hidden_states = output.hidden_states
    if hidden_states is None:
        return None
    if len(hidden_states) == 0:
        raise RuntimeError("model output hidden_states must not be empty.")
    return hidden_states[-1]


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
