from __future__ import annotations

from dataclasses import asdict
from typing import TypedDict

from anytrain.optim import create_llm_lightning_optimizers
from lightning.pytorch import LightningModule
from torch import Tensor, device as TorchDevice
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..config import TrainConfig
from ..model.acoustic import acoustic_features_from_batch_side
from ..model.orchestrator import Orchestrator
from ..types import CausalLMBatch, IGNORE_INDEX, LongCatBatchSide


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
        self.acoustic_feature_extractor = acoustic_feature_extractor
        self.save_hyperparameters({"train": asdict(self.train_config)})

    def forward(self, batch: CausalLMBatch) -> CausalLMOutputWithPast:
        return self.model(batch)

    def training_step(self, batch: CausalLMBatch, batch_idx: int) -> Tensor:
        del batch_idx
        loss = self._loss(batch)
        self.log(
            "train/loss",
            loss,
            batch_size=batch.input_ids.size(0),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/supervised_tokens",
            batch.labels.ne(IGNORE_INDEX).sum().float(),
            batch_size=batch.input_ids.size(0),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: CausalLMBatch, batch_idx: int) -> Tensor:
        del batch_idx
        loss = self._loss(batch)
        self.log(
            "val/loss",
            loss,
            batch_size=batch.input_ids.size(0),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
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
        return CausalLMBatch(
            input_ids=batch.input_ids.to(device=device),
            attention_mask=batch.attention_mask.to(device=device),
            labels=batch.labels.to(device=device),
            logits_to_keep=logits_to_keep,
            source_audio=_move_longcat_side(batch.source_audio, device),
            target_audio=_move_longcat_side(batch.target_audio, device),
        )

    def configure_optimizers(self) -> _LightningOptimizerConfig:
        train = self.train_config
        return create_llm_lightning_optimizers(
            self.model,
            preset=train.optimizer_preset,
            optimizer=train.optimizer,
            lr=train.learning_rate,
            weight_decay=train.weight_decay,
            schedule=train.schedule,
            warmup_steps=train.warmup_steps,
            total_steps=_scheduler_total_steps(train),
            stable_steps=train.stable_steps,
            decay_steps=train.decay_steps,
            min_lr_ratio=train.min_lr_ratio,
        )

    def _loss(self, batch: CausalLMBatch) -> Tensor:
        loss = self.model(batch).loss
        if loss is None:
            raise RuntimeError("model output must include loss.")
        acoustic_weight = self.train_config.acoustic_loss_weight
        if acoustic_weight <= 0.0:
            return loss
        return loss + acoustic_weight * self._acoustic_loss(batch)

    def _acoustic_loss(self, batch: CausalLMBatch) -> Tensor:
        if self.bpe is None:
            raise RuntimeError("acoustic loss requires a LongCat BPE tokenizer.")
        if self.acoustic_feature_extractor is None:
            raise RuntimeError("acoustic loss requires an acoustic feature extractor.")
        if batch.target_audio is None:
            raise RuntimeError("acoustic loss requires target_audio in the batch.")
        target_features, target_mask = acoustic_features_from_batch_side(
            batch.target_audio,
            feature_extractor=self.acoustic_feature_extractor,
        )
        return self.model.acoustic_flow_loss(
            batch,
            self.bpe,
            target_features,
            target_mask=target_mask,
            source_feature_extractor=self.acoustic_feature_extractor,
        )


def _scheduler_total_steps(train: TrainConfig) -> int | None:
    if train.schedule == "constant":
        return None
    return train.max_steps


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
