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
from ..model.orchestrator import Orchestrator
from ..types import CausalLMBatch, IGNORE_INDEX


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
    ) -> None:
        super().__init__()
        self.model = model
        self.train_config = train or TrainConfig()
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
        return loss


def _scheduler_total_steps(train: TrainConfig) -> int | None:
    if train.schedule == "constant":
        return None
    return train.max_steps
