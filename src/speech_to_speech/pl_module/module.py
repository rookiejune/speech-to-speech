from __future__ import annotations

from dataclasses import dataclass

import torch
from anytrain.optim.llm import create_optimizer
from lightning.pytorch import LightningModule

from ..datamodule.types import ModelBatch
from ..loss.module import Loss
from ..loss.types import Outputs
from ..model.acoustic import SpeechToSpeechFlowModel
from .generation import generate_batch, generate_waveforms


@dataclass(frozen=True)
class Config:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01


class SpeechToSpeech(LightningModule):
    def __init__(
        self,
        config: Config,
        *,
        model: SpeechToSpeechFlowModel,
        loss: Loss,
    ) -> None:
        super().__init__()

        self.config = config

        self.model = model
        self.loss = loss
        self._current_loss_outputs: Outputs | None = None

    def training_step(self, batch: ModelBatch, batch_idx: int = 0):
        del batch_idx
        outputs = self.loss.forward(batch, self.model)
        self._current_loss_outputs = outputs
        self.log("train/loss", outputs["loss"], prog_bar=True, on_step=True)
        return outputs

    def current_loss_outputs(self) -> Outputs:
        """Return loss outputs kept alive until the backward pass completes."""
        if self._current_loss_outputs is None:
            raise RuntimeError("loss outputs are unavailable outside a training step")
        return self._current_loss_outputs

    def on_after_backward(self) -> None:
        self._current_loss_outputs = None

    @torch.no_grad()
    def generate_batch(
        self,
        batch: ModelBatch,
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[torch.Tensor]:
        """Generate responses while preserving variable prompt lengths."""
        was_training = self.training
        self.eval()
        try:
            return generate_batch(
                batch,
                self.model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def generate_waveforms(
        self,
        batch: ModelBatch,
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[torch.Tensor]:
        was_training = self.training
        self.eval()
        try:
            return generate_waveforms(
                batch,
                self.model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        finally:
            self.train(was_training)

    def configure_optimizers(self):
        return create_optimizer(
            self.model,
            preset="sft",
            optimizer="adamw",
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
