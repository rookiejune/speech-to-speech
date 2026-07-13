from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from anytrain.optim.llm import create_optimizer
from lightning.pytorch import LightningModule
from torch import nn

from ..datamodule.types import ModelBatch
from ..loss.module import Loss, RVQLoss
from ..loss.types import Outputs
from ..model.protocol import FlowModel, RVQModel
from .generation import Request, Result, generate
from .text import TextProbe, TextProbeResult, evaluate_text


@dataclass(frozen=True)
class Config:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01


class SpeechToSpeech(LightningModule):
    def __init__(
        self,
        config: Config,
        *,
        model: FlowModel | RVQModel,
        loss: Loss | RVQLoss,
    ) -> None:
        super().__init__()

        self.config = config

        self.model = model
        self.loss = loss
        self._current_loss_outputs: Outputs | None = None

    def training_step(self, batch: ModelBatch, batch_idx: int = 0):
        del batch_idx
        outputs = cast(Any, self.loss).forward(batch, self.model)
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
    def generate(
        self,
        requests: Sequence[Request],
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> list[Result]:
        was_training = self.training
        self.eval()
        try:
            return generate(
                requests,
                self.model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                use_cache=use_cache,
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def evaluate_text(
        self,
        probes: Mapping[str, TextProbe],
        *,
        max_new_tokens: int = 128,
    ) -> dict[str, TextProbeResult]:
        was_training = self.training
        self.eval()
        try:
            return evaluate_text(
                probes,
                self.model,
                max_new_tokens=max_new_tokens,
            )
        finally:
            self.train(was_training)

    def configure_optimizers(self):
        return create_optimizer(
            cast(nn.Module, cast(object, self.model)),
            preset="sft",
            optimizer="adamw",
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
