from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import torch
from anytrain.optim.llm import create_optimizer
from lightning.pytorch import LightningModule
from torch import nn

from ..datamodule.types import TrainBatch
from ..generation.service import generate_responses
from ..generation.text import TextProbe, TextProbeResult, evaluate_text
from ..generation.types import Request, Result
from ..loss.objective import Objective
from ..loss.types import Outputs
from ..generation.protocol import TextEvaluationModel


@dataclass(frozen=True)
class Config:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01


ModelT = TypeVar("ModelT", bound=TextEvaluationModel)


class SpeechToSpeechModule(LightningModule, Generic[ModelT]):
    def __init__(
        self,
        config: Config,
        *,
        model: ModelT,
        objective: Objective[ModelT],
    ) -> None:
        super().__init__()

        self.config = config

        self.model = model
        self.objective = objective
        self._current_loss_outputs: Outputs | None = None

    def training_step(self, batch: TrainBatch, batch_idx: int = 0):
        del batch_idx
        outputs = self._loss_outputs(batch)
        self._current_loss_outputs = outputs
        self.log(
            "train/loss",
            outputs["loss"],
            prog_bar=True,
            on_step=True,
            sync_dist=True,
        )
        return outputs

    def _loss_outputs(self, batch: TrainBatch) -> Outputs:
        if not isinstance(batch, tuple):
            outputs = [self.objective.forward(batch, self.model)]
        else:
            outputs = [self.objective.forward(item, self.model) for item in batch]
        return self.objective.reduce(
            outputs,
        )

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
            return generate_responses(
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
