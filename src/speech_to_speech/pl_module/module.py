from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast

import torch
from anytrain.optim.llm import create_optimizer
from lightning.pytorch import LightningModule
from torch import nn

from ..datamodule.types import ModelBatch, RawSingleBatch, TrainBatch, TrainInputBatch
from ..generation.service import generate_responses
from ..generation.text import TextProbe, TextProbeResult, evaluate_text
from ..generation.types import Request, Result
from ..loss.objective import Objective
from ..loss.types import LossItem, Outputs
from ..generation.protocol import TextEvaluationModel


@dataclass(frozen=True)
class Config:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01


ModelT = TypeVar("ModelT", bound=TextEvaluationModel)


class BatchMaterializer(Protocol):
    def __call__(
        self,
        batch: TrainInputBatch,
        *,
        device: torch.device | None = None,
    ) -> TrainBatch: ...


class SpeechToSpeechModule(LightningModule, Generic[ModelT]):
    def __init__(
        self,
        config: Config,
        *,
        model: ModelT,
        objective: Objective[ModelT],
        batch_materializer: BatchMaterializer | None = None,
    ) -> None:
        super().__init__()

        self.config = config

        self.model = model
        self.objective = objective
        self.batch_materializer = batch_materializer
        self._current_loss_outputs: Outputs | None = None

    def training_step(self, batch: TrainInputBatch, batch_idx: int = 0):
        del batch_idx
        batch = self.materialize_batch(batch)
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

    def validation_step(self, batch: TrainInputBatch, batch_idx: int = 0):
        del batch_idx
        outputs = self._loss_outputs(self.materialize_batch(batch))
        token = outputs.get("token")
        if token is not None:
            self._log_validation_metric("token_ce", token, "tokens")
        rvq = outputs.get("rvq")
        if rvq is not None:
            self._log_validation_metric("rvq_ce", rvq, "frames")
            details = rvq.details
            if details is None:
                raise TypeError("RVQ validation requires loss details.")
            for key in sorted(details):
                if not key.startswith("codebook_"):
                    continue
                suffix = "" if key.endswith("_top1") else "_ce"
                self._log_validation_detail(
                    f"rvq_{key}{suffix}",
                    rvq,
                    key,
                    "frames",
                )
        return outputs

    def _log_validation_metric(
        self,
        name: str,
        item: LossItem,
        unit: str,
    ) -> None:
        details = item.details
        if details is None or unit not in details:
            raise TypeError(f"validation loss item requires {unit!r} details.")
        self._log_validation_value(name, item.loss, details[unit])

    def _log_validation_detail(
        self,
        name: str,
        item: LossItem,
        key: str,
        unit: str,
    ) -> None:
        details = item.details
        if details is None or key not in details or unit not in details:
            raise TypeError(f"validation loss item requires {key!r} and {unit!r}.")
        self._log_validation_value(name, details[key], details[unit])

    def _log_validation_value(
        self,
        name: str,
        values: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        if values.shape != weights.shape:
            raise ValueError("validation values and weights must align by row.")
        resolved = weights.to(dtype=values.dtype)
        count = resolved.sum()
        if bool(count.le(0)):
            raise ValueError("validation metric weights must contain a positive total.")
        self.log(
            f"val/{name}",
            (values * resolved).sum() / count,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=int(count.detach().item()),
        )

    def _loss_outputs(self, batch: TrainBatch) -> Outputs:
        if not isinstance(batch, tuple):
            if not isinstance(batch, ModelBatch):
                raise TypeError(
                    "training_step requires ModelBatch unless a batch materializer "
                    "converts the incoming batch."
                )
            outputs = [self.objective.forward(batch, self.model)]
        else:
            if any(not isinstance(item, ModelBatch) for item in batch):
                raise TypeError(
                    "joint training batches must contain ModelBatch values after "
                    "materialization."
                )
            outputs = [self.objective.forward(item, self.model) for item in batch]
        return self.objective.reduce(
            outputs,
        )

    def materialize_batch(self, batch: TrainInputBatch) -> TrainBatch:
        if self.batch_materializer is None:
            if isinstance(batch, RawSingleBatch) or (
                isinstance(batch, tuple)
                and any(isinstance(item, RawSingleBatch) for item in batch)
            ):
                raise TypeError(
                    "raw waveform batches require a batch materializer before loss."
                )
            return cast(TrainBatch, batch)
        return self.batch_materializer(batch, device=self.device)

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
