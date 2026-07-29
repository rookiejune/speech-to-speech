from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

import torch
from anytrain.lightning import validation
from anytrain.optim.llm import create_optimizer
from lightning.pytorch import LightningModule
from torch import nn

from ..datamodule.types import ModelBatch, RawSpeechBatch, TrainBatch, TrainInputBatch
from ..audio_route import Config as AudioRouteConfig
from ..generation.service import generate_responses
from ..generation.text import TextProbe, TextProbeResult, evaluate_text
from ..generation.types import Request, Result
from ..loss import validation_metrics
from ..loss.objective import Objective
from ..loss.protocol import TokenObjectiveModel
from ..loss.types import Outputs
from ..generation.protocol import TextEvaluationModel


@dataclass(frozen=True)
class Config:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01


class ModuleModel(TextEvaluationModel, TokenObjectiveModel, Protocol):
    pass


ModelT = TypeVar("ModelT", bound=ModuleModel)
_AUDIO_ROUTE_KEY = "speech_to_speech_audio_route"


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
        outputs = self._outputs(
            self.materialize_batch(batch),
            self.objective.validation,
        )
        validation.log(self, validation_metrics(outputs))
        return outputs

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        # Raw fallback items must stay together on CPU until the codec materializer
        # rebuilds the complete batch on one device.
        if isinstance(batch, RawSpeechBatch):
            return batch
        transfer = super().transfer_batch_to_device
        if isinstance(batch, tuple):
            return tuple(
                item
                if isinstance(item, RawSpeechBatch)
                else transfer(item, device, dataloader_idx)
                for item in batch
            )
        return transfer(batch, device, dataloader_idx)

    def _loss_outputs(self, batch: TrainBatch) -> Outputs:
        return self._outputs(batch, self.objective.forward)

    def _outputs(
        self,
        batch: TrainBatch,
        objective: Callable[[ModelBatch, ModelT], Outputs],
    ) -> Outputs:
        if not isinstance(batch, tuple):
            if not isinstance(batch, ModelBatch):
                raise TypeError(
                    "training_step requires ModelBatch unless a batch materializer "
                    "converts the incoming batch."
                )
            outputs = [objective(batch, self.model)]
        else:
            if any(not isinstance(item, ModelBatch) for item in batch):
                raise TypeError(
                    "joint training batches must contain ModelBatch values after "
                    "materialization."
                )
            outputs = [objective(item, self.model) for item in batch]
        return self.objective.reduce(
            outputs,
        )

    def materialize_batch(self, batch: TrainInputBatch) -> TrainBatch:
        if self.batch_materializer is None:
            if isinstance(batch, RawSpeechBatch) or (
                isinstance(batch, tuple)
                and any(isinstance(item, RawSpeechBatch) for item in batch)
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

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint[_AUDIO_ROUTE_KEY] = _audio_route_payload(
            self.model.runtime.audio_route
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        expected = _audio_route_payload(self.model.runtime.audio_route)
        if _AUDIO_ROUTE_KEY not in checkpoint:
            if expected is not None:
                raise ValueError(
                    "checkpoint is missing the fixed audio route contract."
                )
            return
        actual = checkpoint[_AUDIO_ROUTE_KEY]
        if actual != expected:
            raise ValueError(
                f"checkpoint audio route does not match runtime: {actual!r} != {expected!r}."
            )

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


def _audio_route_payload(route: AudioRouteConfig | None) -> dict[str, object] | None:
    if route is None:
        return None
    return {
        "prompt": {
            "source": route.prompt.source.value,
            "streams": [stream.value for stream in route.prompt.canonical_streams],
        },
        "output": {
            "streams": [stream.value for stream in route.output.canonical_streams],
        },
        "decode": {
            "semantic": route.decode.semantic.value,
            "acoustic": route.decode.acoustic.value,
        },
        "grammar": "audio-route-v1",
    }
