from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar, cast

import torch
from anytrain.lightning import validation
from anytrain.optim.llm import create_optimizer
from lightning.pytorch import LightningModule
from peft import LoraConfig
from torch import nn

from ..datamodule.types import ModelBatch, RawSpeechBatch, TrainInput
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
    optimizer: str = "adamw"


class ModuleModel(TextEvaluationModel, TokenObjectiveModel, Protocol):
    @property
    def lora_config(self) -> LoraConfig | None: ...


ModelT = TypeVar("ModelT", bound=ModuleModel)
_AUDIO_ROUTE_KEY = "speech_to_speech_audio_route"
_PEFT_KEY = "speech_to_speech_peft"


class BatchMaterializer(Protocol):
    def __call__(
        self,
        batch: TrainInput,
        *,
        device: torch.device | None = None,
    ) -> ModelBatch: ...


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

    def training_step(self, batch: TrainInput, batch_idx: int = 0):
        del batch_idx
        batch = self.materialize_batch(batch)
        outputs = self._loss_outputs(batch)
        self._current_loss_outputs = outputs
        self.log(
            "loss",
            outputs["loss"],
            prog_bar=True,
            on_step=True,
            sync_dist=True,
        )
        return outputs

    def validation_step(self, batch: TrainInput, batch_idx: int = 0):
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
        if isinstance(batch, ModelBatch):
            return batch.to(device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _loss_outputs(self, batch: ModelBatch) -> Outputs:
        return self._outputs(batch, self.objective.forward)

    def _outputs(
        self,
        batch: ModelBatch,
        objective: Callable[[ModelBatch, ModelT], Outputs],
    ) -> Outputs:
        if not isinstance(batch, ModelBatch):
            raise TypeError(
                "training_step requires ModelBatch unless a batch materializer "
                "converts the incoming batch."
            )
        return objective(batch, self.model)

    def materialize_batch(self, batch: TrainInput) -> ModelBatch:
        if self.batch_materializer is None:
            if isinstance(batch, RawSpeechBatch):
                raise TypeError(
                    "raw waveform batches require a batch materializer before loss."
                )
            return batch
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
        checkpoint[_PEFT_KEY] = _peft_payload(self.model.lora_config)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        expected = _audio_route_payload(self.model.runtime.audio_route)
        _validate_audio_route_checkpoint(checkpoint, expected)
        _validate_peft_checkpoint(checkpoint, self.model.lora_config)

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
            optimizer=self.config.optimizer,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )


def _validate_audio_route_checkpoint(
    checkpoint: dict[str, Any],
    expected: dict[str, object] | None,
) -> None:
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


def _validate_peft_checkpoint(
    checkpoint: dict[str, Any],
    config: LoraConfig | None,
) -> None:
    expected = _peft_payload(config)
    if _PEFT_KEY not in checkpoint:
        if config is not None:
            raise ValueError("checkpoint is missing the PEFT LoRA contract.")
        return
    actual = checkpoint[_PEFT_KEY]
    if not _peft_payload_matches(actual, expected):
        raise ValueError(
            f"checkpoint PEFT LoRA contract does not match model: "
            f"{actual!r} != {expected!r}."
        )


def _peft_payload(config: LoraConfig | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return {
        "grammar": "peft-lora-v2",
        "adapter_name": "speech",
        "config": _peft_config(config),
        "defaults": _peft_config(LoraConfig()),
    }


def _peft_config(config: LoraConfig) -> dict[str, Any]:
    values = cast(dict[str, Any], _checkpoint_value(config.to_dict()))
    values.pop("peft_version", None)
    return values


def _peft_payload_matches(
    actual: object,
    expected: dict[str, Any] | None,
) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if not isinstance(actual, dict):
        return False
    if actual.get("grammar") != expected["grammar"]:
        return False
    if actual.get("adapter_name") != expected["adapter_name"]:
        return False
    actual_config = actual.get("config")
    actual_defaults = actual.get("defaults")
    expected_config = expected["config"]
    expected_defaults = expected["defaults"]
    if not isinstance(actual_config, dict) or not isinstance(actual_defaults, dict):
        return False
    common = actual_config.keys() & expected_config.keys()
    if any(actual_config[key] != expected_config[key] for key in common):
        return False
    actual_only = actual_config.keys() - expected_config.keys()
    if any(
        key not in actual_defaults or actual_config[key] != actual_defaults[key]
        for key in actual_only
    ):
        return False
    expected_only = expected_config.keys() - actual_config.keys()
    return not any(
        key not in expected_defaults or expected_config[key] != expected_defaults[key]
        for key in expected_only
    )


def _checkpoint_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _checkpoint_value(value.value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("PEFT checkpoint config keys must be strings.")
            result[key] = _checkpoint_value(item)
        return result
    if isinstance(value, (set, frozenset)):
        items = [_checkpoint_value(item) for item in value]
        try:
            return sorted(items)
        except TypeError as error:
            raise TypeError(
                "PEFT checkpoint config sets must contain sortable values."
            ) from error
    if isinstance(value, (list, tuple)):
        return [_checkpoint_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"unsupported PEFT checkpoint config value: {type(value).__name__}."
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
