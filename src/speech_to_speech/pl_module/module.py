from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar, cast

import torch
from anydataset.types import AudioView
from anytrain.evaluator.text import TextComparisonEvaluator
from anytrain.lightning import validation
from anytrain.optim.llm import create_optimizer
from lightning.pytorch import LightningModule
from peft import LoraConfig
from torch import nn

from ..datamodule.types import (
    FusedBatch,
    ModelBatch,
    RawSpeechBatch,
    TrainBatch,
    TrainInput,
)
from ..generation.batch import requests_from_batch
from ..generation.service import generate_responses
from ..generation.eval.text import (
    TextProbe,
    TextProbeResult,
    decode_text_ids,
    evaluate_text,
)
from ..generation.types import Request, Result
from ..loss.module import Objective
from ..loss.protocol import TokenObjectiveModel
from ..loss.types import Outputs, combine_outputs
from ..loss.validation import validation_metrics
from ..generation.protocol import TextEvaluationModel
from ..prediction import PredictionModality
from ..runtime import AudioSequenceLayout
from ..task import Task


@dataclass(frozen=True)
class Config:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    mt_validation_max_new_tokens: int = 256

    def __post_init__(self) -> None:
        value = self.mt_validation_max_new_tokens
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("mt_validation_max_new_tokens must be positive.")


class ModuleModel(TextEvaluationModel, TokenObjectiveModel, Protocol):
    @property
    def lora_config(self) -> LoraConfig | None: ...


ModelT = TypeVar("ModelT", bound=ModuleModel)
_AUDIO_GRAMMAR_KEY = "speech_to_speech_audio_grammar"
_AUDIO_SEQUENCE_LAYOUT_KEY = "speech_to_speech_audio_sequence_layout"
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
        self._current_gradient_loss_groups: dict[str, Outputs] | None = None
        self.mt_validation_evaluator = TextComparisonEvaluator()
        self._mt_validation_seen = False

    def training_step(self, batch: TrainBatch, batch_idx: int = 0):
        del batch_idx
        self._current_gradient_loss_groups = None
        outputs = self._training_outputs(batch)
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
        materialized = self.materialize_batch(batch)
        if _is_mt_validation_batch(materialized):
            return self._mt_validation_step(materialized)
        outputs = self._outputs(materialized, self.objective.validation)
        validation.log(self, validation_metrics(outputs))
        return outputs

    def on_validation_epoch_start(self) -> None:
        self.mt_validation_evaluator.reset()
        self._mt_validation_seen = False

    def on_validation_epoch_end(self) -> None:
        if not _distributed_any(self._mt_validation_seen, self.device):
            return
        metrics = self.mt_validation_evaluator.compute()
        for name, value in sorted(metrics.items()):
            self.log(
                f"val/mt/{name}",
                float(value),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

    def _mt_validation_step(self, batch: ModelBatch) -> dict[str, torch.Tensor]:
        generations = generate_responses(
            requests_from_batch(batch),
            self.model,
            max_new_tokens=self.config.mt_validation_max_new_tokens,
            do_sample=False,
        )
        predictions = [
            decode_text_ids(self.model.runtime, result["response_ids"])
            for result in generations
        ]
        references = _reference_texts(batch, self.model.runtime)
        self.mt_validation_evaluator.update(predictions, references)
        self._mt_validation_seen = True
        return {
            "mt_validation_samples": batch.input_ids.new_tensor(
                len(predictions),
                dtype=torch.float32,
            )
        }

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
        if isinstance(batch, FusedBatch):
            return FusedBatch(
                tuple(
                    self.transfer_batch_to_device(child, device, dataloader_idx)
                    for child in batch.batches
                ),
                batch.loader_names,
            )
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _training_outputs(self, batch: TrainBatch) -> Outputs:
        if isinstance(batch, FusedBatch):
            outputs = [
                self._loss_outputs(self.materialize_batch(child))
                for child in batch.batches
            ]
            combined = _combine_training_outputs(outputs)
            if batch.loader_names is not None:
                grouped: dict[str, list[Outputs]] = {}
                for name, output in zip(batch.loader_names, outputs):
                    grouped.setdefault(name, []).append(output)
                self._current_gradient_loss_groups = {
                    "batch": combined,
                    **{
                        name: _combine_training_outputs(values)
                        for name, values in grouped.items()
                    },
                }
            return combined
        return self._loss_outputs(self.materialize_batch(batch))

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

    def current_gradient_loss_groups(self) -> Mapping[str, Outputs]:
        if self._current_gradient_loss_groups is None:
            return {"batch": self.current_loss_outputs()}
        return self._current_gradient_loss_groups

    def on_after_backward(self) -> None:
        self._current_loss_outputs = None
        self._current_gradient_loss_groups = None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint[_AUDIO_SEQUENCE_LAYOUT_KEY] = _audio_sequence_layout_payload(
            self.model.runtime.audio_sequence_layout
        )
        checkpoint[_PEFT_KEY] = _peft_payload(self.model.lora_config)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        expected = _audio_sequence_layout_payload(
            self.model.runtime.audio_sequence_layout
        )
        _validate_audio_sequence_layout_checkpoint(
            checkpoint,
            expected,
            self.model.runtime,
        )
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


def _combine_training_outputs(outputs: Sequence[Outputs]) -> Outputs:
    if not outputs:
        raise ValueError("cannot combine an empty fused training step.")
    combined = combine_outputs(outputs)
    combined["loss"] = torch.stack([output["loss"] for output in outputs]).mean()
    return combined


def _is_mt_validation_batch(batch: ModelBatch) -> bool:
    return bool(batch.tasks) and all(
        task is Task.MT and prediction is PredictionModality.TEXT
        for task, prediction in zip(batch.tasks, batch.predictions)
    )


def _distributed_any(value: bool, device: torch.device) -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value
    flag = torch.tensor([int(value)], device=device, dtype=torch.int64)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


def _reference_texts(batch: ModelBatch, runtime: Any) -> list[str]:
    references = []
    for labels in batch.token_labels:
        token_ids = labels[labels.ne(-100)]
        if token_ids.numel() and int(token_ids[-1].item()) == runtime.eos_token_id:
            token_ids = token_ids[:-1]
        references.append(decode_text_ids(runtime, token_ids))
    return references


def _validate_audio_sequence_layout_checkpoint(
    checkpoint: dict[str, Any],
    expected: str,
    runtime: Any,
) -> None:
    if _AUDIO_SEQUENCE_LAYOUT_KEY in checkpoint:
        actual = checkpoint[_AUDIO_SEQUENCE_LAYOUT_KEY]
        if actual != expected:
            raise ValueError(
                f"checkpoint audio sequence layout does not match runtime: "
                f"{actual!r} != {expected!r}."
            )
        return

    actual = checkpoint.get(_AUDIO_GRAMMAR_KEY)
    if actual is None:
        return
    expected_legacy = _legacy_audio_grammar_payload(runtime)
    if actual != expected_legacy:
        raise ValueError(
            f"checkpoint legacy audio grammar does not match runtime layout: "
            f"{actual!r} != {expected_legacy!r}."
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


def _audio_sequence_layout_payload(layout: AudioSequenceLayout) -> str:
    if not isinstance(layout, AudioSequenceLayout):
        raise TypeError("runtime audio_sequence_layout must be an AudioSequenceLayout.")
    return layout.value


def _legacy_audio_grammar_payload(runtime: Any) -> dict[str, object]:
    layout = runtime.audio_sequence_layout
    if not isinstance(layout, AudioSequenceLayout):
        raise TypeError("runtime audio_sequence_layout must be an AudioSequenceLayout.")
    audio_view = runtime.audio_view
    if layout is AudioSequenceLayout.FLATTENED:
        streams = ["acoustic", "semantic"]
        acoustic = "output"
    elif audio_view is AudioView.BICODEC:
        streams = ["semantic"]
        acoustic = "prompt"
    else:
        streams = ["semantic"]
        acoustic = "generator"
    return {
        "output": {
            "streams": streams,
        },
        "decode": {
            "semantic": "output",
            "acoustic": acoustic,
        },
        "grammar": "audio-grammar-v1",
    }
