from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Protocol, TypeVar, cast

import torch
from anydataset.types import Modality
from anytrain.evaluator.text import TextComparisonEvaluator
from anytrain.lightning import validation
from anytrain.lightning.schedule import ScheduleRuntime
from anytrain.optim.llm import create_optimizer
from lightning.pytorch import LightningModule
from peft import LoraConfig
from torch import nn

from ..datamodule.batch import (
    FusedBatch,
    LoaderBatch,
    ModelBatch,
    TrainBatch,
    TrainInput,
)
from ..datamodule.sample import RawSpeechBatch
from ..generation.service import requests_from_batch
from ..generation.service import generate_responses
from ..generation.evaluation import (
    TextProbe,
    TextProbeResult,
    evaluate_text,
)
from ..generation.text import decode_text_ids
from ..generation.contract import Result
from ..loss.contract import Outputs, TokenObjectiveModel, combine_outputs
from ..loss.ctc import CTCConfig
from ..loss.supervised import Objective, validation_metrics
from ..model.checkpoint_contract import ModelCheckpointContract, validate_checkpoint_contract
from ..model.base import Model
from ..generation.contract import TextEvaluationModel
from ..task import PredictionModality, Request, Task
from .optim import Config as OptimConfig


@dataclass(frozen=True)
class Config:
    mt_validation_max_new_tokens: int = 256
    audio_neighbor_smoothing: float = 0.0
    ctc: CTCConfig = field(default_factory=CTCConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.ctc, CTCConfig):
            raise TypeError("pl_module CTC config must be a loss CTCConfig.")
        value = self.mt_validation_max_new_tokens
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("mt_validation_max_new_tokens must be positive.")
        smoothing = self.audio_neighbor_smoothing
        if (
            isinstance(smoothing, bool)
            or not isinstance(smoothing, (int, float))
            or not math.isfinite(smoothing)
            or not 0 <= smoothing < 1
        ):
            raise ValueError("audio_neighbor_smoothing must be in [0, 1).")


class ModuleModel(TextEvaluationModel, TokenObjectiveModel, Protocol):
    @property
    def checkpoint_contract(self) -> ModelCheckpointContract: ...

    @property
    def lora_config(self) -> LoraConfig | None: ...


ModelT = TypeVar("ModelT", bound=ModuleModel)
_MODEL_SCHEMA_KEY = "speech_to_speech_model_schema"
_MODEL_SCHEMA = "v4"
_MODEL_CONTRACT_KEY = "speech_to_speech_model_contract"
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
        optim: OptimConfig | None = None,
        batch_materializer: BatchMaterializer | None = None,
        schedule_runtime: ScheduleRuntime | None = None,
    ) -> None:
        super().__init__()

        self.config = config

        self.model = model
        self.objective = objective
        self.optim = OptimConfig() if optim is None else optim
        self.batch_materializer = batch_materializer
        self.schedule_runtime = schedule_runtime
        self._current_loss_outputs: Outputs | None = None
        self._current_gradient_loss_groups: dict[str, Outputs] | None = None
        self._current_gradient_loader_outputs: tuple[tuple[str, Outputs], ...] | None = None
        self.validation_loader_names: tuple[str, ...] = ()
        self.text_validation_evaluators: dict[str, TextComparisonEvaluator] = {}
        self._text_validation_seen: set[str] = set()

    def training_step(self, batch: TrainBatch, batch_idx: int = 0):
        del batch_idx
        self._current_gradient_loss_groups = None
        self._current_gradient_loader_outputs = None
        outputs = self._training_outputs(batch)
        self._current_loss_outputs = outputs
        self.log(
            "loss",
            outputs["loss"],
            prog_bar=True,
            on_step=True,
            sync_dist=False,
        )
        return outputs

    def validation_step(
        self,
        batch: TrainInput,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
    ):
        del batch_idx
        materialized = self.materialize_batch(batch)
        namespace = _validation_namespace(
            self.validation_loader_names,
            dataloader_idx,
        )
        text_task = _text_generation_validation_task(materialized)
        if text_task is not None:
            return self._text_validation_step(
                materialized,
                task=text_task,
                namespace=namespace,
            )
        outputs = self._outputs(materialized, self.objective.validation)
        validation.log(
            self,
            validation_metrics(outputs),
            prefix="val" if namespace is None else f"val/{namespace}",
        )
        return outputs

    def on_validation_epoch_start(self) -> None:
        for evaluator in self.text_validation_evaluators.values():
            evaluator.reset()
        self._text_validation_seen.clear()

    def on_validation_epoch_end(self) -> None:
        for namespace, evaluator in sorted(self.text_validation_evaluators.items()):
            if not _distributed_any(namespace in self._text_validation_seen, self.device):
                continue
            metrics = evaluator.compute()
            for name, value in sorted(metrics.items()):
                self.log(
                    f"val/{namespace}/{name}",
                    float(value),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=False,
                )

    def set_validation_loader_names(self, names: Sequence[str]) -> None:
        resolved = tuple(names)
        if any(not isinstance(name, str) or not name for name in resolved):
            raise ValueError("validation loader names must be non-empty strings.")
        self.validation_loader_names = resolved
        for name in resolved:
            if name != "validation":
                self.text_validation_evaluators.setdefault(
                    name,
                    TextComparisonEvaluator(),
                )

    def _text_validation_step(
        self,
        batch: ModelBatch,
        *,
        task: Task,
        namespace: str | None,
    ) -> dict[str, torch.Tensor]:
        generations = generate_responses(
            requests_from_batch(batch),
            self.model,
            max_new_tokens=self.config.mt_validation_max_new_tokens,
            do_sample=False,
        )
        predictions = [
            decode_text_ids(self.model.runtime, result["response_ids"]) for result in generations
        ]
        references = _reference_texts(batch, self.model.runtime)
        metric_namespace = task.value if namespace is None else namespace
        evaluator = self.text_validation_evaluators.setdefault(
            metric_namespace,
            TextComparisonEvaluator(),
        )
        evaluator.update(predictions, references)
        self._text_validation_seen.add(metric_namespace)
        return {
            f"{metric_namespace}_validation_samples": batch.input_ids.new_tensor(
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
            hints = self._training_input_hints(batch)
            moved = batch.to(device, non_blocking=True)
            if hints is not None:
                modalities, positions_validated = hints
                moved.set_input_hints(
                    modalities,
                    audio_input_positions_validated=positions_validated,
                )
            return moved
        if isinstance(batch, FusedBatch):
            return FusedBatch(
                tuple(
                    self.transfer_batch_to_device(child, device, dataloader_idx)
                    for child in batch.batches
                ),
                batch.loader_names,
                batch.loss_weights,
            )
        if isinstance(batch, LoaderBatch):
            return LoaderBatch(
                self.transfer_batch_to_device(
                    batch.batch,
                    device,
                    dataloader_idx,
                ),
                batch.loader_name,
                batch.loss_scale,
            )
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def _training_outputs(self, batch: TrainBatch) -> Outputs:
        if isinstance(batch, FusedBatch):
            outputs = [self._loss_outputs(self.materialize_batch(child)) for child in batch.batches]
            combined = _combine_training_outputs(
                outputs,
                loss_weights=batch.loss_weights,
            )
            if batch.loader_names is not None:
                self._current_gradient_loader_outputs = tuple(zip(batch.loader_names, outputs))
            return combined
        if isinstance(batch, LoaderBatch):
            output = self._loss_outputs(self.materialize_batch(batch.batch))
            weighted = _scale_training_output(output, batch.loss_scale)
            self._current_gradient_loss_groups = {
                "batch": weighted,
                batch.loader_name: output,
            }
            return weighted
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
                raise TypeError("raw waveform batches require a batch materializer before loss.")
            return self._with_training_input_hints(batch)
        return self._with_training_input_hints(self.batch_materializer(batch, device=self.device))

    def _with_training_input_hints(self, batch: ModelBatch) -> ModelBatch:
        hints = self._training_input_hints(batch)
        if hints is not None:
            modalities, positions_validated = hints
            batch.set_input_hints(
                modalities,
                audio_input_positions_validated=positions_validated,
            )
        return batch

    def _training_input_hints(
        self,
        batch: ModelBatch,
    ) -> tuple[frozenset[Modality], bool] | None:
        if not isinstance(self.model, Model):
            return None
        return self.model.training_input_hints(
            batch.input_ids,
            batch.audio_input_positions,
        )

    def current_loss_outputs(self) -> Outputs:
        """Return loss outputs kept alive until the backward pass completes."""
        if self._current_loss_outputs is None:
            raise RuntimeError("loss outputs are unavailable outside a training step")
        return self._current_loss_outputs

    def current_gradient_loss_groups(self) -> Mapping[str, Outputs]:
        groups = self._current_gradient_loss_groups
        if groups is not None:
            return groups
        loader_outputs = self._current_gradient_loader_outputs
        if loader_outputs is None:
            return {"batch": self.current_loss_outputs()}
        grouped: dict[str, list[Outputs]] = {}
        for name, output in loader_outputs:
            grouped.setdefault(name, []).append(output)
        groups = {
            "batch": self.current_loss_outputs(),
            **{
                name: (values[0] if len(values) == 1 else _combine_training_outputs(values))
                for name, values in grouped.items()
            },
        }
        self._current_gradient_loss_groups = groups
        return groups

    def on_after_backward(self) -> None:
        self._current_loss_outputs = None
        self._current_gradient_loss_groups = None
        self._current_gradient_loader_outputs = None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint[_MODEL_SCHEMA_KEY] = _MODEL_SCHEMA
        checkpoint[_MODEL_CONTRACT_KEY] = self.model.checkpoint_contract.checkpoint_payload()
        checkpoint[_PEFT_KEY] = _peft_payload(self.model.lora_config)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        _validate_model_schema_checkpoint(checkpoint)
        _validate_model_contract_checkpoint(checkpoint, self.model.checkpoint_contract)
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
        optimizer = create_optimizer(
            cast(nn.Module, cast(object, self.model)),
            preset="sft",
            optimizer=self.optim.name,
            lr=self.optim.learning_rate,
            weight_decay=self.optim.weight_decay,
        )
        if self.schedule_runtime is None:
            return optimizer
        return self.schedule_runtime.configure_optimizers(optimizer)


def _combine_training_outputs(
    outputs: Sequence[Outputs],
    *,
    loss_weights: Sequence[float] | None = None,
) -> Outputs:
    if not outputs:
        raise ValueError("cannot combine an empty fused training step.")
    if loss_weights is not None and len(loss_weights) != len(outputs):
        raise ValueError("loss weights must align with fused training outputs.")
    losses = torch.stack([output["loss"] for output in outputs])
    if loss_weights is None:
        return combine_outputs(outputs, total_loss=losses.mean())
    resolved_weights = tuple(float(weight) for weight in loss_weights)
    if any(not math.isfinite(weight) or weight < 0 for weight in resolved_weights):
        raise ValueError("loss weights must be finite and non-negative.")
    if math.fsum(resolved_weights) <= 0:
        raise ValueError("loss weights must have a positive total.")
    weights = losses.new_tensor(resolved_weights)
    total = weights.sum()
    return combine_outputs(
        outputs,
        total_loss=(losses * weights).sum() / total,
    )


def _scale_training_output(output: Outputs, scale: float | None) -> Outputs:
    if scale is None:
        return output
    result = cast(Outputs, dict(output))
    result["loss"] = output["loss"] * float(scale)
    return result


def _text_generation_validation_task(batch: ModelBatch) -> Task | None:
    if not batch.tasks:
        return None
    task = batch.tasks[0]
    if task not in {Task.MT, Task.S2TT}:
        return None
    if all(
        candidate is task and prediction is PredictionModality.TEXT
        for candidate, prediction in zip(batch.tasks, batch.predictions)
    ):
        return task
    return None


def _validation_namespace(
    names: Sequence[str],
    dataloader_idx: int,
) -> str | None:
    if not names:
        return None
    if dataloader_idx < 0 or dataloader_idx >= len(names):
        raise IndexError(
            f"validation dataloader index {dataloader_idx} is outside {len(names)} loaders."
        )
    name = names[dataloader_idx]
    return None if name == "validation" else name


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


def _validate_model_contract_checkpoint(
    checkpoint: dict[str, Any],
    expected: ModelCheckpointContract,
) -> None:
    if _MODEL_CONTRACT_KEY not in checkpoint:
        raise ValueError("checkpoint is missing the model contract.")
    validate_checkpoint_contract(checkpoint[_MODEL_CONTRACT_KEY], expected)


def _validate_model_schema_checkpoint(checkpoint: dict[str, Any]) -> None:
    actual = checkpoint.get(_MODEL_SCHEMA_KEY)
    if actual != _MODEL_SCHEMA:
        raise ValueError(
            f"checkpoint model schema is incompatible: expected {_MODEL_SCHEMA!r}, got {actual!r}."
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
            f"checkpoint PEFT LoRA contract does not match model: {actual!r} != {expected!r}."
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
            raise TypeError("PEFT checkpoint config sets must contain sortable values.") from error
    if isinstance(value, (list, tuple)):
        return [_checkpoint_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported PEFT checkpoint config value: {type(value).__name__}.")
