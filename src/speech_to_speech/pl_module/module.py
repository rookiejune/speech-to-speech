from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

from anytrain.optim.llm import LightningOptimizerConfig
from lightning.pytorch import LightningModule
from torch import Tensor
from torch import device as TorchDevice
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..config import TrainConfig
from ..datamodule.types import CausalLMBatch
from ..model.acoustic import AcousticFlowLossStats
from ..model.orchestrator import AcousticFlowInputs, Orchestrator
from .acoustic import (
    TensorStats,
    acoustic_condition_metrics,
    acoustic_loss_stats,
    acoustic_reduced_mean,
    acoustic_t_bin_means,
)
from .batch import batch_to_device
from .metrics import (
    ReducedMean,
    log_reduced_mean,
    log_reduced_sum,
    log_reduced_value,
    reduced_weighted_mean,
    scaled_loss,
)
from .optim import configure_optimizers as configure_lightning_optimizers
from .semantic import (
    loss_token_counts,
    semantic_batch,
    semantic_row_loss,
)
from .task_metrics import log_task_group_losses, log_task_losses


@dataclass(frozen=True)
class _LossComponent:
    objective: Tensor
    value: Tensor
    weight: Tensor


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
        _validate_train_config(self.train_config)
        self.bpe = bpe
        self.__dict__["_acoustic_feature_extractor"] = acoustic_feature_extractor
        self.save_hyperparameters({"train": asdict(self.train_config)})

    @property
    def acoustic_feature_extractor(self) -> object | None:
        return self.__dict__.get("_acoustic_feature_extractor")

    def forward(self, batch: CausalLMBatch) -> CausalLMOutputWithPast:
        return self.model(batch)

    def training_step(self, batch: CausalLMBatch, batch_idx: int) -> Tensor:
        del batch_idx
        return self._step(batch, stage=None)

    def validation_step(self, batch: CausalLMBatch, batch_idx: int) -> Tensor:
        del batch_idx
        return self._step(batch, stage="val")

    def transfer_batch_to_device(
        self,
        batch: CausalLMBatch,
        device: TorchDevice,
        dataloader_idx: int,
    ) -> CausalLMBatch:
        del dataloader_idx
        return batch_to_device(batch, device)

    def configure_optimizers(self) -> LightningOptimizerConfig:
        return configure_lightning_optimizers(self.model, self.train_config)

    def _step(self, batch: CausalLMBatch, *, stage: str | None) -> Tensor:
        semantic_batch = self._semantic_batch(batch)
        output = self._semantic_output(semantic_batch)
        row_loss = self._semantic_row_loss(semantic_batch, output) if output is not None else None
        token_counts = (
            loss_token_counts(semantic_batch, dtype=row_loss.dtype)
            if row_loss is not None
            else None
        )
        loss = self._loss(
            batch,
            row_loss,
            token_counts,
            stage=stage,
            hidden_states=_last_hidden_state(output) if output is not None else None,
        )
        if output is not None and row_loss is not None and token_counts is not None:
            self._log_semantic_accuracy(semantic_batch, output, token_counts, stage=stage)
            self._log_task_metrics(semantic_batch, row_loss, token_counts, stage=stage)
            if stage is None:
                log_reduced_sum(
                    self,
                    "supervised_tokens",
                    token_counts.sum(),
                    on_step=True,
                    on_epoch=False,
                )
        self._log_acoustic_frame_count(batch, stage=stage)
        return loss

    def _loss(
        self,
        batch: CausalLMBatch,
        row_loss: Tensor | None = None,
        token_counts: Tensor | None = None,
        *,
        stage: str | None = None,
        hidden_states: Tensor | None = None,
    ) -> Tensor:
        semantic_weight = self.train_config.semantic_loss_weight
        acoustic_weight = self.train_config.acoustic_loss_weight
        if semantic_weight <= 0.0 and acoustic_weight <= 0.0:
            raise ValueError("at least one loss weight must be positive.")

        component: _LossComponent | None = None
        if semantic_weight > 0.0:
            semantic_batch = self._semantic_batch(batch)
            if row_loss is None:
                row_loss = self._semantic_row_loss(semantic_batch)
            if token_counts is None:
                token_counts = loss_token_counts(semantic_batch, dtype=row_loss.dtype)
            semantic_mean = reduced_weighted_mean(row_loss, token_counts)
            component = _combine_loss_components(
                component,
                _weighted_loss_component(semantic_mean, semantic_weight),
            )

        acoustic_weight = self.train_config.acoustic_loss_weight
        if acoustic_weight > 0.0:
            acoustic = self._acoustic_loss(
                batch,
                hidden_states=hidden_states,
                stage=stage,
            )
            acoustic_mean = acoustic_reduced_mean(batch, acoustic)
            component = _combine_loss_components(
                component,
                _weighted_loss_component(acoustic_mean, acoustic_weight),
            )
            log_reduced_mean(
                self,
                _log_name("loss/acoustic", stage=stage),
                acoustic_mean,
                on_step=stage is None,
                on_epoch=True,
            )
            if isinstance(acoustic, AcousticFlowLossStats):
                self._log_acoustic_t_bin_losses(acoustic, stage=stage)
        if component is None:
            raise RuntimeError("loss weights did not produce a training objective.")
        log_reduced_value(
            self,
            _log_name("loss", stage=stage),
            component.value,
            component.weight,
            on_step=stage is None,
            on_epoch=True,
            prog_bar=True,
        )
        if stage is None:
            return component.objective
        return component.value

    def _semantic_output(self, batch: CausalLMBatch) -> CausalLMOutputWithPast | None:
        if self.train_config.semantic_loss_weight <= 0.0:
            return None
        return self.model(
            batch,
            return_hidden_states=self.train_config.acoustic_loss_weight > 0.0,
        )

    def _semantic_batch(self, batch: CausalLMBatch) -> CausalLMBatch:
        stop_weight = self.train_config.stop_loss_weight
        if stop_weight == 1.0:
            return batch
        idspace = getattr(self.model, "idspace", None)
        if idspace is None:
            raise TypeError("stop_loss_weight requires model.idspace.")
        return semantic_batch(
            batch,
            idspace=idspace,
            stop_loss_weight=stop_weight,
        )

    def _semantic_row_loss(
        self,
        batch: CausalLMBatch,
        output: CausalLMOutputWithPast | None = None,
    ) -> Tensor:
        if output is None:
            output = self.model(batch)
        return semantic_row_loss(batch, output)

    def _log_semantic_accuracy(
        self,
        batch: CausalLMBatch,
        output: CausalLMOutputWithPast,
        token_counts: Tensor,
        *,
        stage: str | None,
    ) -> None:
        accuracy = self._semantic_accuracy(batch, output).reshape(())
        mean = reduced_weighted_mean(
            accuracy.detach(),
            token_counts.sum().detach().reshape(()),
        )
        log_reduced_mean(
            self,
            _log_name("accuracy", stage=stage),
            mean,
            on_step=stage is None,
            on_epoch=True,
            prog_bar=True,
        )

    def _semantic_accuracy(
        self,
        batch: CausalLMBatch,
        output: CausalLMOutputWithPast,
    ) -> Tensor:
        accuracy = self.model.semantic_accuracy(batch, output)
        if not isinstance(accuracy, Tensor):
            raise TypeError("model semantic_accuracy() must return a Tensor.")
        return accuracy

    def _log_task_metrics(
        self,
        batch: CausalLMBatch,
        row_loss: Tensor,
        token_counts: Tensor,
        *,
        stage: str | None,
    ) -> None:
        log_task_losses(
            self,
            batch,
            row_loss,
            token_counts,
            name_prefix=_log_prefix(stage),
            on_step=stage is None,
        )
        log_task_group_losses(
            self,
            batch,
            row_loss,
            token_counts,
            name_prefix=_log_prefix(stage),
            on_step=stage is None,
        )

    def _log_acoustic_frame_count(self, batch: CausalLMBatch, *, stage: str | None) -> None:
        if batch.target_audio is None:
            return
        frames = batch.target_audio.acoustic_mask.sum().float()
        log_reduced_sum(
            self,
            _log_name("acoustic_frames", stage=stage),
            frames,
            on_step=stage is None,
            on_epoch=stage is not None,
        )

    def _acoustic_loss(
        self,
        batch: CausalLMBatch,
        *,
        hidden_states: Tensor | None = None,
        stage: str | None = None,
    ) -> Tensor | AcousticFlowLossStats:
        if self.bpe is None:
            raise RuntimeError("acoustic loss requires a LongCat BPE tokenizer.")
        if self.acoustic_feature_extractor is None:
            raise RuntimeError("acoustic loss requires an acoustic feature extractor.")
        if batch.target_audio is None:
            raise RuntimeError("acoustic loss requires target_audio in the batch.")
        inputs, stats = acoustic_loss_stats(
            self.model,
            batch,
            bpe=self.bpe,
            acoustic_feature_extractor=self.acoustic_feature_extractor,
            hidden_states=hidden_states,
        )
        self._log_acoustic_condition_stats(
            inputs,
            timesteps=stats.timesteps,
            stage=stage,
        )
        return stats

    def _log_acoustic_t_bin_losses(
        self,
        stats: AcousticFlowLossStats,
        *,
        stage: str | None,
    ) -> None:
        for name, mean in acoustic_t_bin_means(stats):
            log_reduced_mean(
                self,
                _log_name(f"loss/acoustic_t/{name}", stage=stage),
                mean,
                on_step=stage is None,
                on_epoch=True,
            )

    def _log_acoustic_condition_stats(
        self,
        inputs: AcousticFlowInputs,
        *,
        timesteps: Tensor,
        stage: str | None,
    ) -> None:
        for metric in acoustic_condition_metrics(self.model, inputs, timesteps=timesteps):
            self._log_tensor_stats(
                _log_name(f"condition/{metric.name}", stage=stage),
                metric.stats,
                on_step=stage is None,
                on_epoch=True,
            )

    def _log_tensor_stats(
        self,
        prefix: str,
        stats: TensorStats,
        *,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        log_reduced_value(
            self,
            f"{prefix}_mean",
            stats.mean,
            stats.weight,
            on_step=on_step,
            on_epoch=on_epoch,
        )
        log_reduced_value(
            self,
            f"{prefix}_std",
            stats.std,
            stats.weight,
            on_step=on_step,
            on_epoch=on_epoch,
        )


def _validate_train_config(train: TrainConfig) -> None:
    if not isinstance(train.stop_loss_weight, int | float) or isinstance(
        train.stop_loss_weight,
        bool,
    ):
        raise TypeError("train.stop_loss_weight must be a number.")
    if not isfinite(float(train.stop_loss_weight)) or train.stop_loss_weight <= 0.0:
        raise ValueError("train.stop_loss_weight must be finite and positive.")

def _log_name(name: str, *, stage: str | None) -> str:
    return f"{_log_prefix(stage)}{name}"


def _log_prefix(stage: str | None) -> str:
    if stage is None:
        return ""
    return f"{stage}/"


def _weighted_loss_component(mean: ReducedMean, weight: float) -> _LossComponent:
    return _LossComponent(
        objective=weight * scaled_loss(mean),
        value=weight * mean.value,
        weight=mean.weight,
    )


def _combine_loss_components(
    left: _LossComponent | None,
    right: _LossComponent,
) -> _LossComponent:
    if left is None:
        return right
    return _LossComponent(
        objective=left.objective + right.objective,
        value=left.value + right.value,
        weight=left.weight,
    )


def _last_hidden_state(output: CausalLMOutputWithPast) -> Tensor | None:
    hidden_states = output.hidden_states
    if hidden_states is None:
        return None
    if len(hidden_states) == 0:
        raise RuntimeError("model output hidden_states must not be empty.")
    return hidden_states[-1]
