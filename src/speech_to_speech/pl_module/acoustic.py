"""Acoustic loss preparation and metric helpers for Lightning training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch import device as TorchDevice

from ..types.datamodule import CausalLMBatch
from ..types.model import AcousticFeatureExtractor
from ..model.acoustic import AcousticFlowLossStats, acoustic_features_from_batch_side
from ..model.orchestrator import AcousticFlowInputs, DiTConditionTensors
from .metrics import ReducedMean, reduced_weighted_mean

ACOUSTIC_T_BINS = (
    ("0_025", 0.0, 0.25),
    ("025_050", 0.25, 0.5),
    ("050_075", 0.5, 0.75),
    ("075_100", 0.75, 1.0),
)


class AcousticModel(Protocol):
    def acoustic_flow_inputs(
        self,
        batch: CausalLMBatch,
        bpe: object,
        target_features: Tensor,
        *,
        hidden_states: Tensor | None = None,
        target_mask: Tensor | None = None,
        noise: Tensor | None = None,
        acoustic_condition: Tensor | None = None,
        source_feature_extractor: AcousticFeatureExtractor | None = None,
    ) -> AcousticFlowInputs: ...

    def acoustic_flow_loss_stats_from_inputs(
        self,
        inputs: AcousticFlowInputs,
        *,
        timesteps: Tensor | None = None,
    ) -> AcousticFlowLossStats: ...

    def acoustic_condition_tensors(
        self,
        inputs: AcousticFlowInputs,
        *,
        timesteps: Tensor,
    ) -> DiTConditionTensors: ...


@dataclass(frozen=True)
class TensorStats:
    mean: Tensor
    std: Tensor
    weight: Tensor


@dataclass(frozen=True)
class ConditionMetric:
    name: str
    stats: TensorStats


def acoustic_loss_stats(
    model: AcousticModel,
    batch: CausalLMBatch,
    *,
    bpe: object | None,
    acoustic_feature_extractor: AcousticFeatureExtractor | None,
    hidden_states: Tensor | None = None,
) -> tuple[AcousticFlowInputs, AcousticFlowLossStats]:
    if bpe is None:
        raise RuntimeError("acoustic loss requires a LongCat BPE tokenizer.")
    if acoustic_feature_extractor is None:
        raise RuntimeError("acoustic loss requires an acoustic feature extractor.")
    if batch.target_audio is None:
        raise RuntimeError("acoustic loss requires target_audio in the batch.")
    feature_extractor = feature_extractor_to_device(
        acoustic_feature_extractor,
        batch.target_audio.acoustic_ids.device,
    )
    target_features, target_mask = acoustic_features_from_batch_side(
        batch.target_audio,
        feature_extractor=feature_extractor,
    )
    inputs = model.acoustic_flow_inputs(
        batch,
        bpe,
        target_features,
        hidden_states=hidden_states,
        target_mask=target_mask,
        noise=None,
        acoustic_condition=None,
        source_feature_extractor=feature_extractor,
    )
    if not isinstance(inputs, AcousticFlowInputs):
        raise TypeError("model acoustic_flow_inputs() must return AcousticFlowInputs.")
    stats = model.acoustic_flow_loss_stats_from_inputs(inputs, timesteps=None)
    if not isinstance(stats, AcousticFlowLossStats):
        raise TypeError(
            "model acoustic_flow_loss_stats_from_inputs() must return AcousticFlowLossStats."
        )
    return inputs, stats


def acoustic_reduced_mean(
    batch: CausalLMBatch,
    acoustic: Tensor | AcousticFlowLossStats,
) -> ReducedMean:
    if isinstance(acoustic, AcousticFlowLossStats):
        return reduced_weighted_mean(acoustic.row_loss, acoustic.row_weight)
    if batch.target_audio is None:
        raise RuntimeError("acoustic loss requires target_audio in the batch.")
    weight = batch.target_audio.acoustic_mask.sum().to(
        device=acoustic.device,
        dtype=acoustic.dtype,
    )
    return reduced_weighted_mean(acoustic.reshape(()), weight.reshape(()))


def acoustic_t_bin_means(stats: AcousticFlowLossStats) -> tuple[tuple[str, ReducedMean], ...]:
    timesteps = stats.timesteps.detach()
    row_loss = stats.row_loss.detach()
    row_weight = stats.row_weight.detach()
    means: list[tuple[str, ReducedMean]] = []
    for name, start, end in ACOUSTIC_T_BINS:
        if end >= 1.0:
            mask = timesteps.ge(start) & timesteps.le(end)
        else:
            mask = timesteps.ge(start) & timesteps.lt(end)
        means.append((name, reduced_weighted_mean(row_loss[mask], row_weight[mask])))
    return tuple(means)


def acoustic_condition_metrics(
    model: AcousticModel,
    inputs: AcousticFlowInputs,
    *,
    timesteps: Tensor,
) -> tuple[ConditionMetric, ...]:
    tensors = model.acoustic_condition_tensors(inputs, timesteps=timesteps)
    if not isinstance(tensors, DiTConditionTensors):
        raise TypeError("model acoustic_condition_tensors() must return DiTConditionTensors.")
    mask = inputs.mask.to(device=inputs.last_hidden_state.device, dtype=torch.bool)
    frame_weights = mask.to(dtype=inputs.last_hidden_state.dtype)
    batch_weights = frame_weights.sum(dim=1).gt(0).to(dtype=inputs.last_hidden_state.dtype)
    return (
        ConditionMetric(
            "hidden",
            weighted_tensor_stats(tensors.hidden.detach(), frame_weights),
        ),
        ConditionMetric(
            "time",
            weighted_tensor_stats(tensors.time.detach().squeeze(1), batch_weights),
        ),
        ConditionMetric(
            "acoustic",
            weighted_tensor_stats(tensors.acoustic.detach().squeeze(1), batch_weights),
        ),
    )


def weighted_tensor_stats(values: Tensor, weights: Tensor) -> TensorStats:
    if values.dim() < 1:
        raise ValueError("condition stats values must have at least one dimension.")
    if weights.shape != values.shape[:-1]:
        raise ValueError("condition stats weights must match values except feature dimension.")
    weights = weights.to(device=values.device, dtype=values.dtype)
    expanded_weights = weights.unsqueeze(-1).expand_as(values)
    mean = reduced_weighted_mean(values.reshape(-1), expanded_weights.reshape(-1))
    centered = values - mean.value.to(device=values.device, dtype=values.dtype)
    variance = reduced_weighted_mean(
        centered.square().reshape(-1),
        expanded_weights.reshape(-1),
    )
    return TensorStats(
        mean=mean.value,
        std=variance.value.clamp_min(0.0).sqrt(),
        weight=mean.weight,
    )


def feature_extractor_to_device(
    extractor: AcousticFeatureExtractor,
    device: TorchDevice,
) -> AcousticFeatureExtractor:
    move = getattr(extractor, "to", None)
    if callable(move):
        moved = move(device)
        if moved is not None:
            extractor = moved
    if hasattr(extractor, "device"):
        setattr(extractor, "device", device)
    return extractor
