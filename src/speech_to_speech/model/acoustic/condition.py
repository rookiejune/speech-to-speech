"""Prepare acoustic conditions and continuous-flow losses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import torch
from anytrain.idspace import IdSpaceEmbedding, Modality
from anytrain.tokenizer import CodecBPE
from torch import Tensor, nn

from ...datamodule.types import (
    CausalLMBatch,
    IGNORE_INDEX,
    LongCatBatchSide,
)
from ..types import AcousticCondition, AudioBoundary


class _TimeSampler(Protocol):
    def sample(self, batch_size: int, device: torch.device) -> Tensor: ...


@dataclass(frozen=True)
class AcousticFlowLossStats:
    loss: Tensor
    timesteps: Tensor
    row_loss: Tensor
    row_weight: Tensor


class _FixedTimeSampler:
    def __init__(self, timesteps: Tensor) -> None:
        self.timesteps = timesteps

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        if self.timesteps.dim() != 1 or self.timesteps.numel() != batch_size:
            raise ValueError("timesteps must have shape (batch,).")
        return self.timesteps.to(device=device)


class _RecordingTimeSampler:
    def __init__(self, sampler: _TimeSampler) -> None:
        self.sampler = sampler
        self.timesteps: Tensor | None = None

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        timesteps = self.sampler.sample(batch_size, device)
        self.timesteps = timesteps
        return timesteps


class _FlowModel(nn.Module):
    def __init__(self, dit: nn.Module) -> None:
        super().__init__()
        self.dit = dit

    def forward(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        *,
        last_hidden_state: Tensor,
        acoustic_condition: Tensor,
        mask: Tensor,
    ) -> Tensor:
        outputs = self.dit(
            x_t=x_t,
            last_hidden_state=last_hidden_state,
            timesteps=timesteps,
            acoustic_condition=acoustic_condition,
            attention_mask=mask.to(device=x_t.device, dtype=torch.long),
        )
        prediction = outputs.last_hidden_state
        if prediction.shape != x_t.shape:
            raise ValueError("DiT output and acoustic velocity target must have the same shape.")
        return prediction


def acoustic_velocity(
    dit: nn.Module,
    *,
    x_t: Tensor,
    timesteps: Tensor,
    last_hidden_state: Tensor,
    acoustic_condition: Tensor,
    mask: Tensor,
    position_ids: Tensor | None = None,
    guidance_scale: float = 1.0,
) -> Tensor:
    _validate_guidance_scale(guidance_scale)
    conditional = _dit_velocity(
        dit,
        x_t=x_t,
        timesteps=timesteps,
        last_hidden_state=last_hidden_state,
        acoustic_condition=acoustic_condition,
        mask=mask,
        position_ids=position_ids,
    )
    if guidance_scale == 1.0:
        return conditional
    unconditional = _dit_velocity(
        dit,
        x_t=x_t,
        timesteps=timesteps,
        last_hidden_state=last_hidden_state,
        acoustic_condition=null_acoustic_condition(dit, x_t),
        mask=mask,
        position_ids=position_ids,
    )
    return unconditional + guidance_scale * (conditional - unconditional)


def acoustic_condition(
    *,
    batch: CausalLMBatch,
    hidden_states: Tensor,
    embedding: IdSpaceEmbedding,
    bpe: CodecBPE,
) -> AcousticCondition:
    labels = batch.labels
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must have shape (batch, sequence, dim).")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden_states and labels must align on batch and sequence dimensions.")

    bpe_mask, local_ids = _target_bpe_labels(
        batch,
        embedding=embedding,
        require_following_input=True,
    )
    shifted_hidden = hidden_states.new_zeros(hidden_states.shape)
    shifted_hidden[:, :-1] = hidden_states[:, 1:]

    return _expanded_condition(
        shifted_hidden,
        bpe_mask,
        local_ids,
        bpe=bpe,
    )


def acoustic_condition_from_target_audio_embedding(
    *,
    batch: CausalLMBatch,
    embedding: IdSpaceEmbedding,
    bpe: CodecBPE,
) -> AcousticCondition:
    bpe_mask, local_ids = _target_bpe_labels(
        batch,
        embedding=embedding,
        require_following_input=False,
    )
    audio_ids = local_ids.clamp_min(0)
    hidden = embedding.modality_embeddings[Modality.AUDIO.value](audio_ids)
    hidden = hidden * bpe_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    return _expanded_condition(
        hidden,
        bpe_mask,
        local_ids,
        bpe=bpe,
    )


def _target_bpe_labels(
    batch: CausalLMBatch,
    *,
    embedding: IdSpaceEmbedding,
    require_following_input: bool,
) -> tuple[Tensor, Tensor]:
    labels = batch.labels
    block = embedding.space.modality_block(Modality.AUDIO)
    boa_global_id = embedding.space.special_token_id(AudioBoundary.BOA)
    eoa_global_id = embedding.space.special_token_id(AudioBoundary.EOA)

    active_mask = labels.ne(IGNORE_INDEX)
    audio_mask = (
        (labels.ge(block.start) & labels.lt(block.end))
        | labels.eq(boa_global_id)
        | labels.eq(eoa_global_id)
    )
    if bool((active_mask & ~audio_mask).any()):
        bad_id = int(labels[active_mask & ~audio_mask].reshape(-1)[0].detach().cpu())
        raise ValueError(f"labels contain non-audio target token: {bad_id}.")

    bpe_mask = active_mask & labels.ne(boa_global_id) & labels.ne(eoa_global_id)
    if require_following_input:
        if bool(bpe_mask[:, -1].any()):
            raise ValueError("target BPE labels must have a following input position.")

        shifted_input_ids = batch.input_ids[:, 1:]
        shifted_attention = batch.attention_mask[:, 1:]
        bpe_mask_without_last = bpe_mask[:, :-1]
        labels_without_last = labels[:, :-1]
        if bool((bpe_mask_without_last & shifted_input_ids.ne(labels_without_last)).any()):
            raise ValueError("target BPE labels must match the following input token.")
        if bool((bpe_mask_without_last & shifted_attention.eq(0)).any()):
            raise ValueError("target BPE labels must shift to non-padding input tokens.")

    local_ids = torch.zeros_like(labels)
    local_ids[bpe_mask] = labels[bpe_mask] - block.start
    return bpe_mask, local_ids


def _expanded_condition(
    hidden_states: Tensor,
    bpe_mask: Tensor,
    local_ids: Tensor,
    *,
    bpe: CodecBPE,
) -> AcousticCondition:
    expanded = bpe.repeat_interleave(
        hidden_states,
        local_ids,
        bpe_mask,
        dim=1,
    )
    if len(expanded) != 3:
        raise ValueError("batched acoustic condition expansion must return padded values.")
    expanded_hidden, semantic_frames, mask = expanded
    return AcousticCondition(
        hidden_states=expanded_hidden,
        semantic_ids=_single_codebook_ids(semantic_frames),
        mask=mask,
        chunk_lengths=_chunk_lengths_from_bpe_mask(
            bpe_mask,
            local_ids,
            bpe=bpe,
        ),
    )


def validate_acoustic_features(
    target_features: Tensor,
    condition_mask: Tensor,
    *,
    target_mask: Tensor | None,
) -> None:
    if target_features.dim() != 3:
        raise ValueError("target_features must have shape (batch, time, dim).")
    if target_features.shape[:2] != condition_mask.shape:
        raise ValueError("target_features and acoustic condition must align on batch and time.")
    if target_mask is None:
        return
    if target_mask.shape != condition_mask.shape:
        raise ValueError("target_mask must have the same shape as acoustic condition mask.")
    if not torch.equal(
        target_mask.to(device=condition_mask.device, dtype=torch.bool),
        condition_mask,
    ):
        raise ValueError("target_mask must match acoustic condition mask.")


def acoustic_features_from_batch_side(
    side: LongCatBatchSide,
    *,
    feature_extractor: object,
) -> tuple[Tensor, Tensor]:
    convert = getattr(feature_extractor, "acoustic_codes_to_features", None)
    if not callable(convert):
        raise TypeError("feature_extractor must provide acoustic_codes_to_features().")
    acoustic_ids = side.acoustic_ids
    if acoustic_ids.dim() != 3:
        raise ValueError("LongCat batch acoustic_ids must have shape [batch, nq, time].")
    features = convert(acoustic_ids)
    if not isinstance(features, Tensor):
        raise TypeError("acoustic_codes_to_features() must return a Tensor.")
    if features.dim() != 3:
        raise ValueError("LongCat acoustic features must have shape [batch, time, dim].")
    if features.shape[:2] != side.acoustic_mask.shape:
        raise ValueError("LongCat acoustic features must align with acoustic_mask.")
    if not torch.is_floating_point(features) or torch.is_complex(features):
        raise TypeError("LongCat acoustic features must be floating point tensors.")
    return features, side.acoustic_mask.to(device=features.device, dtype=torch.bool)


def pooled_acoustic_condition_from_batch_side(
    side: LongCatBatchSide,
    *,
    feature_extractor: object,
    empty_condition: Tensor | None = None,
) -> Tensor:
    features, mask = acoustic_features_from_batch_side(
        side,
        feature_extractor=feature_extractor,
    )
    weights = mask.to(device=features.device, dtype=features.dtype).unsqueeze(-1)
    frame_counts = weights.sum(dim=1)
    pooled = (features * weights).sum(dim=1) / frame_counts.clamp_min(1.0)
    empty_rows = frame_counts.eq(0)
    if bool(empty_rows.any()):
        if empty_condition is None:
            raise ValueError("source acoustic condition rows must contain at least one frame.")
        fallback = _expand_acoustic_condition(
            empty_condition,
            batch_size=features.size(0),
            hidden_size=features.size(-1),
            device=features.device,
            dtype=features.dtype,
        )
        pooled = torch.where(empty_rows, fallback, pooled)
    return pooled


def null_acoustic_condition(dit: nn.Module, like: Tensor) -> Tensor:
    value = getattr(dit, "null_acoustic_condition", None)
    if isinstance(value, Tensor):
        return value.to(device=like.device, dtype=like.dtype).expand(like.size(0), -1)
    hidden_size = like.size(-1)
    return like.new_zeros((like.size(0), hidden_size))


def continuous_flow_loss(
    dit: nn.Module,
    target_features: Tensor,
    *,
    x_0: Tensor,
    timesteps: Tensor | None,
    last_hidden_state: Tensor,
    acoustic_condition: Tensor,
    mask: Tensor,
) -> Tensor:
    return continuous_flow_loss_stats(
        dit,
        target_features,
        x_0=x_0,
        timesteps=timesteps,
        last_hidden_state=last_hidden_state,
        acoustic_condition=acoustic_condition,
        mask=mask,
    ).loss


def continuous_flow_loss_stats(
    dit: nn.Module,
    target_features: Tensor,
    *,
    x_0: Tensor,
    timesteps: Tensor | None,
    last_hidden_state: Tensor,
    acoustic_condition: Tensor,
    mask: Tensor,
) -> AcousticFlowLossStats:
    from anytrain.framework.flow_matching import ContinuousFlowMatcher, LogitNormalTimeSampler

    if x_0.shape != target_features.shape:
        raise ValueError("x_0 and target_features must have the same shape.")
    if timesteps is None:
        time_sampler = _RecordingTimeSampler(LogitNormalTimeSampler())
    else:
        timesteps = timesteps.to(device=target_features.device, dtype=target_features.dtype)
        time_sampler = _RecordingTimeSampler(_FixedTimeSampler(timesteps))

    holder: dict[str, Tensor] = {}

    def loss_fn(prediction: Tensor, target: Tensor, extras: object) -> Tensor:
        if not isinstance(extras, Mapping):
            raise TypeError("masked acoustic flow loss requires model extras.")
        stats = _masked_mse_stats(
            prediction,
            target,
            mask=mask,
            timesteps=_recorded_timesteps(time_sampler),
        )
        holder["row_loss"] = stats.row_loss
        holder["row_weight"] = stats.row_weight
        holder["timesteps"] = stats.timesteps
        return stats.loss

    matcher = ContinuousFlowMatcher(
        time_sampler=time_sampler,
        loss_fn=loss_fn,
    )
    loss = matcher.loss(
        _FlowModel(dit),
        target_features,
        x_0=x_0,
        last_hidden_state=last_hidden_state,
        acoustic_condition=acoustic_condition,
        mask=mask,
    )
    return AcousticFlowLossStats(
        loss=loss,
        timesteps=holder["timesteps"],
        row_loss=holder["row_loss"],
        row_weight=holder["row_weight"],
    )


def _masked_mse(prediction: Tensor, target: Tensor, extras: object) -> Tensor:
    if not isinstance(extras, Mapping):
        raise TypeError("masked acoustic flow loss requires model extras.")
    mask = extras["mask"]
    if not isinstance(mask, Tensor):
        raise TypeError("masked acoustic flow loss requires a tensor mask.")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    if prediction.shape[:2] != mask.shape:
        raise ValueError("prediction and mask must align on batch and time.")
    loss = (prediction - target).square()
    weights = mask.to(device=prediction.device, dtype=prediction.dtype).unsqueeze(-1)
    denominator = weights.sum() * prediction.size(-1)
    if denominator <= 0:
        raise ValueError("acoustic mask must contain at least one valid frame.")
    return (loss * weights).sum() / denominator


def _masked_mse_stats(
    prediction: Tensor,
    target: Tensor,
    *,
    mask: Tensor,
    timesteps: Tensor,
) -> AcousticFlowLossStats:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")
    if prediction.shape[:2] != mask.shape:
        raise ValueError("prediction and mask must align on batch and time.")
    if timesteps.dim() != 1 or timesteps.numel() != prediction.size(0):
        raise ValueError("timesteps must have shape (batch,).")
    loss = (prediction - target).square()
    weights = mask.to(device=prediction.device, dtype=prediction.dtype).unsqueeze(-1)
    row_sum = (loss * weights).sum(dim=(1, 2))
    row_weight = weights.sum(dim=(1, 2)) * prediction.size(-1)
    total_weight = row_weight.sum()
    if total_weight <= 0:
        raise ValueError("acoustic mask must contain at least one valid frame.")
    row_loss = torch.zeros_like(row_sum)
    nonempty = row_weight.gt(0)
    row_loss[nonempty] = row_sum[nonempty] / row_weight[nonempty]
    return AcousticFlowLossStats(
        loss=row_sum.sum() / total_weight,
        timesteps=timesteps,
        row_loss=row_loss,
        row_weight=row_weight,
    )


def _recorded_timesteps(time_sampler: _RecordingTimeSampler) -> Tensor:
    if time_sampler.timesteps is None:
        raise RuntimeError("flow matching time sampler did not record timesteps.")
    return time_sampler.timesteps


def _expand_acoustic_condition(
    condition: Tensor,
    *,
    batch_size: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    condition = condition.to(device=device, dtype=dtype)
    if condition.dim() == 1:
        condition = condition.unsqueeze(0)
    if condition.shape == (1, hidden_size):
        condition = condition.expand(batch_size, -1)
    if condition.shape != (batch_size, hidden_size):
        raise ValueError("empty_condition must have shape [hidden] or [batch, hidden].")
    return condition


def _single_codebook_ids(frames: Tensor) -> Tensor:
    if frames.dim() != 3 or frames.size(-1) != 1:
        raise ValueError("LongCat semantic BPE must expand to [batch, time, 1] frames.")
    return frames.squeeze(-1)


def _chunk_lengths_from_bpe_mask(
    mask: Tensor,
    local_ids: Tensor,
    *,
    bpe: CodecBPE,
) -> tuple[tuple[int, ...], ...]:
    lengths: list[tuple[int, ...]] = []
    for row_mask, row_ids in zip(mask.detach().cpu(), local_ids.detach().cpu(), strict=True):
        row_lengths: list[int] = []
        for token_id in row_ids[row_mask].tolist():
            expanded = bpe.expand_ids([int(token_id)])
            row_lengths.append(len(expanded))
        lengths.append(tuple(row_lengths))
    return tuple(lengths)


def _dit_velocity(
    dit: nn.Module,
    *,
    x_t: Tensor,
    timesteps: Tensor,
    last_hidden_state: Tensor,
    acoustic_condition: Tensor,
    mask: Tensor,
    position_ids: Tensor | None,
) -> Tensor:
    kwargs: dict[str, Tensor] = {
        "x_t": x_t,
        "last_hidden_state": last_hidden_state,
        "timesteps": timesteps,
        "acoustic_condition": acoustic_condition,
        "attention_mask": mask.to(device=x_t.device, dtype=torch.long),
    }
    if position_ids is not None:
        kwargs["position_ids"] = position_ids.to(device=x_t.device, dtype=torch.long)
    outputs = dit(**kwargs)
    prediction = outputs.last_hidden_state
    if prediction.shape != x_t.shape:
        raise ValueError("DiT output and acoustic velocity target must have the same shape.")
    return prediction


def _validate_guidance_scale(guidance_scale: float) -> None:
    if isinstance(guidance_scale, bool) or not isinstance(guidance_scale, int | float):
        raise TypeError("guidance_scale must be a number.")
    if guidance_scale < 0.0:
        raise ValueError("guidance_scale must be non-negative.")
