from __future__ import annotations

from collections.abc import Mapping

import torch
from anytrain.idspace import IdSpaceEmbedding, Modality
from anytrain.tokenizer import IntBPE
from torch import Tensor, nn

from ..types import AcousticCondition, AudioBoundary, CausalLMBatch, IGNORE_INDEX


class _FixedTimeSampler:
    def __init__(self, timesteps: Tensor) -> None:
        self.timesteps = timesteps

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        if self.timesteps.dim() != 1 or self.timesteps.numel() != batch_size:
            raise ValueError("timesteps must have shape (batch,).")
        return self.timesteps.to(device=device)


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


def acoustic_condition(
    *,
    batch: CausalLMBatch,
    hidden_states: Tensor,
    embedding: IdSpaceEmbedding,
    bpe: IntBPE,
) -> AcousticCondition:
    labels = batch.labels
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must have shape (batch, sequence, dim).")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("hidden_states and labels must align on batch and sequence dimensions.")

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
    shifted_hidden = hidden_states.new_zeros(hidden_states.shape)
    shifted_hidden[:, :-1] = hidden_states[:, 1:]

    expanded = bpe.repeat_interleave(
        shifted_hidden,
        local_ids,
        bpe_mask,
        dim=1,
    )
    if len(expanded) != 3:
        raise ValueError("batched acoustic condition expansion must return padded values.")
    expanded_hidden, semantic_ids, mask = expanded
    return AcousticCondition(
        hidden_states=expanded_hidden,
        semantic_ids=semantic_ids,
        mask=mask,
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
    from anytrain.framework.flow_matching import ContinuousFlowMatcher

    time_sampler = None
    if timesteps is not None:
        timesteps = timesteps.to(device=target_features.device, dtype=target_features.dtype)
        time_sampler = _FixedTimeSampler(timesteps)

    matcher = ContinuousFlowMatcher(
        time_sampler=time_sampler,
        loss_fn=_masked_mse,
    )
    return matcher.loss(
        _FlowModel(dit),
        target_features,
        x_0=x_0,
        last_hidden_state=last_hidden_state,
        acoustic_condition=acoustic_condition,
        mask=mask,
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
