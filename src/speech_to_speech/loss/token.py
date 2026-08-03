from __future__ import annotations

from collections.abc import Callable, Set

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout
from torch import Tensor, nn

from .._tensor import is_signed_integer_dtype
from ..prediction import PredictionModality
from .types import LossItem


class TokenLoss(nn.Module):
    def __init__(
        self,
        layout: Layout,
    ) -> None:
        super().__init__()
        self.layout = layout

    def forward(
        self,
        hidden_states: Tensor,
        token_labels: Tensor,
        prediction: PredictionModality | Modality,
        token_logits: Callable[..., Tensor],
        *,
        audio_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> LossItem:
        if hidden_states.dim() != 3 or token_labels.dim() != 2:
            raise ValueError(
                "token hidden states and labels must have shapes [B, T, H] and [B, T]."
            )
        if hidden_states.shape[:2] != token_labels.shape:
            raise ValueError("token hidden states and labels must align on sequence.")
        if not is_signed_integer_dtype(token_labels.dtype):
            raise TypeError("token labels must use a signed integer dtype.")
        modalities = _modalities(prediction)
        target = token_labels[:, 1:]
        prediction_states = hidden_states[:, :-1]

        valid = target.ne(-100)
        modality_mask = torch.zeros_like(valid)
        for modality in modalities:
            start, end = self.layout.blocks[modality.value]
            modality_mask |= target.ge(start) & target.lt(end)
        invalid = torch.stack(
            (
                (valid & ~modality_mask).any(),
                ~valid.any(dim=1).all(),
            )
        )
        if bool(invalid.any()):
            if bool(invalid[0]):
                names = ", ".join(sorted(modality.value for modality in modalities))
                raise ValueError(
                    f"labels contain an id outside the supervised layout blocks: {names}."
                )
            raise ValueError(
                "each token label row must contain at least one target token."
            )
        selected_loss = self._loss(
            prediction_states,
            target,
            valid,
            modalities,
            token_logits,
            audio_hidden_states=(
                None
                if audio_hidden_states is None
                else audio_hidden_states[:, :-1]
            ),
            attention_mask=(
                None if attention_mask is None else attention_mask[:, :-1]
            ),
        )
        token_loss = selected_loss.new_zeros(target.shape)
        token_loss[valid] = selected_loss
        text_start, text_end = self.layout.blocks[Modality.TEXT.value]
        audio_start, audio_end = self.layout.blocks[Modality.AUDIO.value]
        text_mask = valid & target.ge(text_start) & target.lt(text_end)
        audio_mask = valid & target.ge(audio_start) & target.lt(audio_end)
        text_count = text_mask.sum(dim=1)
        audio_count = audio_mask.sum(dim=1)
        text_loss = (token_loss * text_mask).sum(dim=1) / text_count.clamp_min(1)
        audio_loss = (token_loss * audio_mask).sum(dim=1) / audio_count.clamp_min(1)
        total_count = text_count + audio_count
        total_loss = (token_loss * valid).sum(dim=1) / total_count.clamp_min(1)
        return LossItem(
            loss=total_loss,
            details={
                "text_loss": text_loss,
                "audio_loss": audio_loss,
                "tokens": total_count.to(dtype=hidden_states.dtype),
                "text_tokens": text_count.to(dtype=hidden_states.dtype),
                "audio_tokens": audio_count.to(dtype=hidden_states.dtype),
            },
        )

    def _loss(
        self,
        prediction_states: Tensor,
        target: Tensor,
        valid: Tensor,
        modalities: Set[Modality],
        token_logits: Callable[..., Tensor],
        *,
        audio_hidden_states: Tensor | None,
        attention_mask: Tensor | None,
    ) -> Tensor:
        losses = prediction_states.new_empty(int(valid.sum().item()))
        flat_offsets = valid.flatten().nonzero(as_tuple=False).flatten()
        loss_offsets = torch.empty_like(valid, dtype=torch.long)
        loss_offsets.flatten()[flat_offsets] = torch.arange(
            flat_offsets.numel(),
            device=target.device,
        )
        for modality in sorted(modalities, key=lambda value: value.value):
            start, end = self.layout.blocks[modality.value]
            mask = valid & target.ge(start) & target.lt(end)
            if not bool(mask.any()):
                continue
            selected_target = (target[mask] - start).to(dtype=torch.long)
            if modality is Modality.AUDIO and audio_hidden_states is not None:
                logits = token_logits(
                    prediction_states[mask],
                    modality,
                    audio_hidden_state=audio_hidden_states[mask],
                )
            else:
                logits = token_logits(prediction_states[mask], modality)
            if logits.shape != (selected_target.numel(), end - start):
                raise ValueError(
                    "token logits do not match selected targets and modality vocabulary."
                )
            group_loss = nn.functional.cross_entropy(
                logits,
                selected_target,
                reduction="none",
            )
            losses[loss_offsets[mask]] = group_loss.to(dtype=losses.dtype)
        return losses


def _modalities(prediction: PredictionModality | Modality) -> frozenset[Modality]:
    if isinstance(prediction, PredictionModality):
        modalities = prediction.supervised_modalities()
        if not modalities:
            raise ValueError(f"prediction modality {prediction.value} has no heads.")
        return modalities
    if prediction not in {Modality.TEXT, Modality.AUDIO}:
        raise ValueError(f"unsupported token modality: {prediction.value}")
    return frozenset({prediction})
