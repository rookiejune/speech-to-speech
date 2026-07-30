from __future__ import annotations

from collections.abc import Callable, Set

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout
from torch import Tensor, nn

from .._tensor import is_signed_integer_dtype
from ..prediction import PredictionModality
from ..runtime.audio_tokenizer import BiCodecAudioTokenizer
from ..runtime.types import AudioTokenizer
from .types import LossItem


class TokenLoss(nn.Module):
    def __init__(
        self,
        layout: Layout,
        audio_tokenizer: AudioTokenizer | None = None,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.audio_tokenizer = audio_tokenizer

    def forward(
        self,
        hidden_states: Tensor,
        token_labels: Tensor,
        prediction: PredictionModality | Modality,
        token_logits: Callable[..., Tensor],
        *,
        token_groups: Tensor | None = None,
        selected_logits: Callable[..., Tensor] | None = None,
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
        if token_groups is None:
            selected_loss = self._ungrouped_loss(
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
        else:
            if modalities != frozenset({Modality.AUDIO}):
                raise ValueError(
                    "token prediction groups are supported only for audio-only targets."
                )
            selected_loss = self._grouped_loss(
                prediction_states if audio_hidden_states is None else audio_hidden_states[:, :-1],
                target,
                valid,
                token_groups[:, 1:],
                Modality.AUDIO,
                selected_logits,
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

    def _ungrouped_loss(
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

    def _grouped_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        valid: Tensor,
        groups: Tensor,
        modality: Modality,
        selected_logits: Callable[..., Tensor] | None,
    ) -> Tensor:
        if modality is not Modality.AUDIO:
            raise ValueError("token prediction groups are supported only for audio targets.")
        tokenizer = self.audio_tokenizer
        if not isinstance(tokenizer, BiCodecAudioTokenizer):
            raise TypeError("token prediction groups require BiCodecAudioTokenizer.")
        if selected_logits is None:
            raise TypeError("token prediction groups require selected token logits.")
        if groups.shape != target.shape:
            raise ValueError("token prediction groups must align with shifted labels.")
        if not is_signed_integer_dtype(groups.dtype):
            raise TypeError("token prediction groups must use a signed integer dtype.")
        if bool((valid & groups.lt(0)).any()) or bool((~valid & groups.ne(-1)).any()):
            raise ValueError("token prediction groups do not align with supervised labels.")

        audio_start, _ = self.layout.blocks[Modality.AUDIO.value]
        losses = prediction.new_empty(int(valid.sum().item()))
        flat_offsets = valid.flatten().nonzero(as_tuple=False).flatten()
        loss_offsets = torch.empty_like(valid, dtype=torch.long)
        loss_offsets.flatten()[flat_offsets] = torch.arange(
            flat_offsets.numel(),
            device=target.device,
        )
        for group_tensor in groups[valid].unique(sorted=True):
            group = int(group_tensor.item())
            mask = valid & groups.eq(group)
            local_allowed = tokenizer.prediction_ids(group, device=target.device)
            allowed = local_allowed + audio_start
            selected = selected_logits(prediction[mask], allowed)
            logits = selected[0] if isinstance(selected, tuple) else selected
            if logits.shape != (int(mask.sum().item()), allowed.numel()):
                raise ValueError(
                    "selected token logits do not match the BiCodec prediction group."
                )
            targets = target[mask]
            matches = targets[:, None].eq(allowed[None, :])
            if not bool(matches.any(dim=1).all()):
                raise ValueError(
                    f"BiCodec prediction group {group} does not contain every target."
                )
            group_loss = nn.functional.cross_entropy(
                logits,
                matches.to(dtype=torch.long).argmax(dim=1),
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
