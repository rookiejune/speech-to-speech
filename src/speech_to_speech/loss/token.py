from __future__ import annotations

from collections.abc import Callable

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout
from torch import Tensor, nn

from .._tensor import is_signed_integer_dtype
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
        modality: Modality,
        token_logits: Callable[[Tensor, Modality], Tensor],
        *,
        token_groups: Tensor | None = None,
        selected_logits: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> LossItem:
        if hidden_states.dim() != 3 or token_labels.dim() != 2:
            raise ValueError(
                "token hidden states and labels must have shapes [B, T, H] and [B, T]."
            )
        if hidden_states.shape[:2] != token_labels.shape:
            raise ValueError("token hidden states and labels must align on sequence.")
        if not is_signed_integer_dtype(token_labels.dtype):
            raise TypeError("token labels must use a signed integer dtype.")
        target = token_labels[:, 1:]
        prediction = hidden_states[:, :-1]

        valid = target.ne(-100)
        start, end = self.layout.blocks[modality.value]
        modality_mask = target.ge(start) & target.lt(end)
        invalid = torch.stack(
            (
                (valid & ~modality_mask).any(),
                ~valid.any(dim=1).all(),
            )
        )
        if bool(invalid.any()):
            if bool(invalid[0]):
                raise ValueError(
                    f"labels contain an id outside the {modality.value} layout block."
                )
            raise ValueError(
                "each token label row must contain at least one target token."
            )
        if token_groups is None:
            selected_target = (target[valid] - start).to(dtype=torch.long)
            logits = token_logits(prediction[valid], modality)
            if logits.shape != (selected_target.numel(), end - start):
                raise ValueError(
                    "token logits do not match selected targets and modality vocabulary."
                )
            selected_loss = nn.functional.cross_entropy(
                logits,
                selected_target,
                reduction="none",
            )
        else:
            selected_loss = self._grouped_loss(
                prediction,
                target,
                valid,
                token_groups[:, 1:],
                modality,
                selected_logits,
            )
        token_loss = selected_loss.new_zeros(target.shape)
        token_loss[valid] = selected_loss
        text_mask = valid if modality is Modality.TEXT else valid & False
        audio_mask = valid if modality is Modality.AUDIO else valid & False
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

    def _grouped_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        valid: Tensor,
        groups: Tensor,
        modality: Modality,
        selected_logits: Callable[[Tensor, Tensor], Tensor] | None,
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
            logits = selected_logits(prediction[mask], allowed)
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
