from collections.abc import Callable

import torch
from anydataset.types import Modality
from anytrain.idspace import Layout
from torch import Tensor, nn

from .._tensor import is_signed_integer_dtype
from .types import LossItem


class TokenLoss(nn.Module):
    def __init__(self, layout: Layout) -> None:
        super().__init__()
        self.layout = layout

    def forward(
        self,
        hidden_states: Tensor,
        token_labels: Tensor,
        modality: Modality,
        token_logits: Callable[[Tensor, Modality], Tensor],
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
        selected_target = (target[valid] - start).to(dtype=torch.long)
        selected_logits = token_logits(prediction[valid], modality)
        if selected_logits.shape != (selected_target.numel(), end - start):
            raise ValueError(
                "token logits do not match selected targets and modality vocabulary."
            )
        selected_loss = nn.functional.cross_entropy(
            selected_logits,
            selected_target,
            reduction="none",
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
                "text_tokens": text_count.to(dtype=hidden_states.dtype),
                "audio_tokens": audio_count.to(dtype=hidden_states.dtype),
            },
        )
