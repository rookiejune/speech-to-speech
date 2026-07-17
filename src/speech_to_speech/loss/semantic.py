from collections.abc import Callable

from anytrain.idspace import Layout
from torch import Tensor, nn

from .types import LossItem


class SemanticLoss(nn.Module):
    def __init__(self, layout: Layout) -> None:
        super().__init__()
        self.layout = layout

    def forward(
        self,
        hidden_states: Tensor,
        labels: Tensor,
        logits: Callable[[Tensor], Tensor],
    ) -> LossItem:
        if hidden_states.dim() != 3 or labels.dim() != 2:
            raise ValueError(
                "semantic hidden states and labels must have shapes [B, T, H] and [B, T]."
            )
        if hidden_states.shape[:2] != labels.shape:
            raise ValueError(
                "semantic hidden states and labels must align on batch and sequence."
            )
        target = labels[:, 1:]
        prediction = hidden_states[:, :-1]

        valid = target.ne(-100)
        text_start, text_end = self.layout.blocks["text"]
        audio_start, audio_end = self.layout.blocks["audio"]
        text_mask = target.ge(text_start) & target.lt(text_end)
        audio_mask = target.ge(audio_start) & target.lt(audio_end)
        if bool((valid & ~(text_mask | audio_mask)).any()):
            raise ValueError(
                "labels contain an id outside the text and audio layout blocks."
            )
        selected_target = target[valid]
        if selected_target.numel() == 0:
            raise ValueError("semantic labels must contain at least one target token.")
        selected_logits = logits(prediction[valid])
        if selected_logits.shape != (selected_target.numel(), self.layout.vocab_size):
            raise ValueError("semantic logits do not match selected targets and layout.")
        selected_loss = nn.functional.cross_entropy(
            selected_logits,
            selected_target,
            reduction="none",
        )
        token_loss = selected_loss.new_zeros(target.shape)
        token_loss[valid] = selected_loss
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
