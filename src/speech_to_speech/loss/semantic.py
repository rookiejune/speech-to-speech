from anytrain.idspace import Layout
from torch import Tensor, nn

from .types import LossItem


class SemanticLoss(nn.Module):
    def __init__(self, layout: Layout) -> None:
        super().__init__()
        self.layout = layout

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
    ) -> LossItem:
        if logits.dim() != 3 or labels.dim() != 2:
            raise ValueError(
                "semantic logits and labels must have shapes [B, T, V] and [B, T]."
            )
        if logits.shape[:2] != labels.shape:
            raise ValueError(
                "semantic logits and labels must align on batch and sequence."
            )
        target = labels[:, 1:]
        prediction = logits[:, :-1]
        if prediction.size(-1) != self.layout.vocab_size:
            raise ValueError(
                "semantic logits do not match the runtime layout vocabulary."
            )

        valid = target.ne(-100)
        text_start, text_end = self.layout.blocks["text"]
        audio_start, audio_end = self.layout.blocks["audio"]
        text_mask = target.ge(text_start) & target.lt(text_end)
        audio_mask = target.ge(audio_start) & target.lt(audio_end)
        if bool((valid & ~(text_mask | audio_mask)).any()):
            raise ValueError(
                "labels contain an id outside the text and audio layout blocks."
            )

        token_loss = nn.functional.cross_entropy(
            prediction.transpose(1, 2),
            target.masked_fill(~valid, -100),
            ignore_index=-100,
            reduction="none",
        )
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
                "text_tokens": text_count.to(dtype=logits.dtype),
                "audio_tokens": audio_count.to(dtype=logits.dtype),
            },
        )
