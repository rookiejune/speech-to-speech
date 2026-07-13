from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .types import LossItem


class CausalAcousticLoss(nn.Module):
    """Masked codebook cross entropy for a frame-parallel RVQ decoder."""

    def forward(
        self,
        logits: tuple[Tensor, ...],
        labels: Tensor,
        mask: Tensor,
    ) -> LossItem:
        if labels.dim() != 3 or mask.shape != labels.shape[:2]:
            raise ValueError("RVQ labels and mask must have shapes [B, F, Q] and [B, F].")
        if mask.dtype != torch.bool:
            raise TypeError("RVQ target mask must be boolean.")
        if len(logits) != labels.size(-1):
            raise ValueError("RVQ logits must provide one tensor per codebook.")

        losses = []
        for codebook, value in enumerate(logits):
            if value.shape[:2] != labels.shape[:2]:
                raise ValueError("RVQ logits must align with labels on batch and frame.")
            target = labels[..., codebook]
            if bool(((target < 0) & mask).any()):
                raise ValueError("valid RVQ targets cannot contain padding IDs.")
            loss = F.cross_entropy(
                value.movedim(-1, 1),
                target.clamp_min(0),
                reduction="none",
            )
            losses.append(loss)
        frame_losses = torch.stack(losses, dim=-1)
        weights = mask.to(dtype=frame_losses.dtype)
        frame_count = weights.sum(dim=1).clamp_min(1)
        codebook_losses = (frame_losses * weights[..., None]).sum(dim=1)
        codebook_losses = codebook_losses / frame_count[:, None]
        return LossItem(
            loss=codebook_losses.mean(dim=-1),
            details={
                f"codebook_{codebook}": codebook_losses[:, codebook]
                for codebook in range(labels.size(-1))
            },
        )
