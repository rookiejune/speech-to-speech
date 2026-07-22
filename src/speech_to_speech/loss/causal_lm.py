from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .._tensor import is_signed_integer_dtype
from .types import LossItem


class CausalAcousticLoss(nn.Module):
    """Masked codebook cross entropy for a frame-parallel RVQ decoder."""

    def forward(
        self,
        logits: tuple[Tensor, ...],
        labels: Tensor,
        mask: Tensor,
        *,
        validate: bool = True,
    ) -> LossItem:
        if labels.dim() != 3 or mask.shape != labels.shape[:2]:
            raise ValueError(
                "RVQ labels and mask must have shapes [B, F, Q] and [B, F]."
            )
        if not is_signed_integer_dtype(labels.dtype):
            raise TypeError("RVQ labels must use a signed integer dtype.")
        if mask.dtype != torch.bool:
            raise TypeError("RVQ target mask must be boolean.")
        if len(logits) != labels.size(-1):
            raise ValueError("RVQ logits must provide one tensor per codebook.")

        if validate:
            valid_targets = labels[mask]
            limits = labels.new_tensor([value.size(-1) for value in logits])
            if bool(((valid_targets < 0) | (valid_targets >= limits)).any()):
                raise ValueError("valid RVQ target is outside its logits codebook.")

        losses = []
        for codebook, value in enumerate(logits):
            if value.shape[:2] != labels.shape[:2]:
                raise ValueError(
                    "RVQ logits must align with labels on batch and frame."
                )
            target = labels[..., codebook]
            safe_value = value.masked_fill(~mask[..., None], 0)
            safe_target = target.masked_fill(~mask, 0).to(dtype=torch.long)
            loss = F.cross_entropy(
                safe_value.movedim(-1, 1),
                safe_target,
                reduction="none",
            ).masked_fill(~mask, 0)
            losses.append(loss)
        frame_losses = torch.stack(losses, dim=-1)
        frame_count = mask.sum(dim=1).clamp_min(1)
        codebook_losses = frame_losses.sum(dim=1)
        codebook_losses = codebook_losses / frame_count[:, None]
        details = {
            f"codebook_{codebook}": codebook_losses[:, codebook]
            for codebook in range(labels.size(-1))
        }
        details["frames"] = frame_count.to(dtype=frame_losses.dtype)
        return LossItem(
            loss=codebook_losses.mean(dim=-1),
            details=details,
        )
