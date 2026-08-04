from __future__ import annotations

from collections.abc import Callable, Set

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout
from torch import Tensor, nn

from .._tensor import is_signed_integer_dtype
from ..model.embedding.fsq import FsqNeighbors
from ..task import PredictionModality
from .types import LossItem


class TokenLoss(nn.Module):
    def __init__(
        self,
        layout: Layout,
        *,
        audio_neighbor_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if (
            isinstance(audio_neighbor_smoothing, bool)
            or not isinstance(audio_neighbor_smoothing, (int, float))
            or not 0 <= audio_neighbor_smoothing < 1
        ):
            raise ValueError("audio neighbor smoothing must be in [0, 1).")
        self.layout = layout
        self.audio_neighbor_smoothing = float(audio_neighbor_smoothing)

    def forward(
        self,
        hidden_states: Tensor,
        token_labels: Tensor,
        prediction: PredictionModality | Modality,
        token_logits: Callable[..., Tensor],
        *,
        audio_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
        audio_neighbors: Callable[[Tensor], FsqNeighbors | None] | None = None,
        validate: bool = True,
    ) -> LossItem:
        if hidden_states.dim() != 3 or token_labels.dim() != 2:
            raise ValueError(
                "token hidden states and labels must have shapes [B, T, H] and [B, T]."
            )
        if hidden_states.shape[:2] != token_labels.shape:
            raise ValueError("token hidden states and labels must align on sequence.")
        if not is_signed_integer_dtype(token_labels.dtype):
            raise TypeError("token labels must use a signed integer dtype.")
        if not isinstance(validate, bool):
            raise TypeError("validate must be a boolean.")
        modalities = _modalities(prediction)
        target = token_labels[:, 1:]
        prediction_states = hidden_states[:, :-1]

        valid = target.ne(-100)
        modality_mask = torch.zeros_like(valid)
        for modality in modalities:
            start, end = self.layout.blocks[modality.value]
            modality_mask |= target.ge(start) & target.lt(end)
        if validate:
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
        token_loss = self._loss(
            prediction_states,
            target,
            valid,
            modalities,
            token_logits,
            audio_hidden_states=(
                None if audio_hidden_states is None else audio_hidden_states[:, :-1]
            ),
            attention_mask=(None if attention_mask is None else attention_mask[:, :-1]),
            audio_neighbors=audio_neighbors,
        )
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
        audio_neighbors: Callable[[Tensor], FsqNeighbors | None] | None,
    ) -> Tensor:
        losses = prediction_states.new_zeros(target.shape)
        for modality in sorted(modalities, key=lambda value: value.value):
            start, end = self.layout.blocks[modality.value]
            mask = valid & target.ge(start) & target.lt(end)
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
            group_loss = self._group_loss(
                logits,
                selected_target,
                modality,
                audio_neighbors,
            )
            losses[mask] = group_loss.to(dtype=losses.dtype)
        return losses

    def _group_loss(
        self,
        logits: Tensor,
        target: Tensor,
        modality: Modality,
        audio_neighbors: Callable[[Tensor], FsqNeighbors | None] | None,
    ) -> Tensor:
        if modality is not Modality.AUDIO or self.audio_neighbor_smoothing == 0:
            return nn.functional.cross_entropy(logits, target, reduction="none")
        if audio_neighbors is None:
            raise ValueError(
                "audio neighbor smoothing requires an FSQ neighbor target provider."
            )
        neighbors = audio_neighbors(target)
        if neighbors is None:
            raise ValueError(
                "audio neighbor smoothing requires a factorized FSQ embedding."
            )
        expected = (target.numel(), neighbors.token_ids.size(-1))
        if (
            neighbors.token_ids.shape != expected
            or neighbors.weights.shape != expected
            or neighbors.valid.shape != expected
        ):
            raise ValueError("FSQ neighbor targets do not align with audio targets.")

        log_probabilities = nn.functional.log_softmax(logits, dim=-1)
        hard = -log_probabilities.gather(-1, target[:, None]).squeeze(-1)
        safe_ids = neighbors.token_ids.masked_fill(~neighbors.valid, 0)
        neighbor_nll = -log_probabilities.gather(-1, safe_ids)
        weights = neighbors.weights.to(
            device=neighbor_nll.device,
            dtype=neighbor_nll.dtype,
        )
        smooth = (neighbor_nll * weights).sum(dim=-1)
        mixed = (1 - self.audio_neighbor_smoothing) * hard
        mixed = mixed + self.audio_neighbor_smoothing * smooth
        return torch.where(neighbors.valid.any(dim=-1), mixed, hard)


def _modalities(prediction: PredictionModality | Modality) -> frozenset[Modality]:
    if isinstance(prediction, PredictionModality):
        modalities = prediction.supervised_modalities()
        if not modalities:
            raise ValueError(f"prediction modality {prediction.value} has no heads.")
        return modalities
    if prediction not in {Modality.TEXT, Modality.AUDIO}:
        raise ValueError(f"unsupported token modality: {prediction.value}")
    return frozenset({prediction})
