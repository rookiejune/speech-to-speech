from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from torch import Tensor


ACOUSTIC_PAD_ID = -1
CTC_PAD_ID = -1


class AcousticTarget(TypedDict):
    semantic_codes: Tensor
    codes: Tensor
    token_positions: Tensor


class CTCTarget(TypedDict):
    """Transcript supervision attached to one audio span.

    ``token_positions`` uses the full teacher-forcing sequence axis. Source
    CTC reads the hidden state at each position; target CTC reads its causal
    predecessor. ``text_token_ids`` stays in the tokenizer-local text space.
    """

    token_positions: Tensor
    text_token_ids: Tensor


@dataclass(frozen=True)
class Labels:
    """Training-only supervision for the response side of a sample."""

    response_ids: Tensor
    token_labels: Tensor
    acoustic_target: AcousticTarget | None = None
    source_ctc: CTCTarget | None = None
    target_ctc: CTCTarget | None = None
    audio_seconds: float = 0.0

__all__ = ["ACOUSTIC_PAD_ID", "CTC_PAD_ID", "AcousticTarget", "CTCTarget", "Labels"]
