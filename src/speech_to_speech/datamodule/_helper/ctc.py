from __future__ import annotations

from anydataset.types import Modality
from torch import Tensor

from ..protocol import DataRuntime
from ..sample import Speech
from ..target import CTCTarget


def ctc_target(
    positions: Tensor,
    speech: Speech,
    runtime: DataRuntime,
) -> CTCTarget:
    """Build local-text CTC supervision for one audio span."""
    labels = speech.text_token_ids
    if labels.numel() == 0:
        raise ValueError("CTC transcript must contain at least one text token.")
    text_start, text_end = runtime.layout.blocks[Modality.TEXT.value]
    text_vocab_size = text_end - text_start
    if bool((labels < 0).any()) or bool((labels >= text_vocab_size).any()):
        raise ValueError("CTC transcript contains an id outside the text vocabulary.")
    blank = runtime.pad_token_id - text_start
    if not 0 <= blank < text_vocab_size:
        raise ValueError("runtime pad token must belong to the text vocabulary for CTC.")
    if bool(labels.eq(blank).any()):
        raise ValueError("CTC transcript must not contain the configured blank token.")
    return CTCTarget(
        token_positions=positions,
        text_token_ids=labels,
    )


__all__ = ["ctc_target"]
