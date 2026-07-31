from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .types import ModelBatch


@dataclass
class PreferenceBatch:
    chosen: ModelBatch
    rejected: ModelBatch
    ref_chosen_logps: Tensor | None = None
    ref_rejected_logps: Tensor | None = None

    def __post_init__(self) -> None:
        batch_size = self.chosen.input_ids.size(0)
        if self.rejected.input_ids.size(0) != batch_size:
            raise ValueError("chosen and rejected batches must have the same batch size.")
        if self.chosen.input_ids.shape != self.chosen.token_labels.shape:
            raise ValueError("chosen input ids and token labels must align.")
        if self.rejected.input_ids.shape != self.rejected.token_labels.shape:
            raise ValueError("rejected input ids and token labels must align.")
        if self.chosen.input_ids.shape != self.rejected.input_ids.shape:
            raise ValueError("chosen and rejected batches must have aligned shapes.")
        if (self.ref_chosen_logps is None) != (self.ref_rejected_logps is None):
            raise ValueError("reference chosen and rejected logps must be provided together.")
        for name, logps in (
            ("ref_chosen_logps", self.ref_chosen_logps),
            ("ref_rejected_logps", self.ref_rejected_logps),
        ):
            if logps is not None and logps.shape != (batch_size,):
                raise ValueError(f"{name} must have shape [batch].")
