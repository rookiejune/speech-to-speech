from __future__ import annotations

from anytrain.lightning import LossSummaryCallback

from ...loss.types import loss_items


class LossSummary(LossSummaryCallback):
    def __init__(self) -> None:
        super().__init__(loss_items_fn=loss_items)
