from __future__ import annotations

from anytrain.lightning import LossSummaryCallback


class LossSummary(LossSummaryCallback):
    def __init__(self) -> None:
        super().__init__()
