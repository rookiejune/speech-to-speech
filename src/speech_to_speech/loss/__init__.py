"""Loss objective exports.

The token/MIMO contracts are lightweight, while preference and flow losses
pull optional audio-training dependencies.  Keep the latter lazy so importing
``speech_to_speech.loss.mimo`` does not require the full codec stack.
"""

from typing import TYPE_CHECKING

from .ctc import CTCAlignmentLoss, CTCConfig
from .mimo import MimoLoss, MimoObjective

if TYPE_CHECKING:
    from .preference import DPOObjective
    from .rollout import GRPOObjective

__all__ = [
    "CTCAlignmentLoss",
    "CTCConfig",
    "DPOObjective",
    "GRPOObjective",
    "MimoLoss",
    "MimoObjective",
]


def __getattr__(name: str) -> object:
    if name == "DPOObjective":
        from .preference import DPOObjective

        return DPOObjective
    if name == "GRPOObjective":
        from .rollout import GRPOObjective

        return GRPOObjective
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
