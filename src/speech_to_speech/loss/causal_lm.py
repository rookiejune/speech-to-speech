from __future__ import annotations

from ..datamodule.types import Batch
from ..model.protocol import CausalLM


def loss(batch: Batch, model: CausalLM):
    """Placeholder for the P2 causal acoustic objective."""
    raise NotImplementedError("P2 causal acoustic loss is not implemented.")
