from __future__ import annotations

from ..datamodule.types import ModelBatch
from ..model.protocol import CausalLM


def loss(batch: ModelBatch, model: CausalLM):
    """Placeholder for the P2 causal acoustic objective."""
    raise NotImplementedError("P2 causal acoustic loss is not implemented.")
