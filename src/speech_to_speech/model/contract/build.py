from __future__ import annotations

from collections.abc import Mapping

from torch import nn

from ._interface import interface_contract, state_dict_contract
from ._protocol import ContractModel
from ._runtime import runtime_contract
from .core import ModelCheckpointContract, canonical_value


def build_model_contract(
    model: ContractModel,
    acoustic: Mapping[str, object],
) -> ModelCheckpointContract:
    """Build a contract from resolved runtime resources and realized modules."""
    if not isinstance(model, nn.Module):
        raise TypeError("checkpoint contract models must be torch modules.")
    return ModelCheckpointContract.from_components(
        {
            "runtime": runtime_contract(model),
            "interface": interface_contract(model),
            "acoustic": canonical_value(acoustic),
            "state_dict": state_dict_contract(model),
        }
    )

__all__ = ["build_model_contract"]
