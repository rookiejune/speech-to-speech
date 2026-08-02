from __future__ import annotations

from anytrain.lightning import ParameterPolicyCallback
from lightning import pytorch as pl
from torch import nn

from speech_to_speech.stage import (
    ParameterPolicyConfig,
    ParameterPolicyTrainability,
)


def build_parameter_policy(
    config: ParameterPolicyConfig,
) -> ParameterPolicyCallback:
    return ParameterPolicyCallback(
        ParameterPolicyTrainability(config.spec()),
        module=_model,
    )


def _model(pl_module: pl.LightningModule) -> nn.Module:
    model = getattr(pl_module, "model", None)
    if not isinstance(model, nn.Module):
        raise TypeError("parameter policy callback requires pl_module.model.")
    return model


__all__ = ["build_parameter_policy"]
