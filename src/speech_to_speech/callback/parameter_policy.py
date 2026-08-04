from __future__ import annotations

from anytrain.lightning import ParameterPolicyCallback
from lightning import pytorch as pl
from torch import nn

from speech_to_speech.model.ctc import CTCRoute
from speech_to_speech.training.parameter_policy import (
    ParameterPolicyConfig,
    ParameterPolicyTrainability,
)


def build_parameter_policy(
    config: ParameterPolicyConfig,
    *,
    active_ctc_routes: frozenset[CTCRoute] = frozenset(CTCRoute),
) -> ParameterPolicyCallback:
    return ParameterPolicyCallback(
        ParameterPolicyTrainability(
            config.spec(),
            active_ctc_routes=active_ctc_routes,
        ),
        module=_model,
    )


def _model(pl_module: pl.LightningModule) -> nn.Module:
    model = getattr(pl_module, "model", None)
    if not isinstance(model, nn.Module):
        raise TypeError("parameter policy callback requires pl_module.model.")
    return model


__all__ = ["build_parameter_policy"]
