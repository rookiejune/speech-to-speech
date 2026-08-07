from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from anytrain import observation
from torch import Tensor, nn

_ModuleT = TypeVar("_ModuleT", bound=nn.Module)


def register(module_type: type[_ModuleT]) -> type[_ModuleT]:
    observation.registry.register(
        module_type,
        (
            observation.OutputObservation(
                "loss",
                _collect_loss,
                reduction=observation.Reduction.Mean,
                recommended=True,
            ),
            observation.ForwardEvent(
                "diagnostics",
                reduction=observation.Reduction.Mean,
            ),
        ),
    )
    return module_type


def recommend(module: nn.Module) -> None:
    observation.registry.recommend(module)


def emit(module: nn.Module, values: Mapping[str, Tensor]) -> None:
    observation.emit(
        module,
        "diagnostics",
        {name: observation.Curve(value.detach()) for name, value in values.items()},
    )


def _collect_loss(
    module: nn.Module,
    inputs: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    output: object,
) -> observation.Curve:
    del module, inputs, kwargs
    if not isinstance(output, Tensor) or output.ndim != 0:
        raise TypeError("loss observation requires a scalar Tensor output.")
    return observation.Curve(output)


__all__ = ["emit", "recommend", "register"]
