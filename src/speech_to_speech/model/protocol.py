from __future__ import annotations

from functools import cached_property
from typing import Protocol

from torch import Tensor, nn

from ..runtime.protocol import TokenModelRuntime


class FlowSample(Protocol):
    final: Tensor


class FlowSamplingRuntime(Protocol):
    def sample(
        self,
        model: nn.Module,
        x_0: Tensor,
        *,
        time_grid: Tensor | None = None,
        **model_extras: object,
    ) -> FlowSample: ...


class FlowModelRuntime(TokenModelRuntime, Protocol):
    @cached_property
    def flow_matching(self) -> FlowSamplingRuntime: ...
