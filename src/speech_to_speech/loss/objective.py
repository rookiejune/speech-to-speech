from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

from torch import nn

from ..datamodule.types import ModelBatch
from .protocol import TokenObjectiveModel
from .types import Outputs, combine_outputs


ModelT_contra = TypeVar(
    "ModelT_contra", bound=TokenObjectiveModel, contravariant=True
)


class Objective(nn.Module, Generic[ModelT_contra], ABC):
    @abstractmethod
    def forward(self, batch: ModelBatch, model: ModelT_contra) -> Outputs: ...

    def reduce(self, outputs: Sequence[Outputs]) -> Outputs:
        return combine_outputs(outputs)
