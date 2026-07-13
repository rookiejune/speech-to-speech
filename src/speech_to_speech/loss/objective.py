from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import nn

from ..datamodule.types import ModelBatch
from ..model.protocol import BaseModel
from .types import Outputs


ModelT_contra = TypeVar("ModelT_contra", bound=BaseModel, contravariant=True)


class Objective(nn.Module, Generic[ModelT_contra], ABC):
    @abstractmethod
    def forward(self, batch: ModelBatch, model: ModelT_contra) -> Outputs: ...
