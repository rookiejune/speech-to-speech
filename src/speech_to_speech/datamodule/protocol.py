from __future__ import annotations

from functools import cached_property
from typing import Protocol

from ..runtime.protocol import DataRuntime
from ..runtime.types import Codec


class DatasetRuntime(DataRuntime, Protocol):
    @cached_property
    def codec(self) -> Codec: ...


__all__ = ["DataRuntime", "DatasetRuntime"]
