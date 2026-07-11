from __future__ import annotations

import sys
from enum import Enum, auto

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        @staticmethod
        def _generate_next_value_(
            name: str,
            start: int,
            count: int,
            last_values: list[str],
        ) -> str:
            return name.lower()

        def __str__(self) -> str:
            return self.value
