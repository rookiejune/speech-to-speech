from __future__ import annotations

from collections.abc import Sequence


def window_summary(
    values: Sequence[float],
    window: int = 20,
) -> dict[str, float | int | None]:
    if not values:
        return {"steps": 0}
    size = min(window, len(values))
    first = sum(values[:size]) / size
    last = sum(values[-size:]) / size
    return {
        "steps": len(values),
        "window": size,
        "first": values[0],
        "last": values[-1],
        "first_mean": first,
        "last_mean": last,
        "last_to_first": None if first == 0 else last / first,
    }


__all__ = ["window_summary"]
