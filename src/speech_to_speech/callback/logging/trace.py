from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Iterator

_STARTED_AT = time.perf_counter()


def event(name: str, state: str, **fields: Any) -> None:
    print(
        json.dumps(
            {
                "event": name,
                "state": state,
                "elapsed_seconds": round(time.perf_counter() - _STARTED_AT, 3),
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


@contextmanager
def stage(name: str, **fields: Any) -> Iterator[None]:
    started_at = time.perf_counter()
    event(name, "start", **fields)
    try:
        yield
    except Exception as error:
        event(
            name,
            "error",
            seconds=round(time.perf_counter() - started_at, 3),
            error_type=type(error).__name__,
            error=str(error),
            **fields,
        )
        raise
    event(
        name,
        "done",
        seconds=round(time.perf_counter() - started_at, 3),
        **fields,
    )


__all__ = ["event", "stage"]
