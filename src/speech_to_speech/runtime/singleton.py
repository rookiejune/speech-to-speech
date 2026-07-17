from __future__ import annotations

from .runtime import Config, Runtime


_runtime: Runtime | None = None


def init_runtime(config: Config) -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = Runtime(config=config)
    elif _runtime.config != config:
        raise RuntimeError("runtime is already initialized with a different config.")
    return _runtime
