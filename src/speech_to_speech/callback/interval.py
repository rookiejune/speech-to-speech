from __future__ import annotations

from collections.abc import Mapping


class TrainInterval:
    """Run at most once per optimizer step when ``step % every_n_steps == 0``.

    Same-step dedup covers ``accumulate_grad_batches`` microbatches that share
    ``global_step``.
    """

    def __init__(self, *, every_n_steps: int) -> None:
        if isinstance(every_n_steps, bool) or not isinstance(every_n_steps, int):
            raise TypeError("every_n_steps must be an integer.")
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive.")
        self.every_n_steps = every_n_steps
        self._last_step: int | None = None

    def should_run(self, step: int) -> bool:
        if self._last_step == step:
            return False
        if self._last_step is not None and step < self._last_step:
            raise RuntimeError("trainer global_step moved backwards.")
        self._last_step = step
        return step % self.every_n_steps == 0

    def state_dict(self) -> dict[str, int | None]:
        return {"last_step": self._last_step}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        last_step = state.get("last_step")
        # Older checkpoints stored None as -1.0.
        if last_step is None or last_step == -1 or last_step == -1.0:
            self._last_step = None
            return
        if isinstance(last_step, bool) or not isinstance(last_step, (int, float)):
            raise ValueError("interval last_step must be None or a non-negative integer.")
        value = int(last_step)
        if value != last_step or value < 0:
            raise ValueError("interval last_step must be None or a non-negative integer.")
        self._last_step = value


__all__ = ["TrainInterval"]
