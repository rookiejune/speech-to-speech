from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from speech_to_speech.model.diagonal import diagonal_flow_sample, serial_forward_count


@dataclass(frozen=True)
class BenchmarkResult:
    serial_seconds: float
    diagonal_seconds: float
    serial_forward_count: int
    diagonal_forward_count: int
    diagonal_packed_row_count: int

    @property
    def speedup(self) -> float:
        if self.diagonal_seconds == 0:
            return float("inf")
        return self.serial_seconds / self.diagonal_seconds


class SyntheticDiT(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.forward_count = 0
        self.packed_row_count = 0

    def reset_counts(self) -> None:
        self.forward_count = 0
        self.packed_row_count = 0

    def forward(
        self,
        *,
        x_t: Tensor,
        last_hidden_state: Tensor,
        timesteps: Tensor,
        acoustic_condition: Tensor,
        attention_mask: Tensor,
    ) -> BaseModelOutputWithPast:
        self.forward_count += 1
        self.packed_row_count += x_t.size(0)
        condition = (
            x_t
            + last_hidden_state
            + acoustic_condition.unsqueeze(1)
            + timesteps.reshape(-1, 1, 1).to(dtype=x_t.dtype)
        )
        velocity = self.proj(condition)
        velocity = velocity * attention_mask.to(dtype=velocity.dtype).unsqueeze(-1)
        return BaseModelOutputWithPast(last_hidden_state=velocity)


@torch.no_grad()
def serial_flow_sample(
    dit: SyntheticDiT,
    x_0: Tensor,
    *,
    last_hidden_state: Tensor,
    acoustic_condition: Tensor,
    mask: Tensor,
    num_steps: int,
    chunk_size: int,
) -> Tensor:
    state = x_0.clone()
    time_grid = torch.linspace(0, 1, num_steps + 1, device=x_0.device, dtype=x_0.dtype)
    for start in range(0, x_0.size(1), chunk_size):
        end = min(start + chunk_size, x_0.size(1))
        for step in range(num_steps):
            outputs = dit(
                x_t=state[:, start:end],
                last_hidden_state=last_hidden_state[:, start:end],
                timesteps=time_grid.new_full((x_0.size(0),), time_grid[step]),
                acoustic_condition=acoustic_condition,
                attention_mask=mask[:, start:end].long(),
            )
            delta = time_grid[step + 1] - time_grid[step]
            updated = state[:, start:end] + delta * outputs.last_hidden_state
            active = mask[:, start:end].unsqueeze(-1)
            state[:, start:end] = torch.where(active, updated, state[:, start:end])
    return state


def benchmark(args: argparse.Namespace) -> BenchmarkResult:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    x_0 = torch.randn(args.batch_size, args.frames, args.hidden_size, device=device)
    hidden = torch.randn_like(x_0)
    acoustic_condition = torch.randn(args.batch_size, args.hidden_size, device=device)
    mask = torch.ones(args.batch_size, args.frames, dtype=torch.bool, device=device)
    dit = SyntheticDiT(args.hidden_size).to(device).eval()

    for _ in range(args.warmup):
        serial_flow_sample(
            dit,
            x_0,
            last_hidden_state=hidden,
            acoustic_condition=acoustic_condition,
            mask=mask,
            num_steps=args.steps,
            chunk_size=args.chunk_size,
        )
        diagonal_flow_sample(
            dit,
            x_0,
            last_hidden_state=hidden,
            acoustic_condition=acoustic_condition,
            mask=mask,
            num_steps=args.steps,
            chunk_size=args.chunk_size,
        )

    dit.reset_counts()
    serial_seconds = _time_repeats(
        args.repeats,
        lambda: serial_flow_sample(
            dit,
            x_0,
            last_hidden_state=hidden,
            acoustic_condition=acoustic_condition,
            mask=mask,
            num_steps=args.steps,
            chunk_size=args.chunk_size,
        ),
        device=device,
    )
    serial_calls = dit.forward_count

    dit.reset_counts()
    diagonal_seconds = _time_repeats(
        args.repeats,
        lambda: diagonal_flow_sample(
            dit,
            x_0,
            last_hidden_state=hidden,
            acoustic_condition=acoustic_condition,
            mask=mask,
            num_steps=args.steps,
            chunk_size=args.chunk_size,
        ),
        device=device,
    )
    diagonal_calls = dit.forward_count // args.repeats
    diagonal_output = diagonal_flow_sample(
        dit,
        x_0,
        last_hidden_state=hidden,
        acoustic_condition=acoustic_condition,
        mask=mask,
        num_steps=args.steps,
        chunk_size=args.chunk_size,
    )

    return BenchmarkResult(
        serial_seconds=serial_seconds,
        diagonal_seconds=diagonal_seconds,
        serial_forward_count=serial_calls // args.repeats,
        diagonal_forward_count=diagonal_calls,
        diagonal_packed_row_count=diagonal_output.packed_row_count,
    )


def _time_repeats(count: int, fn, *, device: torch.device) -> float:
    _sync(device)
    start = time.perf_counter()
    for _ in range(count):
        fn()
    _sync(device)
    return (time.perf_counter() - start) / count


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic diagonal acoustic scheduler benchmark.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=_device, default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = benchmark(args)
    print(
        json.dumps(
            {
                "device": args.device,
                "batch_size": args.batch_size,
                "frames": args.frames,
                "hidden_size": args.hidden_size,
                "chunk_size": args.chunk_size,
                "steps": args.steps,
                "serial_seconds": result.serial_seconds,
                "diagonal_seconds": result.diagonal_seconds,
                "speedup": result.speedup,
                "serial_forward_count": result.serial_forward_count,
                "diagonal_forward_count": result.diagonal_forward_count,
                "diagonal_packed_row_count": result.diagonal_packed_row_count,
                "expected_serial_forward_count": serial_forward_count(
                    frame_count=args.frames,
                    chunk_size=args.chunk_size,
                    num_steps=args.steps,
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
