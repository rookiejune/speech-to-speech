"""Diagonal acoustic flow scheduling utilities.

The scheduler runs a chunked acoustic flow sampler on wavefronts. Each cell is
one `(chunk, flow_step)` pair; cells with the same wave index can be packed into
one DiT forward even though they use different diffusion timesteps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .acoustic import acoustic_velocity


@dataclass(frozen=True)
class DiagonalCell:
    chunk_index: int
    step_index: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class DiagonalBatch:
    wave_index: int
    cells: tuple[DiagonalCell, ...]


@dataclass(frozen=True)
class DiagonalSample:
    final: Tensor
    time_grid: Tensor
    schedule: tuple[DiagonalBatch, ...]
    forward_count: int
    packed_row_count: int


@dataclass(frozen=True)
class SerialSample:
    final: Tensor
    time_grid: Tensor
    forward_count: int


def diagonal_schedule(
    *,
    frame_count: int,
    chunk_size: int,
    num_steps: int,
    wave_stride: int = 1,
) -> tuple[DiagonalBatch, ...]:
    _validate_schedule_args(
        frame_count=frame_count,
        chunk_size=chunk_size,
        num_steps=num_steps,
        wave_stride=wave_stride,
    )
    waves: dict[int, list[DiagonalCell]] = {}
    for chunk_index in range(_chunk_count(frame_count, chunk_size)):
        start = chunk_index * chunk_size
        end = min(start + chunk_size, frame_count)
        for step_index in range(num_steps):
            wave_index = chunk_index * wave_stride + step_index
            waves.setdefault(wave_index, []).append(
                DiagonalCell(
                    chunk_index=chunk_index,
                    step_index=step_index,
                    start=start,
                    end=end,
                )
            )
    return tuple(
        DiagonalBatch(wave_index=wave_index, cells=tuple(cells))
        for wave_index, cells in sorted(waves.items())
    )


def serial_forward_count(
    *,
    frame_count: int,
    chunk_size: int,
    num_steps: int,
) -> int:
    _validate_schedule_args(
        frame_count=frame_count,
        chunk_size=chunk_size,
        num_steps=num_steps,
        wave_stride=1,
    )
    return _chunk_count(frame_count, chunk_size) * num_steps


@torch.no_grad()
def serial_flow_sample(
    dit: nn.Module,
    x_0: Tensor,
    *,
    last_hidden_state: Tensor,
    acoustic_condition: Tensor,
    mask: Tensor,
    num_steps: int,
    chunk_size: int,
    time_grid: Tensor | None = None,
    guidance_scale: float = 1.0,
) -> SerialSample:
    _validate_sample_inputs(
        x_0=x_0,
        last_hidden_state=last_hidden_state,
        acoustic_condition=acoustic_condition,
        mask=mask,
    )
    _validate_schedule_args(
        frame_count=x_0.size(1),
        chunk_size=chunk_size,
        num_steps=num_steps,
        wave_stride=1,
    )
    time_grid = _time_grid(
        num_steps,
        device=x_0.device,
        dtype=x_0.dtype,
        value=time_grid,
    )
    state = x_0.clone()
    forward_count = 0

    for start in range(0, x_0.size(1), chunk_size):
        end = min(start + chunk_size, x_0.size(1))
        chunk_mask = mask[:, start:end]
        for step_index in range(num_steps):
            velocity = acoustic_velocity(
                dit,
                x_t=state[:, start:end],
                last_hidden_state=last_hidden_state[:, start:end],
                timesteps=time_grid[step_index].expand(x_0.size(0)),
                acoustic_condition=acoustic_condition,
                mask=chunk_mask,
                guidance_scale=guidance_scale,
            )
            forward_count += 1
            delta = time_grid[step_index + 1] - time_grid[step_index]
            current = state[:, start:end]
            updated = current + delta * velocity
            active = chunk_mask.unsqueeze(-1)
            state[:, start:end] = torch.where(active, updated, current)

    return SerialSample(
        final=state,
        time_grid=time_grid,
        forward_count=forward_count,
    )


@torch.no_grad()
def diagonal_flow_sample(
    dit: nn.Module,
    x_0: Tensor,
    *,
    last_hidden_state: Tensor,
    acoustic_condition: Tensor,
    mask: Tensor,
    num_steps: int,
    chunk_size: int,
    wave_stride: int = 1,
    time_grid: Tensor | None = None,
    guidance_scale: float = 1.0,
) -> DiagonalSample:
    _validate_sample_inputs(
        x_0=x_0,
        last_hidden_state=last_hidden_state,
        acoustic_condition=acoustic_condition,
        mask=mask,
    )
    schedule = diagonal_schedule(
        frame_count=x_0.size(1),
        chunk_size=chunk_size,
        num_steps=num_steps,
        wave_stride=wave_stride,
    )
    time_grid = _time_grid(
        num_steps,
        device=x_0.device,
        dtype=x_0.dtype,
        value=time_grid,
    )
    state = x_0.clone()
    packed_row_count = 0

    for batch in schedule:
        packed = _pack_cells(
            batch.cells,
            state=state,
            last_hidden_state=last_hidden_state,
            acoustic_condition=acoustic_condition,
            mask=mask,
            time_grid=time_grid,
        )
        packed_row_count += packed.x_t.size(0)
        velocity = acoustic_velocity(
            dit,
            x_t=packed.x_t,
            last_hidden_state=packed.last_hidden_state,
            timesteps=packed.timesteps,
            acoustic_condition=packed.acoustic_condition,
            mask=packed.attention_mask,
            guidance_scale=guidance_scale,
        )
        _scatter_cells(
            batch.cells,
            state=state,
            velocity=velocity,
            mask=mask,
            time_grid=time_grid,
            batch_size=x_0.size(0),
        )

    return DiagonalSample(
        final=state,
        time_grid=time_grid,
        schedule=schedule,
        forward_count=len(schedule),
        packed_row_count=packed_row_count,
    )


@dataclass(frozen=True)
class _PackedCells:
    x_t: Tensor
    last_hidden_state: Tensor
    acoustic_condition: Tensor
    attention_mask: Tensor
    timesteps: Tensor


def _pack_cells(
    cells: tuple[DiagonalCell, ...],
    *,
    state: Tensor,
    last_hidden_state: Tensor,
    acoustic_condition: Tensor,
    mask: Tensor,
    time_grid: Tensor,
) -> _PackedCells:
    batch_size, _, hidden_size = state.shape
    max_length = max(cell.length for cell in cells)
    row_count = batch_size * len(cells)
    x_t = state.new_zeros((row_count, max_length, hidden_size))
    condition_hidden = last_hidden_state.new_zeros((row_count, max_length, hidden_size))
    attention_mask = torch.zeros(
        (row_count, max_length),
        dtype=torch.bool,
        device=state.device,
    )
    conditions = acoustic_condition.new_zeros((row_count, acoustic_condition.size(-1)))
    timesteps = time_grid.new_zeros(row_count)

    for cell_index, cell in enumerate(cells):
        row_start = cell_index * batch_size
        row_end = row_start + batch_size
        length = cell.length
        x_t[row_start:row_end, :length] = state[:, cell.start : cell.end]
        condition_hidden[row_start:row_end, :length] = last_hidden_state[
            :,
            cell.start : cell.end,
        ]
        attention_mask[row_start:row_end, :length] = mask[:, cell.start : cell.end]
        conditions[row_start:row_end] = acoustic_condition
        timesteps[row_start:row_end] = time_grid[cell.step_index]

    return _PackedCells(
        x_t=x_t,
        last_hidden_state=condition_hidden,
        acoustic_condition=conditions,
        attention_mask=attention_mask,
        timesteps=timesteps,
    )


def _scatter_cells(
    cells: tuple[DiagonalCell, ...],
    *,
    state: Tensor,
    velocity: Tensor,
    mask: Tensor,
    time_grid: Tensor,
    batch_size: int,
) -> None:
    for cell_index, cell in enumerate(cells):
        row_start = cell_index * batch_size
        row_end = row_start + batch_size
        length = cell.length
        delta = time_grid[cell.step_index + 1] - time_grid[cell.step_index]
        current = state[:, cell.start : cell.end]
        updated = current + delta * velocity[row_start:row_end, :length]
        active = mask[:, cell.start : cell.end].unsqueeze(-1)
        state[:, cell.start : cell.end] = torch.where(active, updated, current)


def _time_grid(
    num_steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    value: Tensor | None,
) -> Tensor:
    if value is None:
        return torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)
    if value.dim() != 1 or value.numel() != num_steps + 1:
        raise ValueError("time_grid must have shape (num_steps + 1,).")
    if not torch.is_floating_point(value) or torch.is_complex(value):
        raise TypeError("time_grid must be a real floating point tensor.")
    if bool((value[1:] <= value[:-1]).any()):
        raise ValueError("time_grid values must be strictly increasing.")
    return value.to(device=device, dtype=dtype)


def _validate_schedule_args(
    *,
    frame_count: int,
    chunk_size: int,
    num_steps: int,
    wave_stride: int,
) -> None:
    for name, value in {
        "frame_count": frame_count,
        "chunk_size": chunk_size,
        "num_steps": num_steps,
        "wave_stride": wave_stride,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be positive.")


def _validate_sample_inputs(
    *,
    x_0: Tensor,
    last_hidden_state: Tensor,
    acoustic_condition: Tensor,
    mask: Tensor,
) -> None:
    if x_0.dim() != 3:
        raise ValueError("x_0 must have shape [batch, time, dim].")
    if last_hidden_state.shape != x_0.shape:
        raise ValueError("last_hidden_state must have the same shape as x_0.")
    if acoustic_condition.dim() != 2 or acoustic_condition.shape != (
        x_0.size(0),
        x_0.size(-1),
    ):
        raise ValueError("acoustic_condition must have shape [batch, dim].")
    if mask.shape != x_0.shape[:2]:
        raise ValueError("mask must have shape [batch, time].")
    if not torch.is_floating_point(x_0) or torch.is_complex(x_0):
        raise TypeError("x_0 must be a real floating point tensor.")
    if x_0.device != last_hidden_state.device or x_0.device != mask.device:
        raise ValueError("x_0, last_hidden_state, and mask must be on the same device.")
    if x_0.device != acoustic_condition.device:
        raise ValueError("acoustic_condition must be on the same device as x_0.")


def _chunk_count(frame_count: int, chunk_size: int) -> int:
    return (frame_count + chunk_size - 1) // chunk_size


__all__ = [
    "DiagonalBatch",
    "DiagonalCell",
    "DiagonalSample",
    "SerialSample",
    "diagonal_flow_sample",
    "diagonal_schedule",
    "serial_forward_count",
    "serial_flow_sample",
]
