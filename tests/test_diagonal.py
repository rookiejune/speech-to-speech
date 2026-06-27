from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from speech_to_speech.model.diagonal import (
    diagonal_flow_sample,
    diagonal_schedule,
    serial_forward_count,
)
from speech_to_speech.model.orchestrator import Orchestrator
from helpers import MockQwen, MockTokenizer


class DiagonalScheduleTest(unittest.TestCase):
    def test_schedule_packs_wavefront_cells(self) -> None:
        schedule = diagonal_schedule(frame_count=10, chunk_size=4, num_steps=3)

        self.assertEqual(
            [[(cell.chunk_index, cell.step_index) for cell in batch.cells] for batch in schedule],
            [
                [(0, 0)],
                [(0, 1), (1, 0)],
                [(0, 2), (1, 1), (2, 0)],
                [(1, 2), (2, 1)],
                [(2, 2)],
            ],
        )
        self.assertEqual(serial_forward_count(frame_count=10, chunk_size=4, num_steps=3), 9)

    def test_diagonal_sample_matches_serial_independent_euler(self) -> None:
        dit = TrackingVelocityDiT(hidden_size=2)
        x_0 = torch.zeros((1, 5, 2))
        hidden = torch.ones_like(x_0)
        acoustic_condition = torch.zeros((1, 2))
        mask = torch.tensor([[True, True, True, False, True]])

        diagonal = diagonal_flow_sample(
            dit,
            x_0,
            last_hidden_state=hidden,
            acoustic_condition=acoustic_condition,
            mask=mask,
            num_steps=4,
            chunk_size=2,
        )
        serial = _serial_sample(
            TrackingVelocityDiT(hidden_size=2),
            x_0,
            last_hidden_state=hidden,
            acoustic_condition=acoustic_condition,
            mask=mask,
            num_steps=4,
            chunk_size=2,
        )

        self.assertTrue(torch.equal(diagonal.final, serial))
        self.assertEqual(diagonal.forward_count, 6)
        self.assertLess(
            diagonal.forward_count,
            serial_forward_count(frame_count=5, chunk_size=2, num_steps=4),
        )
        self.assertEqual(dit.forward_shapes[1], (2, 2, 2))
        self.assertTrue(torch.equal(diagonal.final[:, 3], torch.zeros((1, 2))))

    def test_orchestrator_exposes_diagonal_acoustic_sample(self) -> None:
        tokenizer = MockTokenizer()
        model = Orchestrator(
            qwen3=MockQwen(),
            dit=TrackingVelocityDiT(hidden_size=4),
            tokenizer=tokenizer,
            bpe_vocab_size=5,
            pretrained=False,
        )
        x_0 = torch.zeros((1, 4, 4))

        sample = model.diagonal_acoustic_sample(
            x_0,
            last_hidden_state=torch.ones_like(x_0),
            acoustic_condition=torch.zeros((1, 4)),
            mask=torch.ones((1, 4), dtype=torch.bool),
            num_steps=2,
            chunk_size=2,
        )

        self.assertEqual(tuple(sample.final.shape), (1, 4, 4))
        self.assertEqual(sample.forward_count, 3)


class TrackingVelocityDiT(nn.Module):
    def __init__(self, *, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.null_acoustic_condition = nn.Parameter(torch.zeros(1, hidden_size))
        self.forward_shapes: list[tuple[int, int, int]] = []

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        last_hidden_state: torch.Tensor,
        timesteps: torch.Tensor,
        acoustic_condition: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> BaseModelOutputWithPast:
        del acoustic_condition
        self.forward_shapes.append(tuple(x_t.shape))
        velocity = last_hidden_state + timesteps.reshape(-1, 1, 1).to(x_t.dtype)
        velocity = velocity * attention_mask.to(dtype=x_t.dtype).unsqueeze(-1)
        return BaseModelOutputWithPast(last_hidden_state=velocity)


def _serial_sample(
    dit: TrackingVelocityDiT,
    x_0: torch.Tensor,
    *,
    last_hidden_state: torch.Tensor,
    acoustic_condition: torch.Tensor,
    mask: torch.Tensor,
    num_steps: int,
    chunk_size: int,
) -> torch.Tensor:
    state = x_0.clone()
    time_grid = torch.linspace(0, 1, num_steps + 1)
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


if __name__ == "__main__":
    unittest.main()
