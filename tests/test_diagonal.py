from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from speech_to_speech.model.diagonal import (
    causal_window_flow_sample,
    diagonal_flow_sample,
    diagonal_flow_sample_chunks,
    diagonal_schedule,
    diagonal_schedule_from_lengths,
    full_sequence_flow_sample,
    serial_forward_count,
    serial_flow_sample,
    serial_flow_sample_chunks,
)
from speech_to_speech.model.DiT.model import DiT
from speech_to_speech.model.qwen3 import Qwen3Config
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
        serial = serial_flow_sample(
            TrackingVelocityDiT(hidden_size=2),
            x_0,
            last_hidden_state=hidden,
            acoustic_condition=acoustic_condition,
            mask=mask,
            num_steps=4,
            chunk_size=2,
        )

        self.assertTrue(torch.equal(diagonal.final, serial.final))
        self.assertEqual(serial.forward_count, 12)
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

    def test_diagonal_sample_uses_cfg_velocity(self) -> None:
        dit = ConditionVelocityDiT(hidden_size=2)

        sample = diagonal_flow_sample(
            dit,
            torch.zeros((1, 2, 2)),
            last_hidden_state=torch.zeros((1, 2, 2)),
            acoustic_condition=torch.tensor([[3.0, 5.0]]),
            mask=torch.ones((1, 2), dtype=torch.bool),
            num_steps=1,
            chunk_size=2,
            guidance_scale=2.0,
        )

        self.assertTrue(torch.equal(sample.final, torch.full((1, 2, 2), 16.0)))
        self.assertEqual(dit.conditions, [[3.0, 5.0], [0.0, 0.0]])

    def test_schedule_from_lengths_uses_nonuniform_chunks(self) -> None:
        schedule = diagonal_schedule_from_lengths(
            chunk_lengths=(3, 4, 2),
            num_steps=2,
        )

        self.assertEqual(
            [[(cell.chunk_index, cell.step_index, cell.start, cell.end) for cell in batch.cells]
             for batch in schedule],
            [
                [(0, 0, 0, 3)],
                [(0, 1, 0, 3), (1, 0, 3, 7)],
                [(1, 1, 3, 7), (2, 0, 7, 9)],
                [(2, 1, 7, 9)],
            ],
        )

    def test_diagonal_sample_chunks_matches_serial_chunks(self) -> None:
        x_0 = torch.zeros((1, 9, 2))
        hidden = torch.ones_like(x_0)
        acoustic_condition = torch.zeros((1, 2))
        mask = torch.ones((1, 9), dtype=torch.bool)

        diagonal = diagonal_flow_sample_chunks(
            TrackingVelocityDiT(hidden_size=2),
            x_0,
            chunk_lengths=(3, 4, 2),
            last_hidden_state=hidden,
            acoustic_condition=acoustic_condition,
            mask=mask,
            num_steps=3,
        )
        serial = serial_flow_sample_chunks(
            TrackingVelocityDiT(hidden_size=2),
            x_0,
            chunk_lengths=(3, 4, 2),
            last_hidden_state=hidden,
            acoustic_condition=acoustic_condition,
            mask=mask,
            num_steps=3,
        )

        self.assertTrue(torch.equal(diagonal.final, serial.final))
        self.assertEqual(diagonal.forward_count, 5)
        self.assertEqual(serial.forward_count, 9)

    def test_full_sequence_sample_runs_one_forward_per_step(self) -> None:
        dit = TrackingVelocityDiT(hidden_size=2)

        sample = full_sequence_flow_sample(
            dit,
            torch.zeros((1, 9, 2)),
            last_hidden_state=torch.ones((1, 9, 2)),
            acoustic_condition=torch.zeros((1, 2)),
            mask=torch.ones((1, 9), dtype=torch.bool),
            num_steps=3,
        )

        self.assertEqual(tuple(sample.final.shape), (1, 9, 2))
        self.assertEqual(sample.forward_count, 3)
        self.assertEqual(dit.forward_shapes, [(1, 9, 2)] * 3)

    def test_real_dit_handles_packed_nonuniform_global_positions(self) -> None:
        config = Qwen3Config()
        config.hidden_size = 8
        config.intermediate_size = 16
        config.num_hidden_layers = 1
        config.num_attention_heads = 2
        config.num_key_value_heads = 2
        dit = DiT(config).eval()

        sample = diagonal_flow_sample_chunks(
            dit,
            torch.zeros((1, 6, 8)),
            chunk_lengths=(2, 3, 1),
            last_hidden_state=torch.ones((1, 6, 8)),
            acoustic_condition=torch.zeros((1, 8)),
            mask=torch.ones((1, 6), dtype=torch.bool),
            num_steps=2,
        )

        self.assertEqual(tuple(sample.final.shape), (1, 6, 8))
        self.assertEqual(sample.forward_count, 4)

    def test_causal_window_sample_uses_left_context(self) -> None:
        dit = TrackingVelocityDiT(hidden_size=2)

        sample = causal_window_flow_sample(
            dit,
            torch.zeros((1, 5, 2)),
            last_hidden_state=torch.ones((1, 5, 2)),
            acoustic_condition=torch.zeros((1, 2)),
            mask=torch.ones((1, 5), dtype=torch.bool),
            num_steps=1,
            chunk_size=2,
            left_context_chunks=1,
        )

        self.assertEqual(tuple(sample.final.shape), (1, 5, 2))
        self.assertEqual(sample.forward_count, 3)
        self.assertEqual(
            dit.forward_shapes,
            [
                (1, 2, 2),
                (1, 4, 2),
                (1, 3, 2),
            ],
        )

    def test_causal_window_sample_uses_step_snapshot(self) -> None:
        dit = StateSumDiT(hidden_size=1)

        sample = causal_window_flow_sample(
            dit,
            torch.zeros((1, 2, 1)),
            last_hidden_state=torch.zeros((1, 2, 1)),
            acoustic_condition=torch.zeros((1, 1)),
            mask=torch.ones((1, 2), dtype=torch.bool),
            num_steps=1,
            chunk_size=1,
            left_context_chunks=1,
        )

        self.assertEqual(sample.final.tolist(), [[[1.0], [1.0]]])


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
        position_ids: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPast:
        del acoustic_condition, position_ids
        self.forward_shapes.append(tuple(x_t.shape))
        velocity = last_hidden_state + timesteps.reshape(-1, 1, 1).to(x_t.dtype)
        velocity = velocity * attention_mask.to(dtype=x_t.dtype).unsqueeze(-1)
        return BaseModelOutputWithPast(last_hidden_state=velocity)


class ConditionVelocityDiT(nn.Module):
    def __init__(self, *, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.null_acoustic_condition = nn.Parameter(torch.zeros(1, hidden_size))
        self.conditions: list[list[float]] = []

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        last_hidden_state: torch.Tensor,
        timesteps: torch.Tensor,
        acoustic_condition: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPast:
        del last_hidden_state, timesteps, attention_mask, position_ids
        self.conditions.append(
            [float(value) for value in acoustic_condition.detach().reshape(-1)]
        )
        value = acoustic_condition.sum(dim=-1).reshape(-1, 1, 1)
        return BaseModelOutputWithPast(last_hidden_state=x_t + value)


class StateSumDiT(nn.Module):
    def __init__(self, *, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.null_acoustic_condition = nn.Parameter(torch.zeros(1, hidden_size))

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        last_hidden_state: torch.Tensor,
        timesteps: torch.Tensor,
        acoustic_condition: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPast:
        del last_hidden_state, timesteps, acoustic_condition, attention_mask, position_ids
        value = x_t.sum(dim=(1, 2), keepdim=True) + 1.0
        return BaseModelOutputWithPast(last_hidden_state=value.expand_as(x_t))


if __name__ == "__main__":
    unittest.main()
