"""DiT acoustic feature generator used by waveform generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Protocol

import torch
from torch import Tensor, nn

from .._module import module_dtype
from .condition import null_acoustic_condition
from .diagonal import (
    causal_window_flow_sample,
    diagonal_flow_sample,
    diagonal_flow_sample_chunks,
    full_sequence_flow_sample,
)
from ..types import AcousticCondition


class AcousticSampler(StrEnum):
    SERIAL = auto()
    DIAGONAL = auto()
    DIAGONAL_BPE = auto()
    CAUSAL_WINDOW = auto()


class DiTAcousticModel(Protocol):
    dit: nn.Module | None
    acoustic_condition_adapter: nn.Module

    def acoustic_condition_hidden(
        self,
        condition: AcousticCondition,
        *,
        dtype: torch.dtype | None = None,
    ) -> Tensor: ...


@dataclass(frozen=True)
class DiTAcousticFeatureGenerator:
    model: DiTAcousticModel
    num_steps: int
    chunk_size: int | None = None
    left_context_chunks: int | None = None
    guidance_scale: float = 1.0
    sampler: AcousticSampler = AcousticSampler.SERIAL
    acoustic_condition: Tensor | None = None

    @torch.no_grad()
    def __call__(self, condition: AcousticCondition) -> Tensor:
        if self.model.dit is None:
            raise RuntimeError("DiT acoustic feature generation requires a DiT decoder.")
        hidden = condition.hidden_states
        flow_dtype = module_dtype(self.model.dit, hidden.dtype)
        hidden = self.model.acoustic_condition_hidden(condition, dtype=flow_dtype)
        adapter = self.model.acoustic_condition_adapter
        adapter_dtype = module_dtype(adapter, condition.hidden_states.dtype)
        initial = _acoustic_flow_source_sample_like(hidden)
        if self.acoustic_condition is None:
            acoustic_condition = null_acoustic_condition(self.model.dit, initial)
        else:
            acoustic_condition = self.acoustic_condition.to(
                device=hidden.device,
                dtype=adapter_dtype,
            )
            if acoustic_condition.size(-1) != hidden.size(-1):
                acoustic_condition = adapter(acoustic_condition).to(dtype=flow_dtype)
            else:
                acoustic_condition = acoustic_condition.to(dtype=flow_dtype)
            if acoustic_condition.shape != (hidden.size(0), hidden.size(-1)):
                raise ValueError(
                    "acoustic_condition must have shape [batch, acoustic feature dim]."
                )
        chunk_size = self.chunk_size or hidden.size(1)
        sample_kwargs = {
            "last_hidden_state": hidden,
            "acoustic_condition": acoustic_condition,
            "mask": condition.mask.to(device=hidden.device, dtype=torch.bool),
            "num_steps": self.num_steps,
            "chunk_size": chunk_size,
            "guidance_scale": self.guidance_scale,
        }
        match self.sampler:
            case AcousticSampler.SERIAL:
                sample = full_sequence_flow_sample(
                    self.model.dit,
                    initial,
                    last_hidden_state=hidden,
                    acoustic_condition=acoustic_condition,
                    mask=condition.mask.to(device=hidden.device, dtype=torch.bool),
                    num_steps=self.num_steps,
                    guidance_scale=self.guidance_scale,
                )
            case AcousticSampler.DIAGONAL:
                sample = diagonal_flow_sample(self.model.dit, initial, **sample_kwargs)
            case AcousticSampler.DIAGONAL_BPE:
                chunk_lengths = single_chunk_lengths(condition)
                active_frames = sum(chunk_lengths)
                sample = diagonal_flow_sample_chunks(
                    self.model.dit,
                    initial[:, :active_frames],
                    chunk_lengths=chunk_lengths,
                    last_hidden_state=hidden[:, :active_frames],
                    acoustic_condition=acoustic_condition,
                    mask=condition.mask[:, :active_frames].to(
                        device=hidden.device,
                        dtype=torch.bool,
                    ),
                    num_steps=self.num_steps,
                    guidance_scale=self.guidance_scale,
                )
                final = initial.clone()
                final[:, :active_frames] = sample.final
                return final
            case AcousticSampler.CAUSAL_WINDOW:
                sample = causal_window_flow_sample(
                    self.model.dit,
                    initial,
                    left_context_chunks=left_context_chunks(
                        self.left_context_chunks,
                        frame_count=hidden.size(1),
                        chunk_size=chunk_size,
                    ),
                    **sample_kwargs,
                )
            case _:
                raise ValueError(f"unsupported acoustic sampler: {self.sampler}")
        return sample.final


def left_context_chunks(
    value: int | None,
    *,
    frame_count: int,
    chunk_size: int,
) -> int:
    if value is not None:
        return value
    return max(0, (frame_count + chunk_size - 1) // chunk_size - 1)


def single_chunk_lengths(condition: AcousticCondition) -> tuple[int, ...]:
    lengths = condition.chunk_lengths
    if lengths is None:
        raise ValueError("BPE diagonal acoustic generation requires condition chunk_lengths.")
    if len(lengths) != 1:
        raise ValueError("BPE diagonal acoustic generation currently requires batch size 1.")
    if sum(lengths[0]) != int(condition.mask[0].sum().detach().cpu()):
        raise ValueError("condition chunk_lengths must sum to active frame count.")
    if bool(condition.mask[0, : sum(lengths[0])].logical_not().any()):
        raise ValueError("BPE diagonal acoustic generation requires contiguous active frames.")
    return lengths[0]


def _acoustic_flow_source_sample_like(target_features: Tensor) -> Tensor:
    from speech_to_speech.model import acoustic

    return acoustic.acoustic_flow_source_sample_like(target_features)
