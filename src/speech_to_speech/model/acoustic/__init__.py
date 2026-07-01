"""Public acoustic model helpers.

This package exposes the acoustic training and generation boundary while
keeping condition, flow, and scheduling details in local implementation files.
"""

from __future__ import annotations

from .condition import (
    AcousticFlowLossStats,
    acoustic_condition,
    acoustic_condition_from_target_audio_embedding,
    acoustic_features_from_batch_side,
    acoustic_velocity,
    continuous_flow_loss,
    continuous_flow_loss_stats,
    null_acoustic_condition,
    pooled_acoustic_condition_from_batch_side,
    validate_acoustic_features,
)
from .condition_encoder import ConditionEncoder, condition_encoder_config
from .diagonal import (
    CausalWindowSample,
    DiagonalBatch,
    DiagonalCell,
    DiagonalSample,
    FullSequenceSample,
    SerialSample,
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
from .flow import (
    AcousticFlowSample,
    acoustic_flow_source_sample_like,
    full_sequence_acoustic_flow_sample,
)
from .generation import (
    AcousticSampler,
    DiTAcousticFeatureGenerator,
    left_context_chunks,
    single_chunk_lengths,
)

__all__ = [
    "AcousticFlowSample",
    "AcousticFlowLossStats",
    "AcousticSampler",
    "CausalWindowSample",
    "ConditionEncoder",
    "DiagonalBatch",
    "DiagonalCell",
    "DiagonalSample",
    "DiTAcousticFeatureGenerator",
    "FullSequenceSample",
    "SerialSample",
    "acoustic_condition",
    "acoustic_condition_from_target_audio_embedding",
    "acoustic_flow_source_sample_like",
    "acoustic_features_from_batch_side",
    "acoustic_velocity",
    "causal_window_flow_sample",
    "condition_encoder_config",
    "continuous_flow_loss",
    "continuous_flow_loss_stats",
    "diagonal_flow_sample",
    "diagonal_flow_sample_chunks",
    "diagonal_schedule",
    "diagonal_schedule_from_lengths",
    "full_sequence_acoustic_flow_sample",
    "full_sequence_flow_sample",
    "left_context_chunks",
    "null_acoustic_condition",
    "pooled_acoustic_condition_from_batch_side",
    "serial_flow_sample",
    "serial_flow_sample_chunks",
    "serial_forward_count",
    "single_chunk_lengths",
    "validate_acoustic_features",
]
