from __future__ import annotations

import pytest
import torch
from semantic_acoustic_codec.rl import SACRLAdapter

from speech_to_speech.rl import (
    S2SGenerationResult,
    S2SRLAdapter,
    S2SRewardBatch,
    S2SRollout,
)
from speech_to_speech.task import Task


def test_s2s_adapter_delegates_grpo_contract_to_sac() -> None:
    adapter = S2SRLAdapter(SACRLAdapter())
    rewards = S2SRewardBatch(torch.tensor([[1.0, 0.5], [0.0, 2.0]]))
    policy = torch.zeros(2, 2, 3)
    old = torch.full((2, 2, 3), -0.1)
    mask = torch.ones(2, 2, 3, dtype=torch.bool)

    batch = adapter.to_grpo_batch(
        rewards,
        policy_token_logps=policy,
        old_token_logps=old,
        response_mask=mask,
    )

    assert batch["policy_token_logps"] is policy
    assert batch["old_token_logps"] is old
    assert batch["rewards"] is rewards.rewards
    assert batch["response_mask"] is mask


def test_s2s_adapter_delegates_continuous_and_neighbor_contracts() -> None:
    adapter = S2SRLAdapter(SACRLAdapter())
    rewards = S2SRewardBatch(torch.ones(2, 3))
    steps = torch.zeros(2, 3, 4)
    step_mask = torch.ones(2, 3, 4, dtype=torch.bool)

    continuous = adapter.to_continuous_grpo_batch(
        rewards,
        policy_step_logps=steps,
        old_step_logps=steps - 0.1,
        step_mask=step_mask,
    )

    assert continuous["policy_step_logps"].shape == (2, 3, 4)
    assert continuous["rewards"].shape == (2, 3)

    neighbor = torch.zeros(2, 5, 4, 3)
    neighbor_mask = torch.ones(2, 5, 4, 3, dtype=torch.bool)
    neighbor_batch = adapter.to_neighbor_grpo_batch(
        rewards,
        policy_neighbor_logps=neighbor,
        old_neighbor_logps=neighbor - 0.1,
        neighbor_mask=neighbor_mask,
    )

    assert neighbor_batch["policy_neighbor_logps"].shape == (2, 5, 4, 3)
    assert neighbor_batch["rewards"].shape == (2, 3)


def test_s2s_score_requires_external_rewards() -> None:
    adapter = S2SRLAdapter(SACRLAdapter())
    result = S2SGenerationResult(
        sample_id=0,
        group_id=0,
        candidate_id=0,
        request={
            "prompt_ids": torch.tensor([1, 2], dtype=torch.long),
            "task": Task.TTS,
            "audio_input_positions": None,
        },
        response_ids=torch.tensor([3], dtype=torch.long),
    )
    rollout = S2SRollout((result,), batch_size=1, group_size=1)

    with pytest.raises(ValueError, match="task-specific"):
        adapter.score(rollout)
