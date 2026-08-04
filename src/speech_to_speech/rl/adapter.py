from __future__ import annotations

from typing import Mapping

from anytrain.framework.rl import ContinuousRolloutBatch, NeighborRolloutBatch, RolloutBatch
from semantic_acoustic_generator.rl import GeneratorRLAdapter, GeneratorRewardBatch, GeneratorRollout
from torch import Tensor

from .types import S2SRewardBatch, S2SRollout


class S2SRLAdapter:
    """S2S-owned RL adapter that delegates acoustic-generator tensors to the plugin.

    Speech-to-speech keeps request/result/reward semantics here. The wrapped generator
    adapter converts only tensor-level policy data into anytrain RL contracts.
    """

    def __init__(self, sac_adapter: GeneratorRLAdapter) -> None:
        self.sac_adapter = sac_adapter

    def from_sac_rollout(self, rollout: GeneratorRollout) -> GeneratorRollout:
        """Expose the generator substrate for S2S callers that build results separately."""

        return rollout

    def score(
        self,
        rollout: S2SRollout,
        *,
        rewards: Tensor | None = None,
        group_mask: Tensor | None = None,
        components: Mapping[str, Tensor] | None = None,
    ) -> S2SRewardBatch:
        if rewards is None:
            raise ValueError("S2S reward scoring is task-specific; pass rewards explicitly.")
        if rewards.shape != (rollout.batch_size, rollout.group_size):
            raise ValueError("rewards must have shape [batch, group].")
        return S2SRewardBatch(
            rewards=rewards,
            group_mask=group_mask,
            components={} if components is None else components,
        )

    def to_sac_rewards(self, reward_batch: S2SRewardBatch) -> GeneratorRewardBatch:
        return GeneratorRewardBatch(
            rewards=reward_batch.rewards,
            group_mask=reward_batch.group_mask,
            components=reward_batch.components,
        )

    def to_grpo_batch(
        self,
        reward_batch: S2SRewardBatch,
        *,
        policy_token_logps: Tensor,
        old_token_logps: Tensor,
        response_mask: Tensor,
        ref_token_logps: Tensor | None = None,
        group_mask: Tensor | None = None,
    ) -> RolloutBatch:
        return self.sac_adapter.to_grpo_batch(
            self.to_sac_rewards(reward_batch),
            policy_token_logps=policy_token_logps,
            old_token_logps=old_token_logps,
            response_mask=response_mask,
            ref_token_logps=ref_token_logps,
            group_mask=group_mask,
        )

    def to_continuous_grpo_batch(
        self,
        reward_batch: S2SRewardBatch,
        *,
        policy_step_logps: Tensor,
        old_step_logps: Tensor,
        step_mask: Tensor,
        kl_values: Tensor | None = None,
        group_mask: Tensor | None = None,
    ) -> ContinuousRolloutBatch:
        return self.sac_adapter.to_continuous_grpo_batch(
            self.to_sac_rewards(reward_batch),
            policy_step_logps=policy_step_logps,
            old_step_logps=old_step_logps,
            step_mask=step_mask,
            kl_values=kl_values,
            group_mask=group_mask,
        )

    def to_neighbor_grpo_batch(
        self,
        reward_batch: S2SRewardBatch,
        *,
        policy_neighbor_logps: Tensor,
        old_neighbor_logps: Tensor,
        neighbor_mask: Tensor,
        kl_values: Tensor | None = None,
        group_mask: Tensor | None = None,
        anchor_mask: Tensor | None = None,
    ) -> NeighborRolloutBatch:
        return self.sac_adapter.to_neighbor_grpo_batch(
            self.to_sac_rewards(reward_batch),
            policy_neighbor_logps=policy_neighbor_logps,
            old_neighbor_logps=old_neighbor_logps,
            neighbor_mask=neighbor_mask,
            kl_values=kl_values,
            group_mask=group_mask,
            anchor_mask=anchor_mask,
        )


__all__ = ["S2SRLAdapter"]
