from __future__ import annotations

import unittest

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from speech_to_speech.model.acoustic import (
    acoustic_velocity,
    acoustic_features_from_batch_side,
    pooled_acoustic_condition_from_batch_side,
)
from speech_to_speech.types import LongCatBatchSide


class AcousticFeatureBoundaryTest(unittest.TestCase):
    def test_acoustic_features_from_batch_side_uses_batched_codes_and_mask(self) -> None:
        side = LongCatBatchSide(
            semantic_ids=torch.tensor([[1, 2, 0]]),
            semantic_mask=torch.tensor([[True, True, False]]),
            acoustic_ids=torch.tensor([[[4, 5, 0], [6, 7, 0]]]),
            acoustic_mask=torch.tensor([[True, True, False]]),
        )
        extractor = FakeFeatureExtractor()

        features, mask = acoustic_features_from_batch_side(
            side,
            feature_extractor=extractor,
        )

        self.assertTrue(torch.equal(extractor.acoustic_ids, side.acoustic_ids))
        self.assertTrue(torch.equal(features, extractor.features))
        self.assertTrue(torch.equal(mask, side.acoustic_mask))

    def test_acoustic_features_from_batch_side_checks_feature_mask_alignment(self) -> None:
        side = LongCatBatchSide(
            semantic_ids=torch.tensor([[1, 2]]),
            semantic_mask=torch.tensor([[True, True]]),
            acoustic_ids=torch.tensor([[[4, 5], [6, 7]]]),
            acoustic_mask=torch.tensor([[True, True]]),
        )

        with self.assertRaisesRegex(ValueError, "align with acoustic_mask"):
            acoustic_features_from_batch_side(
                side,
                feature_extractor=FakeFeatureExtractor(time=3),
            )

    def test_pooled_acoustic_condition_from_batch_side_uses_masked_mean(self) -> None:
        side = LongCatBatchSide(
            semantic_ids=torch.tensor([[1, 2, 0], [3, 0, 0]]),
            semantic_mask=torch.tensor([[True, True, False], [True, False, False]]),
            acoustic_ids=torch.zeros((2, 2, 3), dtype=torch.long),
            acoustic_mask=torch.tensor([[True, True, False], [True, False, False]]),
        )
        features = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]],
                [[5.0, 6.0], [9.0, 9.0], [9.0, 9.0]],
            ]
        )

        condition = pooled_acoustic_condition_from_batch_side(
            side,
            feature_extractor=FakeFeatureExtractor(features=features),
        )

        self.assertTrue(torch.equal(condition, torch.tensor([[2.0, 3.0], [5.0, 6.0]])))

    def test_pooled_acoustic_condition_uses_empty_condition_for_missing_rows(self) -> None:
        side = LongCatBatchSide(
            semantic_ids=torch.tensor([[0, 0], [1, 0]]),
            semantic_mask=torch.tensor([[False, False], [True, False]]),
            acoustic_ids=torch.zeros((2, 2, 2), dtype=torch.long),
            acoustic_mask=torch.tensor([[False, False], [True, False]]),
        )
        features = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[3.0, 4.0], [9.0, 9.0]],
            ]
        )

        condition = pooled_acoustic_condition_from_batch_side(
            side,
            feature_extractor=FakeFeatureExtractor(features=features),
            empty_condition=torch.tensor([[7.0, 8.0], [0.0, 0.0]]),
        )

        self.assertTrue(torch.equal(condition, torch.tensor([[7.0, 8.0], [3.0, 4.0]])))

    def test_acoustic_velocity_applies_classifier_free_guidance(self) -> None:
        dit = GuidanceDiT()

        velocity = acoustic_velocity(
            dit,
            x_t=torch.zeros((1, 2, 2)),
            timesteps=torch.tensor([0.5]),
            last_hidden_state=torch.zeros((1, 2, 2)),
            acoustic_condition=torch.tensor([[3.0, 5.0]]),
            mask=torch.tensor([[True, True]]),
            guidance_scale=2.0,
        )

        self.assertTrue(torch.equal(velocity, torch.full((1, 2, 2), 14.0)))
        self.assertEqual(dit.conditions, [[3.0, 5.0], [1.0, 1.0]])

    def test_acoustic_velocity_skips_unconditional_forward_at_unit_guidance(self) -> None:
        dit = GuidanceDiT()

        acoustic_velocity(
            dit,
            x_t=torch.zeros((1, 2, 2)),
            timesteps=torch.tensor([0.5]),
            last_hidden_state=torch.zeros((1, 2, 2)),
            acoustic_condition=torch.tensor([[3.0, 5.0]]),
            mask=torch.tensor([[True, True]]),
        )

        self.assertEqual(dit.conditions, [[3.0, 5.0]])


class FakeFeatureExtractor:
    def __init__(self, *, features: torch.Tensor | None = None, time: int = 3) -> None:
        self.features = features if features is not None else torch.ones((1, time, 4))
        self.acoustic_ids: torch.Tensor | None = None

    def acoustic_codes_to_features(self, acoustic_ids: torch.Tensor) -> torch.Tensor:
        self.acoustic_ids = acoustic_ids
        return self.features


class GuidanceDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.null_acoustic_condition = nn.Parameter(torch.ones((1, 2)))
        self.conditions: list[list[float]] = []

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        last_hidden_state: torch.Tensor,
        timesteps: torch.Tensor,
        acoustic_condition: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> BaseModelOutputWithPast:
        del last_hidden_state, timesteps, attention_mask
        self.conditions.append(
            [float(value) for value in acoustic_condition.detach().reshape(-1)]
        )
        value = acoustic_condition.sum(dim=-1).reshape(-1, 1, 1)
        return BaseModelOutputWithPast(last_hidden_state=x_t + value)


if __name__ == "__main__":
    unittest.main()
