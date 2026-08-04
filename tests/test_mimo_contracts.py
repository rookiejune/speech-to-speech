from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F

from speech_to_speech.datamodule.mimo import MimoBatch, MimoSample, collate_mimo
from speech_to_speech.datamodule.config import DataLoaderConfig
from speech_to_speech.datamodule.mimo import MimoDataModule
from speech_to_speech.loss.mimo import MimoObjective
from speech_to_speech.runtime.backbone import (
    BackboneReadout,
    DualStreamBodyAdapter,
    DualStreamHiddenStates,
    fuse_dual_embeddings,
    shared_dual_hidden_states,
)


class MimoBatchContractTest(unittest.TestCase):
    def test_transfer_reuses_validated_batch_contract(self) -> None:
        batch = MimoBatch.from_samples(
            [
                MimoSample(
                    torch.tensor([1, 2]),
                    torch.tensor([3, 4]),
                    torch.tensor([-100, 1]),
                    torch.tensor([-100, 1]),
                )
            ],
            text_pad_token_id=99,
            audio_pad_token_id=98,
        )

        with patch(
            "speech_to_speech.datamodule.mimo.batch._validate_token_matrix",
            side_effect=AssertionError("trusted transfer must not revalidate"),
        ):
            moved = batch.to(torch.device("cpu"), non_blocking=True)

        self.assertIsInstance(moved, MimoBatch)
        torch.testing.assert_close(moved.text_input_ids, batch.text_input_ids)
        for actual, expected in zip(
            moved.supervised_token_counts,
            batch.supervised_token_counts,
        ):
            torch.testing.assert_close(actual, expected)

    def test_collate_pads_streams_and_keeps_feature_mask_explicit(self) -> None:
        first = MimoSample(
            text_input_ids=torch.tensor([1, 2, 3, 4]),
            audio_input_ids=torch.tensor([5, 6, 7, 8]),
            text_labels=torch.tensor([-100, 2, 3, 4]),
            audio_labels=torch.tensor([-100, -100, 7, 8]),
            text_loss_mask=torch.tensor([False, True, True, True]),
            audio_loss_mask=torch.tensor([False, False, True, True]),
            audio_features=torch.ones(4, 2),
            audio_feature_mask=torch.tensor([True, True, False, False]),
            task_id="audio_to_text",
            recording_id="rec-a",
        )
        second = MimoSample(
            text_input_ids=torch.tensor([9, 10, 11]),
            audio_input_ids=torch.tensor([12, 13, 14]),
            text_labels=torch.tensor([-100, 10, 11]),
            audio_labels=torch.tensor([-100, 13, 14]),
            text_loss_mask=torch.tensor([False, True, True]),
            audio_loss_mask=torch.tensor([False, False, True]),
            task_id="text_only",
            recording_id="rec-b",
        )

        batch = collate_mimo(
            [first, second],
            text_pad_token_id=99,
            audio_pad_token_id=98,
        )

        self.assertIsInstance(batch, MimoBatch)
        self.assertEqual(batch.text_input_ids.shape, (2, 4))
        self.assertEqual(batch.audio_input_ids[1, -1].item(), 98)
        self.assertEqual(batch.text_input_ids[1, -1].item(), 99)
        if (
            batch.attention_mask is None
            or batch.text_loss_mask is None
            or batch.audio_loss_mask is None
            or batch.audio_features is None
            or batch.audio_feature_mask is None
        ):
            self.fail("MimoBatch did not normalize optional tensors")
        self.assertFalse(bool(batch.attention_mask[1, -1]))
        self.assertFalse(bool(batch.text_loss_mask[1, -1]))
        self.assertFalse(bool(batch.audio_loss_mask[1, -1]))
        self.assertEqual(batch.audio_features.shape, (2, 4, 2))
        self.assertTrue(
            torch.equal(batch.audio_feature_mask[0], torch.tensor([True, True, False, False]))
        )
        self.assertFalse(bool(batch.audio_feature_mask[1].any()))
        self.assertEqual(batch.task_ids, ("audio_to_text", "text_only"))
        self.assertEqual(batch.recording_ids, ("rec-a", "rec-b"))
        text_count, audio_count = batch.supervised_token_counts
        torch.testing.assert_close(text_count, torch.tensor([3, 2]))
        torch.testing.assert_close(audio_count, torch.tensor([2, 1]))

    def test_batch_rejects_feature_mask_without_features_and_masked_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature_mask requires"):
            MimoBatch(
                text_input_ids=torch.tensor([[1, 2]]),
                audio_input_ids=torch.tensor([[3, 4]]),
                text_labels=torch.tensor([[-100, 1]]),
                audio_labels=torch.tensor([[-100, 1]]),
                audio_feature_mask=torch.zeros(1, 2, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "cannot select ignore"):
            MimoBatch(
                text_input_ids=torch.tensor([[1, 2]]),
                audio_input_ids=torch.tensor([[3, 4]]),
                text_labels=torch.tensor([[-100, 1]]),
                audio_labels=torch.tensor([[-100, 1]]),
                text_loss_mask=torch.ones(1, 2, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "supervised target"):
            MimoBatch(
                text_input_ids=torch.tensor([[1]]),
                audio_input_ids=torch.tensor([[3]]),
                text_labels=torch.tensor([[1]]),
                audio_labels=torch.tensor([[-100]]),
            )

    def test_batch_rejects_continuous_features_on_audio_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "supervised audio target"):
            MimoSample(
                text_input_ids=torch.tensor([1, 2]),
                audio_input_ids=torch.tensor([3, 4]),
                text_labels=torch.tensor([-100, -100]),
                audio_labels=torch.tensor([-100, 4]),
                audio_features=torch.ones(2, 3),
                audio_feature_mask=torch.tensor([True, True]),
            )
        with self.assertRaisesRegex(ValueError, "supervised audio target"):
            MimoBatch(
                text_input_ids=torch.tensor([[1, 2]]),
                audio_input_ids=torch.tensor([[3, 4]]),
                text_labels=torch.tensor([[-100, -100]]),
                audio_labels=torch.tensor([[-100, 4]]),
                audio_features=torch.ones(1, 2, 3),
                audio_feature_mask=torch.tensor([[True, True]]),
            )

    def test_prepared_data_module_collates_mimo_samples(self) -> None:
        sample = MimoSample(
            text_input_ids=torch.tensor([1, 2]),
            audio_input_ids=torch.tensor([3, 4]),
            text_labels=torch.tensor([-100, 2]),
            audio_labels=torch.tensor([-100, 4]),
        )

        class Samples(Dataset[MimoSample]):
            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int) -> MimoSample:
                if index not in {0, 1}:
                    raise IndexError(index)
                return sample

        module = MimoDataModule(
            Samples(),
            dataloader=DataLoaderConfig(batch_size=2, num_workers=0),
            text_pad_token_id=0,
            audio_pad_token_id=0,
        )

        batch = next(iter(module.train_dataloader()))

        self.assertIsInstance(batch, MimoBatch)
        self.assertEqual(batch.batch_size, 2)


class MimoObjectiveContractTest(unittest.TestCase):
    def test_causal_loss_normalizes_text_and_audio_routes_separately(self) -> None:
        torch.manual_seed(3)
        text_logits = torch.randn(2, 4, 5, requires_grad=True)
        audio_logits = torch.randn(2, 4, 7, requires_grad=True)
        text_labels = torch.tensor(
            [[-100, 1, 2, -100], [-100, 3, -100, -100]],
        )
        audio_labels = torch.tensor(
            [[-100, 1, -100, 2], [-100, -100, 4, -100]],
        )
        text_mask = text_labels.ne(-100)
        audio_mask = audio_labels.ne(-100)

        objective = MimoObjective()
        item = objective(  # first logit predicts second label
            text_logits,
            audio_logits,
            text_labels,
            audio_labels,
            text_loss_mask=text_mask,
            audio_loss_mask=audio_mask,
        )

        text_target_mask = text_mask[:, 1:]
        audio_target_mask = audio_mask[:, 1:]
        text_target = text_labels[:, 1:].masked_fill(~text_target_mask, -100)
        audio_target = audio_labels[:, 1:].masked_fill(~audio_target_mask, -100)
        text_per_token = F.cross_entropy(
            text_logits[:, :-1].transpose(1, 2),
            text_target,
            reduction="none",
            ignore_index=-100,
        )
        audio_per_token = F.cross_entropy(
            audio_logits[:, :-1].transpose(1, 2),
            audio_target,
            reduction="none",
            ignore_index=-100,
        )
        text_expected = (text_per_token * text_target_mask).sum(1) / text_target_mask.sum(
            1
        ).clamp_min(1)
        audio_expected = (audio_per_token * audio_target_mask).sum(1) / audio_target_mask.sum(
            1
        ).clamp_min(1)
        torch.testing.assert_close(item.loss, text_expected + audio_expected)
        self.assertIsNotNone(item.details)
        details = item.details or {}
        torch.testing.assert_close(details["text_loss"], text_expected)
        torch.testing.assert_close(details["audio_loss"], audio_expected)
        torch.testing.assert_close(details["text_tokens"], torch.tensor([2.0, 1.0]))
        torch.testing.assert_close(details["audio_tokens"], torch.tensor([2.0, 1.0]))
        text_global = (text_expected * torch.tensor([2.0, 1.0])).sum() / 3
        audio_global = (audio_expected * torch.tensor([2.0, 1.0])).sum() / 3
        torch.testing.assert_close(objective.mean(item), text_global + audio_global)
        torch.testing.assert_close(
            objective.mean(item, distributed=True), text_global + audio_global
        )
        item.loss.mean().backward()
        self.assertIsNotNone(text_logits.grad)
        self.assertIsNotNone(audio_logits.grad)

    def test_from_hidden_states_and_batch_protocol(self) -> None:
        sample = MimoSample(
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6]),
            torch.tensor([-100, 1, 2]),
            torch.tensor([-100, 1, 2]),
            text_loss_mask=torch.tensor([False, True, True]),
            audio_loss_mask=torch.tensor([False, True, True]),
        )
        batch = MimoBatch.from_samples(
            [sample],
            text_pad_token_id=99,
            audio_pad_token_id=98,
        )
        text_hidden = torch.randn(1, 3, 4)
        audio_hidden = torch.randn(1, 3, 4)
        text_head = nn.Linear(4, 3)
        audio_head = nn.Linear(4, 3)
        objective = MimoObjective()
        expected = objective.from_hidden_states(
            text_hidden,
            audio_hidden,
            batch.text_labels,
            batch.audio_labels,
            text_readout=text_head,
            audio_readout=audio_head,
            text_loss_mask=batch.text_loss_mask,
            audio_loss_mask=batch.audio_loss_mask,
        )

        class Model:
            def dual_hidden_states(self, value: MimoBatch) -> DualStreamHiddenStates:
                del value
                return DualStreamHiddenStates(text_hidden, audio_hidden)

            def dual_logits(
                self, hidden: DualStreamHiddenStates
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return text_head(hidden.text), audio_head(hidden.audio)

        actual = objective.from_batch(batch, Model())
        torch.testing.assert_close(actual.loss, expected.loss)


class DualStreamBackboneContractTest(unittest.TestCase):
    def test_fusion_masks_continuous_features(self) -> None:
        text = torch.ones(1, 2, 3)
        audio = torch.full((1, 2, 3), 2.0)
        features = torch.full((1, 2, 3), 4.0)
        mask = torch.tensor([[True, False]])
        fused = fuse_dual_embeddings(
            text,
            audio,
            audio_features=features,
            audio_feature_mask=mask,
        )
        torch.testing.assert_close(fused, torch.tensor([[[7.0, 7.0, 7.0], [3.0, 3.0, 3.0]]]))

    def test_body_adapter_calls_body_once_and_reads_two_branches(self) -> None:
        calls: list[torch.Tensor] = []

        def body(**kwargs: object) -> SimpleNamespace:
            inputs = kwargs["inputs_embeds"]
            if not isinstance(inputs, torch.Tensor):
                raise TypeError("test body received non-tensor inputs")
            calls.append(inputs)
            return SimpleNamespace(
                last_hidden_state=(inputs + 1, inputs + 2),
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )

        adapter = DualStreamBodyAdapter(
            body,
            text_readout=BackboneReadout("last_hidden_state[0]"),
            audio_readout=BackboneReadout("last_hidden_state[1]"),
        )
        output = adapter.encode_dual(
            text_inputs_embeds=torch.ones(1, 2, 3),
            audio_inputs_embeds=torch.full((1, 2, 3), 2.0),
        )
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(output.text, torch.full((1, 2, 3), 4.0))
        torch.testing.assert_close(output.audio, torch.full((1, 2, 3), 5.0))

    def test_hidden_container_rejects_unaligned_streams(self) -> None:
        with self.assertRaisesRegex(ValueError, "align"):
            DualStreamHiddenStates(torch.zeros(1, 2, 3), torch.zeros(1, 3, 3))
        shared = shared_dual_hidden_states(torch.zeros(1, 2, 3))
        self.assertIs(shared.text, shared.audio)
        self.assertIs(shared.shared, shared.text)


if __name__ == "__main__":
    unittest.main()
