from __future__ import annotations

import unittest

import torch

from speech_to_speech.loss import CausalAcousticLoss
from speech_to_speech.model.acoustic import AcousticRVQDecoder


class AcousticRVQTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = AcousticRVQDecoder(
            condition_dim=6,
            codebooks=3,
            codebook_size=(5, 6, 7),
            hidden_dim=8,
            layers=2,
            heads=2,
            ffn_ratio=2,
        ).eval()

    def test_shapes_and_backward(self):
        condition = torch.randn(2, 4, 6, requires_grad=True)
        labels = torch.stack(
            (
                torch.randint(0, 5, (2, 4)),
                torch.randint(0, 6, (2, 4)),
                torch.randint(0, 7, (2, 4)),
            ),
            dim=-1,
        )

        logits = self.model(condition, labels)
        sum(value.square().mean() for value in logits).backward()

        self.assertEqual(
            [value.shape for value in logits], [(2, 4, 5), (2, 4, 6), (2, 4, 7)]
        )
        self.assertIsNotNone(condition.grad)

    def test_future_codebook_does_not_change_previous_logits(self):
        condition = torch.randn(1, 2, 6)
        labels = torch.tensor([[[1, 2, 3], [2, 3, 4]]])
        changed = labels.clone()
        changed[..., 1] = (changed[..., 1] + 1) % 6

        baseline = self.model(condition, labels)
        future_changed = self.model(condition, changed)

        torch.testing.assert_close(baseline[0], future_changed[0])
        torch.testing.assert_close(baseline[1], future_changed[1])
        self.assertFalse(torch.equal(baseline[2], future_changed[2]))

    def test_training_packs_valid_frames_and_zeros_padding_logits(self):
        condition = torch.randn(2, 3, 6, requires_grad=True)
        labels = torch.stack(
            (
                torch.randint(0, 5, (2, 3)),
                torch.randint(0, 6, (2, 3)),
                torch.randint(0, 7, (2, 3)),
            ),
            dim=-1,
        )
        mask = torch.tensor([[True, True, False], [True, False, False]])
        labels[~mask] = -1
        batches = []
        handle = self.model.decoder.register_forward_pre_hook(
            lambda _module, args, kwargs: batches.append(
                kwargs["inputs_embeds"].size(0)
            ),
            with_kwargs=True,
        )
        try:
            logits = self.model(condition, labels, mask=mask)
        finally:
            handle.remove()

        self.assertEqual(batches, [int(mask.sum())])
        for value in logits:
            self.assertTrue(torch.equal(value[~mask], torch.zeros_like(value[~mask])))
        sum(value[mask].square().mean() for value in logits).backward()
        self.assertIsNotNone(condition.grad)
        self.assertTrue(
            torch.equal(condition.grad[~mask], torch.zeros_like(condition.grad[~mask]))
        )

    def test_default_depth_is_eight_qwen_layers(self):
        model = AcousticRVQDecoder(
            condition_dim=8,
            codebooks=2,
            codebook_size=4,
            heads=2,
        )

        self.assertEqual(len(model.decoder.layers), 8)

    def test_structurally_unused_embeddings_are_frozen(self):
        self.assertFalse(self.model.decoder.embed_tokens.weight.requires_grad)
        self.assertFalse(self.model.codebook_embeddings[-1].weight.requires_grad)
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in self.model.embedding_projections[-1].parameters()
            )
        )
        self.assertTrue(self.model.codebook_embeddings[-2].weight.requires_grad)

    def test_generation_uses_one_cached_token_per_codebook(self):
        lengths = []
        handle = self.model.decoder.register_forward_pre_hook(
            lambda _module, args, kwargs: lengths.append(
                kwargs["inputs_embeds"].size(1)
            ),
            with_kwargs=True,
        )
        try:
            output = self.model.generate(torch.randn(2, 4, 6))
        finally:
            handle.remove()

        self.assertEqual(output.shape, (2, 4, 3))
        self.assertEqual(lengths, [1, 1, 1])

    def test_generation_packs_frames_without_consuming_padding_rng(self):
        condition = torch.randn(2, 3, 6)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        batches = []
        handle = self.model.decoder.register_forward_pre_hook(
            lambda _module, args, kwargs: batches.append(
                kwargs["inputs_embeds"].size(0)
            ),
            with_kwargs=True,
        )
        try:
            padded = self.model.generate(
                condition,
                mask=mask,
                generator=torch.Generator().manual_seed(9),
            )
        finally:
            handle.remove()
        packed = self.model.generate(
            condition[mask][None],
            generator=torch.Generator().manual_seed(9),
        )

        self.assertEqual(batches, [int(mask.sum())] * self.model.codebooks)
        self.assertTrue(torch.equal(padded[mask], packed[0]))
        self.assertTrue(torch.equal(padded[~mask], torch.zeros_like(padded[~mask])))

    def test_training_and_generation_require_a_valid_frame_in_every_row(self):
        condition = torch.randn(2, 2, 6)
        mask = torch.tensor([[True, False], [False, False]])
        calls = (
            lambda: self.model(condition, mask=mask),
            lambda: self.model.generate(condition, mask=mask),
            lambda: self.model(torch.empty(1, 0, 6)),
            lambda: self.model.generate(torch.empty(1, 0, 6)),
        )

        for call in calls:
            with (
                self.subTest(call=call),
                self.assertRaisesRegex(ValueError, "each acoustic condition row"),
            ):
                call()

    def test_training_rejects_invalid_target_code_dtype_and_range(self):
        condition = torch.randn(1, 1, 6)

        with self.assertRaisesRegex(TypeError, "signed integer"):
            self.model(condition, torch.zeros(1, 1, 3))
        with self.assertRaisesRegex(ValueError, "outside its codebook"):
            self.model(condition, torch.tensor([[[0, 0, 7]]]))

    def test_causal_loss_ignores_padding_frames(self):
        labels = torch.tensor([[[1, 2], [2, 1], [-1, -1]]])
        mask = torch.tensor([[True, True, False]])
        logits = (
            torch.randn(1, 3, 4),
            torch.randn(1, 3, 3),
        )
        changed = tuple(value.clone() for value in logits)
        for value in changed:
            value[:, -1] = 1000

        baseline = CausalAcousticLoss()(logits, labels, mask)
        padded_changed = CausalAcousticLoss()(changed, labels, mask)

        torch.testing.assert_close(baseline.loss, padded_changed.loss)

    def test_causal_loss_accepts_signed_integer_label_dtypes(self):
        loss = CausalAcousticLoss()
        item = loss(
            (torch.randn(1, 2, 4),),
            torch.tensor([[[1], [2]]], dtype=torch.int32),
            torch.tensor([[True, True]]),
        )

        self.assertTrue(torch.isfinite(item.loss).all())
        for dtype in (torch.float32, torch.uint64):
            with (
                self.subTest(dtype=dtype),
                self.assertRaisesRegex(TypeError, "signed integer"),
            ):
                loss(
                    (torch.randn(1, 2, 4),),
                    torch.tensor([[[1], [2]]], dtype=dtype),
                    torch.tensor([[True, True]]),
                )

    def test_causal_loss_ignores_nonfinite_padding_in_forward_and_backward(self):
        labels = torch.tensor([[[1], [-1]]])
        mask = torch.tensor([[True, False]])
        logits = torch.tensor(
            [[[0.0, 1.0, 0.0], [float("nan"), float("inf"), 0.0]]],
            requires_grad=True,
        )

        item = CausalAcousticLoss()((logits,), labels, mask)
        item.loss.mean().backward()

        self.assertTrue(torch.isfinite(item.loss).all())
        self.assertIsNotNone(logits.grad)
        gradient = logits.grad
        if gradient is None:
            self.fail("RVQ logits gradient is unavailable")
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.equal(gradient[:, 1], torch.zeros_like(gradient[:, 1])))


if __name__ == "__main__":
    unittest.main()
