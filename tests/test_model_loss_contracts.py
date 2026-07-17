from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from anytrain.idspace import Layout
from torch import Tensor, nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from speech_to_speech.datamodule.types import ModelBatch, Task
from speech_to_speech.loss import Loss, RVQLoss, SemanticObjective
from speech_to_speech.loss.semantic import SemanticLoss
from speech_to_speech.model.base import Config, SemanticModel
from speech_to_speech.runtime.audio_tokenizer import NativeAudioTokenizer


class _ConditionModel(SemanticModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)

    def _input_embedding(self, input_ids: Tensor) -> Tensor:
        return input_ids[..., None].to(dtype=torch.float32)


class _Backbone(nn.Module):
    def __init__(self, *, text_vocab_size: int = 4, embedding_rows: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=2)
        self.input_embeddings = nn.Embedding(embedding_rows, 2)
        self.output_embeddings = nn.Linear(2, embedding_rows, bias=False)
        self.text_vocab_size = text_vocab_size

    def get_input_embeddings(self) -> nn.Embedding:
        return self.input_embeddings

    def get_output_embeddings(self) -> nn.Module:
        return self.output_embeddings


class _Codec:
    acoustic_feature_dim = 2
    acoustic_codebook_sizes = (3,)
    semantic_codebook = torch.randn(3, 2)


class _Decoder(nn.Module):
    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        del t, condition, mask
        return torch.zeros_like(x_t)

    def forward_with_features(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self(x_t, t, condition=condition, mask=mask), torch.ones_like(condition)


class _FlowRuntime:
    def training_sample(self, x_1: Tensor, *, x_0: Tensor | None = None):
        del x_0
        return SimpleNamespace(
            x_t=torch.zeros_like(x_1),
            velocity=torch.ones_like(x_1),
            t=torch.zeros(x_1.size(0), device=x_1.device),
        )


class _Teacher:
    feature_dim = 2

    def __call__(
        self,
        semantic_ids: Tensor,
        acoustic_ids: Tensor,
        mask: Tensor,
    ) -> Tensor:
        del semantic_ids, acoustic_ids
        return torch.ones(mask.shape + (self.feature_dim,))


class _FlowModel:
    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.acoustic_decoder = _Decoder()
        self.positions: Tensor | None = None
        self.semantic_hidden_calls = 0
        self.logit_rows = 0

    def __call__(self, input_ids: Tensor, **kwargs) -> CausalLMOutputWithPast:
        logits = torch.zeros(
            *input_ids.shape,
            self.layout.vocab_size,
            dtype=torch.float32,
        )
        return CausalLMOutputWithPast(
            logits=logits,
        )

    def semantic_hidden(self, input_ids: Tensor, **kwargs) -> Tensor:
        del kwargs
        self.semantic_hidden_calls += 1
        return torch.zeros(*input_ids.shape, 2)

    def semantic_logits(self, hidden_states: Tensor) -> Tensor:
        self.logit_rows += hidden_states.size(0)
        return torch.zeros(hidden_states.size(0), self.layout.vocab_size)

    def target_frame_condition(
        self, hidden_states: Tensor, target_positions: Tensor
    ) -> Tensor:
        self.positions = target_positions.clone()
        return torch.zeros(target_positions.shape + (2,))

    def acoustic_target_latent(self, acoustic_labels: Tensor) -> Tensor:
        return acoustic_labels.to(dtype=torch.float32)


class _SemanticForwardModel:
    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.semantic_hidden_calls = 0
        self.logit_rows = 0

    def __call__(self, input_ids: Tensor, **kwargs) -> CausalLMOutputWithPast:
        return CausalLMOutputWithPast(
            logits=torch.zeros(
                *input_ids.shape,
                self.layout.vocab_size,
                dtype=torch.float32,
            )
        )

    def semantic_hidden(self, input_ids: Tensor, **kwargs) -> Tensor:
        del kwargs
        self.semantic_hidden_calls += 1
        return torch.zeros(*input_ids.shape, 2)

    def semantic_logits(self, hidden_states: Tensor) -> Tensor:
        self.logit_rows += hidden_states.size(0)
        return torch.zeros(hidden_states.size(0), self.layout.vocab_size)


class _RVQModel(_FlowModel):
    def acoustic_logits(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
        acoustic_labels: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        del acoustic_labels
        condition = self.target_frame_condition(hidden_states, target_positions)
        return (torch.zeros(*condition.shape[:2], 3),)


class ModelLossContractTest(unittest.TestCase):
    def test_sparse_semantic_logits_match_dense_cross_entropy(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        labels = torch.tensor(
            [
                [-100, -100, 1, 4, -100],
                [-100, 5, -100, 2, 6],
            ]
        )
        hidden_values = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3) / 10
        weight_values = torch.arange(21, dtype=torch.float32).reshape(7, 3) / 10
        sparse_hidden = hidden_values.clone().requires_grad_()
        sparse_weight = weight_values.clone().requires_grad_()
        dense_hidden = hidden_values.clone().requires_grad_()
        dense_weight = weight_values.clone().requires_grad_()

        item = SemanticLoss(layout)(
            sparse_hidden,
            labels,
            lambda selected: nn.functional.linear(selected, sparse_weight),
        )
        dense_logits = nn.functional.linear(dense_hidden, dense_weight)

        target = labels[:, 1:]
        valid = target.ne(-100)
        token_loss = torch.zeros_like(target, dtype=torch.float32)
        token_loss[valid] = nn.functional.cross_entropy(
            dense_logits[:, :-1][valid],
            target[valid],
            reduction="none",
        )
        text = target.ge(0) & target.lt(4)
        audio = target.ge(4) & target.lt(7)
        text_count = text.sum(dim=1)
        audio_count = audio.sum(dim=1)
        total_count = text_count + audio_count

        torch.testing.assert_close(
            item.loss,
            (token_loss * valid).sum(dim=1) / total_count,
        )
        self.assertIsNotNone(item.details)
        details = item.details or {}
        torch.testing.assert_close(
            details["text_loss"],
            (token_loss * text).sum(dim=1) / text_count.clamp_min(1),
        )
        torch.testing.assert_close(
            details["audio_loss"],
            (token_loss * audio).sum(dim=1) / audio_count.clamp_min(1),
        )
        torch.testing.assert_close(details["text_tokens"], text_count.float())
        torch.testing.assert_close(details["audio_tokens"], audio_count.float())

        item.loss.mean().backward()
        ((token_loss * valid).sum(dim=1) / total_count).mean().backward()
        if sparse_hidden.grad is None or dense_hidden.grad is None:
            self.fail("semantic hidden gradients are unavailable")
        if sparse_weight.grad is None or dense_weight.grad is None:
            self.fail("semantic head gradients are unavailable")
        torch.testing.assert_close(sparse_hidden.grad, dense_hidden.grad)
        torch.testing.assert_close(sparse_weight.grad, dense_weight.grad)

    def test_backbone_text_embedding_has_one_registered_path(self):
        backbone = _Backbone()
        rt = SimpleNamespace(
            layout=Layout(text=(0, 4), audio=(4, 9)),
            backbone=backbone,
            codec=_Codec(),
            audio_tokenizer=NativeAudioTokenizer(vocab_size=3),
        )
        model = SemanticModel(
            Config(
                semantic_audio_adapter=None,
                semantic_audio_output_adapter=None,
                acoustic_prompt_adapter=None,
            ),
            runtime_snapshot=rt,
        )

        paths = [
            name
            for name, module in model.named_modules(remove_duplicate=False)
            if module is backbone.input_embeddings
        ]

        self.assertEqual(paths, ["backbone.input_embeddings"])

    def test_text_logits_only_cover_the_layout_vocabulary(self):
        backbone = _Backbone(text_vocab_size=4, embedding_rows=4)
        rt = SimpleNamespace(
            layout=Layout(text=(2, 6), audio=(6, 11)),
            backbone=backbone,
            codec=_Codec(),
            audio_tokenizer=NativeAudioTokenizer(vocab_size=3),
        )
        model = SemanticModel(
            Config(
                semantic_audio_adapter=None,
                semantic_audio_output_adapter=None,
                acoustic_prompt_adapter=None,
            ),
            runtime_snapshot=rt,
        )
        with torch.no_grad():
            backbone.output_embeddings.weight.copy_(
                torch.arange(8, dtype=torch.float32).reshape(4, 2)
            )

        logits = model.text_logits(torch.ones(1, 2))

        self.assertEqual(logits.shape, (1, 4))
        torch.testing.assert_close(logits, torch.tensor([[1.0, 5.0, 9.0, 13.0]]))

    def test_backbone_embeddings_must_cover_the_text_layout(self):
        backbone = _Backbone(text_vocab_size=4, embedding_rows=3)
        rt = SimpleNamespace(
            layout=Layout(text=(0, 4), audio=(4, 9)),
            backbone=backbone,
            codec=_Codec(),
            audio_tokenizer=NativeAudioTokenizer(vocab_size=3),
        )

        with self.assertRaisesRegex(ValueError, "input embedding"):
            SemanticModel(
                Config(
                    semantic_audio_adapter=None,
                    semantic_audio_output_adapter=None,
                    acoustic_prompt_adapter=None,
                ),
                runtime_snapshot=rt,
            )

    def test_condition_methods_own_the_causal_shift(self):
        model = _ConditionModel()
        hidden = torch.tensor([[[10.0], [20.0], [30.0]]])
        positions = torch.tensor([[1, 2, -1]])

        condition = model.target_frame_condition(hidden, positions)
        oracle = model.target_frame_label_condition(
            torch.tensor([[-100, 4, 5]]), positions
        )

        self.assertTrue(torch.equal(condition, torch.tensor([[[10.0], [20.0], [0.0]]])))
        self.assertTrue(torch.equal(oracle, torch.tensor([[[4.0], [5.0], [0.0]]])))

    def test_text_target_uses_semantic_objective_only(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _FlowModel(layout)
        loss = Loss(layout, _FlowRuntime())
        batch = _batch(Task.ASR, labels=torch.tensor([[-100, 1]]))

        outputs = loss(batch, model)

        self.assertIn("semantic", outputs)
        self.assertNotIn("flow_matching", outputs)
        self.assertEqual(model.semantic_hidden_calls, 1)

    def test_semantic_objective_does_not_require_acoustic_model(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _SemanticForwardModel(layout)
        batch = _batch(Task.ASR, labels=torch.tensor([[-100, 1]]))

        outputs = SemanticObjective(layout)(batch, model)

        self.assertIn("semantic", outputs)
        self.assertNotIn("flow_matching", outputs)
        self.assertEqual(model.semantic_hidden_calls, 1)

    def test_semantic_objective_projects_only_supervised_positions(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _SemanticForwardModel(layout)
        batch = _batch(Task.ASR, labels=torch.tensor([[-100, -100, 1, 2]]))

        SemanticObjective(layout)(batch, model)

        self.assertEqual(model.logit_rows, 2)

    def test_audio_target_automatically_adds_flow_objective(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _FlowModel(layout)
        loss = Loss(layout, _FlowRuntime())
        positions = torch.tensor([[1]])
        batch = _batch(
            Task.TTS,
            labels=torch.tensor([[-100, 4]]),
            acoustic_labels=torch.tensor([[[2]]]),
            acoustic_label_positions=positions,
        )

        outputs = loss(batch, model)

        self.assertIn("flow_matching", outputs)
        self.assertEqual(model.semantic_hidden_calls, 1)
        self.assertTrue(torch.equal(model.positions, positions))
        self.assertEqual(outputs["loss"].shape, ())
        self.assertTrue(torch.isfinite(outputs["loss"]))

    def test_unified_audio_target_uses_semantic_objective_only(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _FlowModel(layout)
        loss = Loss(layout, _FlowRuntime())
        batch = _batch(Task.TTS, labels=torch.tensor([[-100, 4]]))

        outputs = loss(batch, model)

        self.assertIn("semantic", outputs)
        self.assertNotIn("flow_matching", outputs)
        self.assertEqual(model.semantic_hidden_calls, 1)

    def test_repa_is_an_explicit_audio_objective(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _FlowModel(layout)
        loss = Loss(
            layout,
            _FlowRuntime(),
            repa={"weight": 0.1, "teacher": _Teacher()},
        )
        batch = _batch(
            Task.TTS,
            labels=torch.tensor([[-100, 4]]),
            acoustic_labels=torch.tensor([[[2]]]),
            acoustic_label_positions=torch.tensor([[1]]),
        )

        outputs = loss(batch, model)

        self.assertIn("repa", outputs)
        self.assertTrue(torch.isfinite(outputs["loss"]))

    def test_audio_target_automatically_adds_rvq_objective(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _RVQModel(layout)
        positions = torch.tensor([[1]])
        batch = _batch(
            Task.TTS,
            labels=torch.tensor([[-100, 4]]),
            acoustic_labels=torch.tensor([[[2]]]),
            acoustic_label_positions=positions,
        )

        outputs = RVQLoss(layout)(batch, model)

        self.assertIn("causal_lm", outputs)
        self.assertEqual(model.semantic_hidden_calls, 1)
        self.assertTrue(torch.equal(model.positions, positions))
        self.assertEqual(outputs["loss"].shape, ())
        self.assertTrue(torch.isfinite(outputs["loss"]))


def _batch(
    task: Task,
    *,
    labels: Tensor,
    acoustic_labels: Tensor | None = None,
    acoustic_label_positions: Tensor | None = None,
) -> ModelBatch:
    batch = ModelBatch(
        input_ids=labels.masked_fill(labels.eq(-100), 0),
        labels=labels,
        acoustic_input_ids=None,
        acoustic_input_positions=None,
        semantic_frame_labels=None if acoustic_labels is None else acoustic_labels,
        acoustic_labels=acoustic_labels,
        acoustic_label_positions=acoustic_label_positions,
        tasks=[task],
    )
    batch.__dict__["attention_mask"] = torch.ones_like(labels, dtype=torch.bool)
    return batch


if __name__ == "__main__":
    unittest.main()
