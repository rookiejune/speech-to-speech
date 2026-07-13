from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from anytrain.idspace import Layout
from torch import Tensor, nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from speech_to_speech.datamodule.types import ModelBatch, Task
from speech_to_speech.loss import Loss, RVQLoss
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
        self.requested_hidden_states = False

    def __call__(self, input_ids: Tensor, **kwargs) -> CausalLMOutputWithPast:
        self.requested_hidden_states = kwargs["output_hidden_states"]
        logits = torch.zeros(
            *input_ids.shape,
            self.layout.vocab_size,
            dtype=torch.float32,
        )
        hidden_states = (torch.zeros(*input_ids.shape, 2),)
        return CausalLMOutputWithPast(
            logits=logits,
            hidden_states=hidden_states if self.requested_hidden_states else None,
        )

    def target_frame_condition(
        self, hidden_states: Tensor, target_positions: Tensor
    ) -> Tensor:
        self.positions = target_positions.clone()
        return torch.zeros(target_positions.shape + (2,))

    def acoustic_target_latent(self, acoustic_labels: Tensor) -> Tensor:
        return acoustic_labels.to(dtype=torch.float32)


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
        backbone = _Backbone(text_vocab_size=4, embedding_rows=6)
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

        logits = model.text_logits(torch.zeros(1, 2))

        self.assertEqual(logits.shape, (1, 4))

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
        self.assertFalse(model.requested_hidden_states)

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
        self.assertTrue(model.requested_hidden_states)
        self.assertTrue(torch.equal(model.positions, positions))
        self.assertEqual(outputs["loss"].shape, ())
        self.assertTrue(torch.isfinite(outputs["loss"]))

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
        self.assertTrue(model.requested_hidden_states)
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
