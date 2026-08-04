from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch
from anydataset.types import Modality
from anytrain.loss import PackedCodebookLogits
from anytrain.module.idspace import Layout
from lightning.pytorch import LightningModule
from peft import LoraConfig
from torch import Tensor, nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from speech_to_speech.datamodule.types import (
    FusedBatch,
    Language,
    LoaderBatch,
    ModelBatch,
    RawSpeech,
    RawSpeechBatch,
    SpeechTaskSample,
    Text,
)
from speech_to_speech.audio import AudioCodes, AudioStream
from semantic_acoustic_codec.loss.flow import FlowLoss

from speech_to_speech.loss.module import FlowObjective, Objective, RVQObjective, TokenObjective
from speech_to_speech.loss.token import TokenLoss
from speech_to_speech.loss.types import LossItem, Outputs, combine_outputs
from speech_to_speech.loss.validation import validation_metrics
from speech_to_speech.model._assembly import text_embedding
from speech_to_speech.model.base import Config, Model
from speech_to_speech.model.audio_output import (
    AudioOutputAdapterConfig,
    AudioOutputAdapterType,
)
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.pl_module import SpeechToSpeechModule
from speech_to_speech.task import PredictionModality
from speech_to_speech.runtime.audio_tokenizer import (
    BiCodecAudioTokenizer,
    NativeAudioTokenizer,
)
from speech_to_speech.task import Task


class _ConditionModel(Model):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.layout = Layout(text=(0, 4), audio=(4, 7))

    def _input_embedding(self, input_ids: Tensor) -> Tensor:
        return input_ids[..., None].to(dtype=torch.float32)


class _BackboneConfig(SimpleNamespace):
    def get(self, key: str, default: object = None) -> object:
        return getattr(self, key, default)


class _Backbone(nn.Module):
    def __init__(self, *, text_vocab_size: int = 4, embedding_rows: int = 4) -> None:
        super().__init__()
        self.config = _BackboneConfig(hidden_size=2)
        self.input_embeddings = nn.Embedding(embedding_rows, 2)
        self.output_embeddings = nn.Linear(2, embedding_rows, bias=False)
        self.q_proj = nn.Linear(2, 2, bias=False)
        self.text_vocab_size = text_vocab_size

    def get_input_embeddings(self) -> nn.Embedding:
        return self.input_embeddings

    def get_output_embeddings(self) -> nn.Module:
        return self.output_embeddings


class _SettableBackbone(_Backbone):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__(embedding_rows=embedding.num_embeddings)
        self.input_embeddings = embedding
        self.set_input_embeddings_calls = 0

    def set_input_embeddings(self, embedding: nn.Embedding) -> None:
        self.set_input_embeddings_calls += 1
        self.input_embeddings = embedding


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


class _NonfinitePaddedDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.prediction = nn.Parameter(torch.tensor([[[1.0], [float("nan")]]]))

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        del x_t, t, condition, mask
        return self.prediction


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
        semantic_codes: Tensor,
        acoustic_codes: Tensor,
        mask: Tensor,
    ) -> Tensor:
        del semantic_codes, acoustic_codes
        return torch.ones(mask.shape + (self.feature_dim,))


class _FlowModel:
    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.acoustic_decoder = _Decoder()
        self.positions: Tensor | None = None
        self.token_hidden_calls = 0
        self.logit_rows = 0
        self.logit_modalities: list[Modality] = []
        self.projected_audio_rows: list[int] = []

    def __call__(self, input_ids: Tensor, **kwargs) -> CausalLMOutputWithPast:
        logits = torch.zeros(
            *input_ids.shape,
            self.layout.vocab_size,
            dtype=torch.float32,
        )
        return CausalLMOutputWithPast(
            logits=logits,
        )

    def token_hidden_states(self, input_ids: Tensor, **kwargs) -> Tensor:
        del kwargs
        self.token_hidden_calls += 1
        return torch.zeros(*input_ids.shape, 2)

    def token_logits(
        self,
        hidden_states: Tensor,
        modality: Modality | None = None,
        **kwargs: object,
    ) -> Tensor:
        del kwargs
        if modality is None:
            raise ValueError("objective must select a token modality")
        self.logit_rows += hidden_states.size(0)
        self.logit_modalities.append(modality)
        start, end = self.layout.blocks[modality.value]
        return torch.zeros(hidden_states.size(0), end - start)

    def project_audio_hidden(
        self,
        hidden_state: Tensor,
        *,
        attention_mask: Tensor | None = None,
        selection_mask: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, object | None]:
        del attention_mask, past_key_values, use_cache
        if selection_mask is None:
            self.projected_audio_rows.append(hidden_state.numel() // hidden_state.size(-1))
            return hidden_state, None
        self.projected_audio_rows.append(int(selection_mask.sum().item()))
        return hidden_state[selection_mask], None

    def target_frame_condition(
        self, hidden_states: Tensor, target_positions: Tensor
    ) -> Tensor:
        self.positions = target_positions.clone()
        return torch.zeros(target_positions.shape + (2,))

    def acoustic_target_latent(self, target_acoustic_codes: Tensor) -> Tensor:
        return target_acoustic_codes.to(dtype=torch.float32)


class _TokenForwardModel:
    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.token_hidden_calls = 0
        self.project_audio_calls = 0
        self.projected_audio_rows: list[int] = []
        self.logit_rows = 0
        self.logit_modalities: list[Modality] = []

    def __call__(self, input_ids: Tensor, **kwargs) -> CausalLMOutputWithPast:
        return CausalLMOutputWithPast(
            logits=torch.zeros(
                *input_ids.shape,
                self.layout.vocab_size,
                dtype=torch.float32,
            )
        )

    def token_hidden_states(self, input_ids: Tensor, **kwargs) -> Tensor:
        del kwargs
        self.token_hidden_calls += 1
        return torch.zeros(*input_ids.shape, 2)

    def token_logits(
        self,
        hidden_states: Tensor,
        modality: Modality | None = None,
        **kwargs: object,
    ) -> Tensor:
        del kwargs
        if modality is None:
            raise ValueError("objective must select a token modality")
        self.logit_rows += hidden_states.size(0)
        self.logit_modalities.append(modality)
        start, end = self.layout.blocks[modality.value]
        return torch.zeros(hidden_states.size(0), end - start)

    def project_audio_hidden(
        self,
        hidden_state: Tensor,
        *,
        attention_mask: Tensor | None = None,
        selection_mask: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, object | None]:
        del attention_mask, past_key_values, use_cache
        self.project_audio_calls += 1
        if selection_mask is None:
            self.projected_audio_rows.append(hidden_state.numel() // hidden_state.size(-1))
            return hidden_state, None
        self.projected_audio_rows.append(int(selection_mask.sum().item()))
        return hidden_state[selection_mask], None


class _RVQModel(_FlowModel):
    def __init__(self, layout: Layout) -> None:
        super().__init__(layout)
        self.padded_logit_calls = 0
        self.packed_validations: list[bool] = []

    def acoustic_logits(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
        target_acoustic_codes: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        del target_acoustic_codes
        self.padded_logit_calls += 1
        condition = self.target_frame_condition(hidden_states, target_positions)
        return (torch.zeros(*condition.shape[:2], 3),)

    def acoustic_packed_logits(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
        target_acoustic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        validate: bool = True,
    ) -> PackedCodebookLogits:
        self.packed_validations.append(validate)
        condition = self.target_frame_condition(hidden_states, target_positions)
        mask = target_positions.ge(0) if mask is None else mask
        frame_indices = mask.flatten().nonzero().flatten()
        return PackedCodebookLogits(
            logits=(
                torch.zeros(
                    frame_indices.numel(),
                    3,
                    device=condition.device,
                ),
            ),
            labels=target_acoustic_codes.flatten(0, 1).index_select(
                0,
                frame_indices,
            ),
            row_indices=frame_indices.div(mask.size(1), rounding_mode="floor"),
            batch_size=mask.size(0),
        )


class _BatchObjective(Objective[Any]):
    def __init__(self) -> None:
        super().__init__()
        self.tasks: list[Task] = []

    def forward(self, batch: ModelBatch, model: Any) -> Outputs:
        del model
        self.tasks.append(batch.tasks[0])
        value = 1.0 if batch.tasks[0] is Task.ASR else 3.0
        tokens = 1.0 if batch.tasks[0] is Task.ASR else 3.0
        return {
            "loss": torch.tensor(value),
            "token": LossItem(
                torch.tensor([value]),
                {"tokens": torch.tensor([tokens])},
            ),
        }


class ModelLossContractTest(unittest.TestCase):
    def test_bicodec_audio_loss_uses_full_audio_vocab(self):
        case = _bicodec_full_vocab_case()
        calls: list[tuple[Modality | None, int]] = []

        def token_logits(hidden: Tensor, modality: Modality | None) -> Tensor:
            calls.append((modality, hidden.size(0)))
            start, end = case.layout.blocks[Modality.AUDIO.value]
            return torch.zeros(hidden.size(0), end - start)

        item = TokenLoss(case.layout)(
            torch.zeros(1, case.input_ids.size(1), 3),
            case.labels,
            Modality.AUDIO,
            token_logits,
        )

        self.assertTrue(torch.isfinite(item.loss).all())
        self.assertEqual(
            calls,
            [(Modality.AUDIO, int(case.labels.ne(-100).sum()))],
        )

    def test_token_objective_supervises_parallel_text_and_audio_heads(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _TokenForwardModel(layout)
        batch = _batch(
            Task.PARALLEL_AR,
            token_labels=torch.tensor([[-100, 1, 4]]),
            prediction=PredictionModality.PARALLEL,
        )

        outputs = TokenObjective(layout)(batch, model)

        self.assertIn("token", outputs)
        self.assertEqual(model.token_hidden_calls, 1)
        self.assertEqual(model.project_audio_calls, 1)
        self.assertEqual(model.projected_audio_rows, [1])
        self.assertEqual(model.logit_modalities, [Modality.AUDIO, Modality.TEXT])
        self.assertEqual(model.logit_rows, 2)
        details = _require_details(self, outputs["token"], "token loss")
        self.assertEqual(float(details["text_tokens"].sum()), 1.0)
        self.assertEqual(float(details["audio_tokens"].sum()), 1.0)

    def test_token_objective_supervises_interleaved_text_and_audio_heads(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _TokenForwardModel(layout)
        batch = _batch(
            Task.INTERLEAVED_AR,
            token_labels=torch.tensor([[-100, 4, 1, 5]]),
            prediction=PredictionModality.INTERLEAVED,
        )

        outputs = TokenObjective(layout)(batch, model)

        self.assertEqual(model.project_audio_calls, 1)
        self.assertEqual(model.projected_audio_rows, [2])
        self.assertEqual(
            set(model.logit_modalities),
            {Modality.TEXT, Modality.AUDIO},
        )
        self.assertEqual(model.logit_rows, 3)
        details = _require_details(self, outputs["token"], "token loss")
        self.assertEqual(float(details["text_tokens"].sum()), 1.0)
        self.assertEqual(float(details["audio_tokens"].sum()), 2.0)

    def test_token_objective_uses_full_audio_logits_for_bicodec_tokens(self):
        case = _bicodec_full_vocab_case()
        model = _TokenForwardModel(case.layout)
        batch = ModelBatch(
            input_ids=case.input_ids,
            token_labels=case.labels,
            acoustic_target=None,
            tasks=[Task.TTS],
            predictions=[PredictionModality.AUDIO],
            pad_token_id=99,
        )

        outputs = TokenObjective(case.layout)(batch, model)

        self.assertIn("token", outputs)
        self.assertEqual(model.token_hidden_calls, 1)
        self.assertEqual(model.project_audio_calls, 1)
        self.assertEqual(
            model.projected_audio_rows,
            [int(case.labels.ne(-100).sum())],
        )
        self.assertEqual(model.logit_rows, int(case.labels.ne(-100).sum()))
        self.assertEqual(model.logit_modalities, [Modality.AUDIO])

    def test_validation_metrics_maps_outputs_and_rvq_top1(self):
        token = LossItem(
            torch.tensor([1.0]),
            {"tokens": torch.tensor([2.0])},
        )
        rvq = LossItem(
            torch.tensor([0.5]),
            {
                "frames": torch.tensor([4.0]),
                "codebook_0": torch.tensor([0.5]),
                "codebook_0_top1": torch.tensor([1.0]),
            },
        )
        metrics = validation_metrics({"loss": torch.tensor(1.5), "token": token, "rvq": rvq})
        self.assertEqual(set(metrics), {
            "token/loss",
            "acoustic/rvq/loss",
            "acoustic/rvq/codebook_0",
            "acoustic/rvq/codebook_0_top1",
        })
        self.assertEqual(float(metrics["token/loss"].values.sum()), 1.0)
        self.assertEqual(float(metrics["token/loss"].weights.sum()), 2.0)
        self.assertEqual(float(metrics["acoustic/rvq/codebook_0_top1"].values.sum()), 1.0)

    def test_transfer_batch_requests_nonblocking_model_batch_copy(self):
        batch = _batch(Task.MT, token_labels=torch.tensor([[-100, 1]]))
        module = _module()
        device = torch.device("cpu")

        with patch.object(ModelBatch, "to", autospec=True, return_value=batch) as move:
            moved = module.transfer_batch_to_device(batch, device, 0)

        self.assertIs(moved, batch)
        move.assert_called_once_with(batch, device, non_blocking=True)

    def test_transfer_carries_exact_cpu_input_modality_hints(self):
        model = _model(_Backbone())
        module = _module(model=model)
        batch = ModelBatch(
            input_ids=torch.tensor([[0, 4]]),
            token_labels=torch.tensor([[-100, 4]]),
            acoustic_target=None,
            tasks=[Task.TTS],
            predictions=[Task.TTS.prediction_modality],
            pad_token_id=99,
        )

        moved = module.transfer_batch_to_device(batch, torch.device("cpu"), 0)

        self.assertEqual(
            moved.input_modalities,
            frozenset({Modality.TEXT, Modality.AUDIO}),
        )
        self.assertTrue(moved.audio_input_positions_validated)

    def test_transfer_batch_keeps_raw_fallback_on_cpu(self):
        raw = _raw_tts_batch()
        module = _module()
        device = torch.device("cpu")

        with patch.object(
            LightningModule,
            "transfer_batch_to_device",
            autospec=True,
        ) as transfer:
            moved = module.transfer_batch_to_device(raw, device, 0)

        self.assertIs(moved, raw)
        raw_target = raw.samples[0].target
        self.assertIsInstance(raw_target, RawSpeech)
        self.assertEqual(raw_target.waveform.device.type, "cpu")
        transfer.assert_not_called()

    def test_sparse_modality_logits_match_dense_modality_cross_entropy(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        labels = torch.tensor(
            [
                [-100, -100, 4, 5, -100],
                [-100, 5, -100, 4, 6],
            ]
        )
        hidden_values = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3) / 10
        weight_values = torch.arange(21, dtype=torch.float32).reshape(7, 3) / 10
        sparse_hidden = hidden_values.clone().requires_grad_()
        sparse_weight = weight_values.clone().requires_grad_()
        dense_hidden = hidden_values.clone().requires_grad_()
        dense_weight = weight_values.clone().requires_grad_()

        item = TokenLoss(layout)(
            sparse_hidden,
            labels,
            Modality.AUDIO,
            lambda selected, modality: nn.functional.linear(
                selected,
                sparse_weight[slice(*layout.blocks[modality.value])],
            ),
        )
        audio_start, audio_end = layout.blocks[Modality.AUDIO.value]
        dense_logits = nn.functional.linear(
            dense_hidden,
            dense_weight[audio_start:audio_end],
        )

        target = labels[:, 1:]
        valid = target.ne(-100)
        token_loss = torch.zeros_like(target, dtype=torch.float32)
        token_loss[valid] = nn.functional.cross_entropy(
            dense_logits[:, :-1][valid],
            target[valid] - audio_start,
            reduction="none",
        )
        text = torch.zeros_like(valid)
        audio = valid
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
            self.fail("token hidden gradients are unavailable")
        if sparse_weight.grad is None or dense_weight.grad is None:
            self.fail("token head gradients are unavailable")
        torch.testing.assert_close(sparse_hidden.grad, dense_hidden.grad)
        torch.testing.assert_close(sparse_weight.grad, dense_weight.grad)

    def test_token_loss_rejects_a_batch_row_without_targets(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        loss = TokenLoss(layout)

        with self.assertRaisesRegex(ValueError, "each token label row"):
            loss(
                torch.zeros(2, 2, 3),
                torch.tensor([[-100, -100], [-100, 1]]),
                Modality.TEXT,
                lambda hidden, modality: torch.zeros(
                    hidden.size(0),
                    layout.blocks[modality.value][1] - layout.blocks[modality.value][0],
                ),
            )

        with self.assertRaisesRegex(TypeError, "signed integer"):
            loss(
                torch.zeros(1, 2, 3),
                torch.tensor([[-100, 1]], dtype=torch.float32),
                Modality.TEXT,
                lambda hidden, modality: torch.zeros(
                    hidden.size(0),
                    layout.blocks[modality.value][1] - layout.blocks[modality.value][0],
                ),
            )

        item = loss(
            torch.zeros(1, 2, 3),
            torch.tensor([[-100, 1]], dtype=torch.int32),
            Modality.TEXT,
            lambda hidden, modality: torch.zeros(
                hidden.size(0),
                layout.blocks[modality.value][1] - layout.blocks[modality.value][0],
            ),
        )
        self.assertTrue(torch.isfinite(item.loss).all())

    def test_trusted_token_loss_avoids_host_predicates_for_empty_modality(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        modalities: list[Modality] = []

        def logits(hidden: Tensor, modality: Modality) -> Tensor:
            modalities.append(modality)
            start, end = layout.blocks[modality.value]
            return hidden.new_zeros((hidden.size(0), end - start))

        with patch.object(
            torch.Tensor,
            "__bool__",
            side_effect=AssertionError("trusted token loss must stay on device"),
        ):
            item = TokenLoss(layout)(
                torch.zeros(1, 3, 2),
                torch.tensor([[-100, 1, 2]]),
                PredictionModality.PARALLEL,
                logits,
                validate=False,
            )

        self.assertEqual(set(modalities), {Modality.TEXT, Modality.AUDIO})
        self.assertTrue(torch.isfinite(item.loss).all())

    def test_token_objective_uses_effective_token_mean(self):
        layout = Layout(text=(0, 2), audio=(2, 4))
        loss = TokenLoss(layout)
        hidden = torch.tensor(
            [[[0.0], [0.0], [0.0], [0.0]], [[0.0], [2.0], [2.0], [2.0]]]
        )
        labels = torch.tensor([[-100, 1, -100, -100], [-100, 1, 1, 1]])

        def logits(values: Tensor, modality: Modality) -> Tensor:
            del modality
            value = values[:, 0]
            return torch.stack((value, -value), dim=-1)

        item = loss(hidden, labels, Modality.TEXT, logits)
        details = _require_details(self, item, "token loss")

        weighted = item.weighted_mean(details["tokens"])
        unweighted = item.loss.mean()

        self.assertNotEqual(float(weighted), float(unweighted))
        torch.testing.assert_close(
            weighted,
            nn.functional.cross_entropy(
                torch.tensor([[0.0, -0.0], [0.0, -0.0], [2.0, -2.0], [2.0, -2.0]]),
                torch.tensor([1, 1, 1, 1]),
            ),
        )

    def test_training_step_consumes_one_accumulation_microbatch(self):
        objective = _BatchObjective()
        module = _module(objective=objective)
        asr = _batch(Task.ASR, token_labels=torch.tensor([[-100, 1]]))
        mt = _batch(Task.MT, token_labels=torch.tensor([[-100, 1]]))

        with patch.object(module, "log"):
            first = module.training_step(asr, 0)
            second = module.training_step(mt, 1)

        self.assertEqual(objective.tasks, [Task.ASR, Task.MT])
        torch.testing.assert_close(first["loss"], torch.tensor(1.0))
        torch.testing.assert_close(second["loss"], torch.tensor(3.0))

    def test_training_step_fuses_accumulation_window_for_static_ddp(self):
        objective = _BatchObjective()
        module = _module(objective=objective)
        asr = _batch(Task.ASR, token_labels=torch.tensor([[-100, 1]]))
        mt = _batch(Task.MT, token_labels=torch.tensor([[-100, 1]]))

        with (
            patch.object(module, "log"),
            patch.object(
                torch.Tensor,
                "__bool__",
                side_effect=AssertionError("fused loss weights must stay on device"),
            ),
            patch(
                "speech_to_speech.pl_module.module.combine_outputs",
                wraps=combine_outputs,
            ) as combine,
            patch(
                "anytrain.loss.output.loss_item_mean",
                side_effect=AssertionError(
                    "fused training must not reduce concatenated objective losses"
                ),
            ),
        ):
            outputs = module.training_step(
                FusedBatch(
                    (asr, mt),
                    loader_names=("asr", "mt"),
                    loss_weights=(0.9, 0.1),
                ),
                0,
            )
            self.assertEqual(combine.call_count, 1)
            groups = module.current_gradient_loss_groups()
            self.assertEqual(combine.call_count, 1)

        self.assertEqual(objective.tasks, [Task.ASR, Task.MT])
        torch.testing.assert_close(outputs["loss"], torch.tensor(1.2))
        torch.testing.assert_close(outputs["token"].loss, torch.tensor([1.0, 3.0]))
        details = _require_details(self, outputs["token"], "fused token loss")
        torch.testing.assert_close(
            details["tokens"],
            torch.tensor([1.0, 3.0]),
        )
        self.assertEqual(tuple(groups), ("batch", "asr", "mt"))
        torch.testing.assert_close(groups["batch"]["loss"], torch.tensor(1.2))
        torch.testing.assert_close(groups["asr"]["loss"], torch.tensor(1.0))
        torch.testing.assert_close(groups["mt"]["loss"], torch.tensor(3.0))

    def test_serial_joint_scales_each_accumulation_microbatch(self):
        objective = _BatchObjective()
        module = _module(objective=objective)
        asr = _batch(Task.ASR, token_labels=torch.tensor([[-100, 1]]))
        mt = _batch(Task.MT, token_labels=torch.tensor([[-100, 1]]))

        with patch.object(module, "log"):
            first = module.training_step(LoaderBatch(asr, "asr", 1.8), 0)
            second = module.training_step(LoaderBatch(mt, "mt", 0.2), 1)

        torch.testing.assert_close(first["loss"], torch.tensor(1.8))
        torch.testing.assert_close(second["loss"], torch.tensor(0.6))
        torch.testing.assert_close(
            (first["loss"] + second["loss"]) / 2,
            torch.tensor(1.2),
        )

    def test_validation_step_logs_effective_unit_weighted_metrics(self):
        module = _module(objective=_BatchObjective())
        outputs: Outputs = {
            "loss": torch.tensor(3.0),
            "token": LossItem(
                torch.tensor([1.0, 3.0]),
                {"tokens": torch.tensor([1.0, 3.0])},
            ),
            "rvq": LossItem(
                torch.tensor([2.0, 5.0]),
                {
                    "frames": torch.tensor([2.0, 1.0]),
                    "codebook_0": torch.tensor([1.0, 4.0]),
                    "codebook_0_top1": torch.tensor([0.5, 1.0]),
                    "codebook_1": torch.tensor([3.0, 6.0]),
                    "codebook_1_top1": torch.tensor([1.0, 0.0]),
                },
            ),
            "flow_matching": LossItem(
                torch.tensor([1.0, 3.0]),
                {"frames": torch.tensor([3.0, 1.0])},
            ),
            "repa": LossItem(
                torch.tensor([0.2, 0.6]),
                {"frames": torch.tensor([1.0, 1.0])},
            ),
        }
        batch = _batch(Task.TTS, token_labels=torch.tensor([[-100, 1]]))

        with (
            patch.object(module, "_outputs", return_value=outputs) as collect,
            patch.object(module, "log") as log,
        ):
            returned = module.validation_step(batch, 0)

        self.assertIs(returned, outputs)
        self.assertEqual(collect.call_args.args[1], module.objective.validation)
        calls = {item.args[0]: item for item in log.call_args_list}
        expected = {
            "val/token/loss": (torch.tensor(2.5), 4),
            "val/acoustic/rvq/loss": (torch.tensor(3.0), 3),
            "val/acoustic/rvq/codebook_0": (torch.tensor(2.0), 3),
            "val/acoustic/rvq/codebook_0_top1": (torch.tensor(2.0 / 3.0), 3),
            "val/acoustic/rvq/codebook_1": (torch.tensor(4.0), 3),
            "val/acoustic/rvq/codebook_1_top1": (torch.tensor(2.0 / 3.0), 3),
            "val/acoustic/flow_matching/loss": (torch.tensor(1.5), 4),
            "val/acoustic/repa/loss": (torch.tensor(0.4), 2),
        }
        self.assertEqual(set(calls), set(expected))
        for name, (value, batch_size) in expected.items():
            with self.subTest(metric=name):
                torch.testing.assert_close(calls[name].args[1], value)
                self.assertEqual(
                    calls[name].kwargs,
                    {
                        "on_step": False,
                        "on_epoch": True,
                        "sync_dist": True,
                        "batch_size": batch_size,
                    },
                )

    def test_mt_validation_generates_and_logs_anytrain_text_metrics(self):
        class _Tokenizer:
            def decode(self, token_ids, *, skip_special_tokens):
                self.skip_special_tokens = skip_special_tokens
                return " ".join(str(token_id) for token_id in token_ids)

        tokenizer = _Tokenizer()
        runtime = SimpleNamespace(
            text_tokenizer=tokenizer,
            layout=Layout(text=(0, 32), audio=(32, 36)),
            pad_token_id=0,
            eos_token_id=31,
        )
        module = _module(
            ModuleConfig(mt_validation_max_new_tokens=8),
            model=SimpleNamespace(runtime=runtime),
            objective=_BatchObjective(),
        )
        batch = _batch(
            Task.MT,
            token_labels=torch.tensor([[-100, 4, 5, 31]]),
        )

        module.on_validation_epoch_start()
        with (
            patch(
                "speech_to_speech.pl_module.module.generate_responses",
                return_value=[
                    {
                        "response_ids": torch.tensor([4, 5]),
                        "audio": None,
                    }
                ],
            ) as generate,
            patch.object(module, "log") as log,
        ):
            returned = module.validation_step(batch, 0)
            module.on_validation_epoch_end()

        self.assertEqual(float(returned["mt_validation_samples"]), 1.0)
        self.assertTrue(tokenizer.skip_special_tokens)
        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 8)
        self.assertFalse(generate.call_args.kwargs["do_sample"])
        metrics = {call.args[0]: call.args[1] for call in log.call_args_list}
        self.assertEqual(
            set(metrics),
            {"val/mt/bleu", "val/mt/chrf", "val/mt/wer"},
        )
        self.assertEqual(metrics["val/mt/bleu"], 100.0)
        self.assertEqual(metrics["val/mt/chrf"], 100.0)
        self.assertEqual(metrics["val/mt/wer"], 0.0)

    def test_combined_outputs_use_effective_units_without_loader_weights(self):
        first = LossItem(
            torch.tensor([1.0]),
            {"tokens": torch.tensor([10.0])},
        )
        second = LossItem(
            torch.tensor([3.0]),
            {"tokens": torch.tensor([30.0])},
        )

        outputs = combine_outputs(
            [
                {"loss": torch.tensor(1.0), "token": first},
                {"loss": torch.tensor(3.0), "token": second},
            ]
        )

        torch.testing.assert_close(outputs["loss"], torch.tensor(2.5))
        torch.testing.assert_close(outputs["token"].loss, torch.tensor([1.0, 3.0]))

    def test_backbone_text_embedding_has_one_registered_path(self):
        backbone = _Backbone()
        model = _model(backbone)
        text_embedding = model.text_embedding

        paths = [
            name
            for name, module in model.named_modules(remove_duplicate=False)
            if module is text_embedding
        ]
        state_paths = [
            name
            for name, value in model.state_dict(keep_vars=True).items()
            if value is text_embedding.weight
        ]

        self.assertIs(backbone.get_input_embeddings(), text_embedding)
        self.assertEqual(paths, ["backbone.input_embeddings"])
        self.assertEqual(state_paths, ["backbone.input_embeddings.weight"])

    def test_text_embedding_prefix_keeps_the_backbone_table(self):
        source = nn.Embedding(
            6,
            2,
            padding_idx=1,
            max_norm=1.25,
            norm_type=1.5,
            scale_grad_by_freq=True,
            sparse=True,
            dtype=torch.float64,
        )
        with torch.no_grad():
            source.weight.copy_(torch.arange(12, dtype=torch.float64).reshape(6, 2))
        source.weight.requires_grad_(False)
        backbone = _SettableBackbone(source)

        resolved = text_embedding(cast(Any, backbone), 4)

        self.assertIs(resolved, source)
        self.assertIs(backbone.get_input_embeddings(), source)
        self.assertEqual(backbone.set_input_embeddings_calls, 0)
        token_ids = torch.tensor([0, 1, 3])
        torch.testing.assert_close(resolved(token_ids), source(token_ids))

    def test_exact_text_vocabulary_does_not_replace_backbone_embedding(self):
        source = nn.Embedding(4, 2)
        backbone = _SettableBackbone(source)

        resolved = text_embedding(cast(Any, backbone), 4)

        self.assertIs(resolved, source)
        self.assertIs(backbone.get_input_embeddings(), source)
        self.assertEqual(backbone.set_input_embeddings_calls, 0)

    def test_reusing_runtime_keeps_a_real_backbone_embedding(self):
        backbone = _Backbone()
        runtime = _runtime(backbone)

        first = Model(
            Config(
                semantic_audio_adapter=None,
                audio_output_adapter=AudioOutputAdapterConfig(
                    type=AudioOutputAdapterType.NONE
                ),
            ),
            runtime=runtime,
        )
        second = Model(
            Config(
                semantic_audio_adapter=None,
                audio_output_adapter=AudioOutputAdapterConfig(
                    type=AudioOutputAdapterType.NONE
                ),
            ),
            runtime=runtime,
        )

        embedding = backbone.get_input_embeddings()
        self.assertIsInstance(embedding, nn.Embedding)
        self.assertIs(first.text_embedding, embedding)
        self.assertIs(second.text_embedding, embedding)

    def test_token_model_injects_the_configured_peft_adapter(self):
        backbone = _Backbone()
        model = _model(backbone, lora=LoraConfig(r=1, target_modules=["q_proj"]))

        self.assertIs(model.backbone, backbone)
        self.assertIn("speech", backbone.q_proj.lora_A)
        self.assertFalse(backbone.q_proj.base_layer.weight.requires_grad)
        self.assertTrue(backbone.q_proj.lora_A["speech"].weight.requires_grad)

    def test_text_logits_only_cover_the_layout_vocabulary(self):
        backbone = _Backbone(text_vocab_size=4, embedding_rows=6)
        model = _model(backbone, layout=Layout(text=(2, 6), audio=(6, 12)))
        with torch.no_grad():
            backbone.input_embeddings.weight.copy_(
                torch.arange(12, dtype=torch.float32).reshape(6, 2)
            )

        logits = model.text_logits(torch.ones(1, 2))

        self.assertIs(model.text_embedding, backbone.input_embeddings)
        self.assertEqual(model.text_embedding.num_embeddings, 6)
        self.assertEqual(logits.shape, (1, 4))
        torch.testing.assert_close(logits, torch.tensor([[1.0, 5.0, 9.0, 13.0]]))

    def test_backbone_embeddings_must_cover_the_text_layout(self):
        backbone = _Backbone(text_vocab_size=4, embedding_rows=3)
        with self.assertRaisesRegex(ValueError, "input embedding"):
            _model(backbone)

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

    def test_label_condition_uses_a_valid_placeholder_for_offset_text_layout(self):
        model = _model(
            _Backbone(embedding_rows=6),
            layout=Layout(text=(2, 6), audio=(6, 12)),
        )

        condition = model.target_frame_label_condition(
            torch.tensor([[2, -100]]),
            torch.tensor([[0, -1]]),
        )

        self.assertEqual(tuple(condition.shape), (1, 2, 2))
        torch.testing.assert_close(condition[:, 1], torch.zeros(1, 2))

    def test_text_target_uses_token_objective_only(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _FlowModel(layout)
        loss = FlowObjective(layout, _FlowRuntime())
        batch = _batch(Task.ASR, token_labels=torch.tensor([[-100, 1]]))

        outputs = loss(batch, model)

        self.assertIn("token", outputs)
        self.assertNotIn("flow_matching", outputs)
        self.assertEqual(model.token_hidden_calls, 1)

    def test_text_target_does_not_require_acoustic_target_for_rvq(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _RVQModel(layout)
        batch = _batch(Task.ASR, token_labels=torch.tensor([[-100, 1]]))

        outputs = RVQObjective(layout)(batch, model)

        self.assertIn("token", outputs)
        self.assertNotIn("rvq", outputs)
        self.assertEqual(model.token_hidden_calls, 1)

    def test_token_objective_does_not_require_acoustic_model(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _TokenForwardModel(layout)
        batch = _batch(Task.ASR, token_labels=torch.tensor([[-100, 1]]))

        outputs = TokenObjective(layout)(batch, model)

        self.assertIn("token", outputs)
        self.assertNotIn("flow_matching", outputs)
        self.assertEqual(model.token_hidden_calls, 1)

    def test_token_objective_projects_only_supervised_positions(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _TokenForwardModel(layout)
        batch = _batch(
            Task.ASR,
            token_labels=torch.tensor([[-100, -100, 1, 2]]),
        )

        TokenObjective(layout)(batch, model)

        self.assertEqual(model.logit_rows, 2)
        self.assertEqual(model.logit_modalities, [Modality.TEXT])

    def test_audio_target_automatically_adds_flow_objective(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _FlowModel(layout)
        loss = FlowObjective(layout, _FlowRuntime())
        positions = torch.tensor([[1]])
        batch = _batch(
            Task.TTS,
            token_labels=torch.tensor([[-100, 4]]),
            target_acoustic_codes=torch.tensor([[[2]]]),
            target_audio_token_positions=positions,
        )

        outputs = loss(batch, model)

        self.assertIn("flow_matching", outputs)
        self.assertEqual(model.token_hidden_calls, 1)
        self.assertEqual(model.logit_modalities, [Modality.AUDIO])
        self.assertTrue(torch.equal(model.positions, positions))
        self.assertEqual(outputs["loss"].shape, ())
        self.assertTrue(torch.isfinite(outputs["loss"]))

    def test_flow_loss_ignores_nonfinite_padding_in_forward_and_backward(self):
        decoder = _NonfinitePaddedDecoder()
        mask = torch.tensor([[True, False]])

        item = FlowLoss()(
            decoder,
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2, 1),
            mask,
            _FlowRuntime(),
        )
        item.loss.mean().backward()

        self.assertTrue(torch.isfinite(item.loss).all())
        self.assertIsNotNone(decoder.prediction.grad)
        gradient = decoder.prediction.grad
        if gradient is None:
            self.fail("flow prediction gradient is unavailable")
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(torch.equal(gradient[:, 1], torch.zeros_like(gradient[:, 1])))

    def test_acoustic_objectives_require_audio_target_data(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        batch = _batch(Task.TTS, token_labels=torch.tensor([[-100, 4]]))

        objectives = (
            (FlowObjective(layout, _FlowRuntime()), _FlowModel(layout)),
            (RVQObjective(layout), _RVQModel(layout)),
        )
        for objective, model in objectives:
            with self.subTest(objective=type(objective).__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"{type(objective).__name__} requires acoustic target data",
                ):
                    objective(batch, model)
                self.assertEqual(model.token_hidden_calls, 0)

    def test_repa_is_an_explicit_audio_objective(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _FlowModel(layout)
        loss = FlowObjective(
            layout,
            _FlowRuntime(),
            repa={"weight": 0.1, "teacher": _Teacher()},
        )
        batch = _batch(
            Task.TTS,
            token_labels=torch.tensor([[-100, 4]]),
            target_acoustic_codes=torch.tensor([[[2]]]),
            target_audio_token_positions=torch.tensor([[1]]),
        )

        outputs = loss(batch, model)

        self.assertIn("repa", outputs)
        self.assertTrue(torch.isfinite(outputs["loss"]))

    def test_repa_uses_frame_weighting_and_requires_frame_counts(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _FlowModel(layout)
        objective = FlowObjective(
            layout,
            _FlowRuntime(),
            repa={"weight": 0.1, "teacher": _Teacher()},
        )
        batch = _batch(
            Task.TTS,
            token_labels=torch.tensor([[-100, 4], [-100, 4]]),
            target_acoustic_codes=torch.tensor(
                [[[2], [2], [2]], [[2], [-1], [-1]]]
            ),
            target_audio_token_positions=torch.tensor(
                [[1, 1, 1], [1, -1, -1]]
            ),
        )
        token = LossItem(
            torch.zeros(2),
            {"tokens": torch.ones(2)},
        )
        flow = LossItem(
            torch.zeros(2),
            {"frames": torch.tensor([3.0, 1.0])},
        )
        repa = LossItem(
            torch.tensor([1.0, 3.0]),
            {"frames": torch.tensor([3.0, 1.0])},
        )
        representation = torch.zeros(2, 3, 2)

        with (
            patch.object(objective.token, "forward", return_value=token),
            patch.object(
                objective.flow_matching,
                "forward_with_features",
                return_value=(flow, representation),
            ),
            patch.object(objective.repa_loss, "forward", return_value=repa) as loss,
        ):
            outputs = objective(batch, model)
            torch.testing.assert_close(outputs["loss"], torch.tensor(0.15))

            loss.return_value = LossItem(torch.tensor([1.0, 3.0]))
            with self.assertRaisesRegex(ValueError, "frames"):
                objective(batch, model)

    def test_audio_target_automatically_adds_rvq_objective(self):
        layout = Layout(text=(0, 4), audio=(4, 7))
        model = _RVQModel(layout)
        positions = torch.tensor([[1]])
        batch = _batch(
            Task.TTS,
            token_labels=torch.tensor([[-100, 4]]),
            target_acoustic_codes=torch.tensor([[[2]]]),
            target_audio_token_positions=positions,
        )

        objective = RVQObjective(layout)
        outputs = objective(batch, model)
        validation = objective.validation(batch, model)

        self.assertIn("rvq", outputs)
        training_details = _require_details(self, outputs["rvq"], "RVQ training loss")
        validation_details = _require_details(self, validation["rvq"], "RVQ validation loss")
        self.assertNotIn("codebook_0_top1", training_details)
        self.assertIn("codebook_0_top1", validation_details)
        self.assertEqual(model.token_hidden_calls, 2)
        self.assertTrue(torch.equal(model.positions, positions))
        self.assertEqual(model.padded_logit_calls, 0)
        self.assertEqual(model.packed_validations, [False, False])
        self.assertEqual(outputs["loss"].shape, ())
        self.assertTrue(torch.isfinite(outputs["loss"]))


def _module(
    config: ModuleConfig | None = None,
    *,
    model: object | None = None,
    objective: object | None = None,
) -> SpeechToSpeechModule[Any]:
    return SpeechToSpeechModule(
        ModuleConfig() if config is None else config,
        model=cast(Any, SimpleNamespace() if model is None else model),
        objective=cast(Any, SimpleNamespace() if objective is None else objective),
    )


def _require_details(
    test: unittest.TestCase,
    item: LossItem,
    name: str,
) -> dict[str, Tensor]:
    details = item.details
    if details is None:
        test.fail(f"{name} details are unavailable")
    return details


def _bicodec_full_vocab_case() -> SimpleNamespace:
    tokenizer = BiCodecAudioTokenizer(
        semantic_codebook_size=4,
        global_codebook_sizes=(2,),
        global_unit_length=1,
    )
    layout = Layout(text=(0, 4), audio=(4, 4 + tokenizer.vocab_size))
    codes = AudioCodes(
        semantic_codes=torch.tensor([[1]]),
        global_codes=torch.tensor([[0]]),
    )
    local_ids = tokenizer.encode_streams(
        codes,
        (AudioStream.GLOBAL, AudioStream.SEMANTIC),
    )
    global_ids = layout.to_global(Modality.AUDIO.value, local_ids)
    input_ids = torch.cat((torch.tensor([1]), global_ids)).unsqueeze(0)
    labels = torch.full_like(input_ids, -100)
    labels[0, 1:] = global_ids
    return SimpleNamespace(
        tokenizer=tokenizer,
        layout=layout,
        input_ids=input_ids,
        labels=labels,
    )


def _runtime(
    backbone: _Backbone,
    *,
    layout: Layout | None = None,
    tokenizer: NativeAudioTokenizer | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        layout=Layout(text=(0, 4), audio=(4, 10)) if layout is None else layout,
        backbone=backbone,
        codec=_Codec(),
        audio_tokenizer=NativeAudioTokenizer(vocab_size=3) if tokenizer is None else tokenizer,
    )


def _model(
    backbone: _Backbone,
    *,
    layout: Layout | None = None,
    lora: LoraConfig | None = None,
) -> Model:
    return Model(
        Config(
            semantic_audio_adapter=None,
            audio_output_adapter=AudioOutputAdapterConfig(
                type=AudioOutputAdapterType.NONE
            ),
            lora=lora,
        ),
        runtime=_runtime(backbone, layout=layout),
    )


def _raw_tts_batch() -> RawSpeechBatch:
    return RawSpeechBatch(
        samples=(
            SpeechTaskSample(
                source=Text(torch.tensor([1]), Language.EN),
                target=RawSpeech(
                    text_token_ids=torch.tensor([2]),
                    waveform=torch.ones(4),
                    sample_rate=4,
                    language=Language.ZH,
                ),
                task=Task.TTS,
                prediction=Task.TTS.prediction_modality,
            ),
        ),
        pad_token_id=99,
    )


def _batch(
    task: Task,
    *,
    token_labels: Tensor,
    target_acoustic_codes: Tensor | None = None,
    target_audio_token_positions: Tensor | None = None,
    prediction: PredictionModality | None = None,
) -> ModelBatch:
    resolved = (
        task.prediction_modality if prediction is None else prediction
    )
    batch = ModelBatch(
        input_ids=token_labels.masked_fill(token_labels.eq(-100), 0),
        token_labels=token_labels,
        acoustic_target=(
            None
            if target_acoustic_codes is None or target_audio_token_positions is None
            else {
                "semantic_codes": target_acoustic_codes,
                "codes": target_acoustic_codes,
                "token_positions": target_audio_token_positions,
            }
        ),
        tasks=[task] * token_labels.size(0),
        predictions=[resolved] * token_labels.size(0),
        pad_token_id=99,
    )
    return batch


if __name__ == "__main__":
    unittest.main()
