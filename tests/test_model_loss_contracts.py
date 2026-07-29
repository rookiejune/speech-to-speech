from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch
from anydataset.types import Modality
from anytrain.codec import SemanticAcousticCodes
from anytrain.module.idspace import Layout
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from speech_to_speech.datamodule.types import (
    Language,
    ModelBatch,
    RawSpeech,
    RawSpeechBatch,
    SpeechTaskSample,
    Text,
)
from speech_to_speech.audio_route import (
    AudioStream,
    BICODEC_PREDICT_ACOUSTIC,
    BICODEC_REUSE_PROMPT_ACOUSTIC,
)
from speech_to_speech.loss import (
    FlowObjective,
    LossItem,
    Objective,
    Outputs,
    RVQObjective,
    TokenObjective,
)
from speech_to_speech.loss.flow_matching import AcousticFlowLoss
from speech_to_speech.loss.token import TokenLoss
from speech_to_speech.loss.types import combine_outputs
from speech_to_speech.model.base import Config, TokenModel
from speech_to_speech.model.lora import LoraConfig
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.pl_module import SpeechToSpeechModule
from speech_to_speech.runtime.audio_tokenizer import (
    BiCodecAudioTokenizer,
    NativeAudioTokenizer,
)
from speech_to_speech.task import Task


class _ConditionModel(TokenModel):
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
    ) -> Tensor:
        if modality is None:
            raise ValueError("objective must select a token modality")
        self.logit_rows += hidden_states.size(0)
        self.logit_modalities.append(modality)
        start, end = self.layout.blocks[modality.value]
        return torch.zeros(hidden_states.size(0), end - start)

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
    ) -> Tensor:
        if modality is None:
            raise ValueError("objective must select a token modality")
        self.logit_rows += hidden_states.size(0)
        self.logit_modalities.append(modality)
        start, end = self.layout.blocks[modality.value]
        return torch.zeros(hidden_states.size(0), end - start)


class _RVQModel(_FlowModel):
    def acoustic_logits(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
        target_acoustic_codes: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        del target_acoustic_codes
        condition = self.target_frame_condition(hidden_states, target_positions)
        return (torch.zeros(*condition.shape[:2], 3),)


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
    def test_bicodec_grouped_loss_restricts_each_prediction_head(self):
        tokenizer = BiCodecAudioTokenizer(
            semantic_vocab_size=4,
            acoustic_codebook_sizes=(2,),
            acoustic_unit_length=1,
        )
        layout = Layout(text=(0, 4), audio=(4, 4 + tokenizer.vocab_size))
        codes = {
            "semantic": torch.tensor([[1]]),
            "acoustic": torch.tensor([[0]]),
        }
        local_ids, local_groups = tokenizer.encode_streams_with_groups(
            codes,
            (AudioStream.ACOUSTIC, AudioStream.SEMANTIC),
        )
        global_ids = layout.to_global(Modality.AUDIO.value, local_ids)
        input_ids = torch.cat((torch.tensor([1]), global_ids)).unsqueeze(0)
        labels = torch.full_like(input_ids, -100)
        groups = torch.full_like(input_ids, -1)
        supervised = local_groups.ge(0)
        labels[0, 1:][supervised] = global_ids[supervised]
        groups[0, 1:][supervised] = local_groups[supervised]
        calls: list[Tensor] = []

        def selected(hidden: Tensor, allowed: Tensor) -> Tensor:
            calls.append(allowed.detach().clone())
            return torch.zeros(hidden.size(0), allowed.numel())

        item = TokenLoss(layout, tokenizer)(
            torch.zeros(1, input_ids.size(1), 3),
            labels,
            Modality.AUDIO,
            lambda hidden, modality: torch.empty(0),
            token_groups=groups,
            selected_logits=selected,
        )

        self.assertTrue(torch.isfinite(item.loss).all())
        self.assertEqual(len(calls), 3)
        expected = {
            tuple(
                layout.to_global(
                    Modality.AUDIO.value,
                    tokenizer.prediction_ids(group, device=torch.device("cpu")),
                ).tolist()
            )
            for group in (0, 1, 2)
        }
        self.assertEqual({tuple(call.tolist()) for call in calls}, expected)

    def test_checkpoint_audio_route_is_immutable(self):
        model = SimpleNamespace(
            runtime=SimpleNamespace(audio_route=BICODEC_REUSE_PROMPT_ACOUSTIC),
            lora_config=LoraConfig(),
        )
        module = SpeechToSpeechModule(
            ModuleConfig(),
            model=cast(Any, model),
            objective=cast(Any, SimpleNamespace()),
        )
        checkpoint: dict[str, object] = {}

        module.on_save_checkpoint(checkpoint)
        module.on_load_checkpoint(checkpoint)

        model.runtime.audio_route = BICODEC_PREDICT_ACOUSTIC
        with self.assertRaisesRegex(ValueError, "does not match"):
            module.on_load_checkpoint(checkpoint)

    def test_checkpoint_lora_contract_roundtrips_complete_config(self):
        config = LoraConfig(
            enabled=True,
            rank=8,
            alpha=16,
            dropout=0.1,
            target_modules=["v_proj", "q_proj"],
            use_rslora=True,
        )
        module = _checkpoint_module(config)
        checkpoint: dict[str, object] = {}

        module.on_save_checkpoint(checkpoint)
        module.on_load_checkpoint(checkpoint)

        self.assertEqual(
            checkpoint["speech_to_speech_lora"],
            {
                "grammar": "peft-lora-v1",
                "backend": "huggingface-peft",
                "adapter_name": "speech",
                "bias": "none",
                "enabled": True,
                "rank": 8,
                "alpha": 16,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
                "use_rslora": True,
            },
        )

    def test_checkpoint_requires_lora_contract_only_when_enabled(self):
        legacy_checkpoint = {"speech_to_speech_audio_route": None}

        _checkpoint_module(LoraConfig()).on_load_checkpoint(legacy_checkpoint)
        with self.assertRaisesRegex(ValueError, "missing the PEFT LoRA contract"):
            _checkpoint_module(
                LoraConfig(enabled=True),
            ).on_load_checkpoint(legacy_checkpoint)

    def test_checkpoint_rejects_lora_config_mismatch(self):
        checkpoint: dict[str, object] = {}
        _checkpoint_module(
            LoraConfig(enabled=True, alpha=16),
        ).on_save_checkpoint(checkpoint)

        with self.assertRaisesRegex(ValueError, "LoRA contract does not match"):
            _checkpoint_module(
                LoraConfig(enabled=True, alpha=32),
            ).on_load_checkpoint(checkpoint)

    def test_transfer_batch_rebuilds_frozen_audio_context(self):
        context = SemanticAcousticCodes(
            semantic=torch.tensor([[1]]),
            acoustic=torch.tensor([[2, 3]]),
        )
        batch = ModelBatch(
            input_ids=torch.tensor([[0, 1]]),
            token_labels=torch.tensor([[-100, 1]]),
            acoustic_target=None,
            tasks=[Task.TTS],
            pad_token_id=99,
            audio_contexts=(context,),
        )
        module = SpeechToSpeechModule(
            ModuleConfig(),
            model=cast(Any, SimpleNamespace()),
            objective=cast(Any, SimpleNamespace()),
        )

        with patch.object(
            LightningModule,
            "transfer_batch_to_device",
            autospec=True,
        ) as transfer:
            moved = module.transfer_batch_to_device(batch, torch.device("cpu"), 0)

        self.assertIsInstance(moved, ModelBatch)
        self.assertIsNot(moved, batch)
        transfer.assert_not_called()
        moved_context = moved.audio_contexts
        if moved_context is None or moved_context[0] is None:
            self.fail("transferred batch lost its audio context")
        self.assertIsNot(moved_context[0], context)
        torch.testing.assert_close(moved_context[0].semantic, context.semantic)
        torch.testing.assert_close(moved_context[0].acoustic, context.acoustic)

    def test_transfer_batch_keeps_raw_fallback_on_cpu(self):
        raw = RawSpeechBatch(
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
                ),
            ),
            pad_token_id=99,
        )
        module = SpeechToSpeechModule(
            ModuleConfig(),
            model=cast(Any, SimpleNamespace()),
            objective=cast(Any, SimpleNamespace()),
        )
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
        details = item.details
        if details is None:
            self.fail("token loss details are unavailable")

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
        module = SpeechToSpeechModule(
            ModuleConfig(),
            model=cast(Any, SimpleNamespace()),
            objective=objective,
        )
        asr = _batch(Task.ASR, token_labels=torch.tensor([[-100, 1]]))
        mt = _batch(Task.MT, token_labels=torch.tensor([[-100, 1]]))

        with patch.object(module, "log"):
            first = module.training_step(asr, 0)
            second = module.training_step(mt, 1)

        self.assertEqual(objective.tasks, [Task.ASR, Task.MT])
        torch.testing.assert_close(first["loss"], torch.tensor(1.0))
        torch.testing.assert_close(second["loss"], torch.tensor(3.0))

    def test_validation_step_logs_effective_unit_weighted_metrics(self):
        module = SpeechToSpeechModule(
            ModuleConfig(),
            model=cast(Any, SimpleNamespace()),
            objective=_BatchObjective(),
        )
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
            "val/token_ce": (torch.tensor(2.5), 4),
            "val/rvq_ce": (torch.tensor(3.0), 3),
            "val/rvq_codebook_0_ce": (torch.tensor(2.0), 3),
            "val/rvq_codebook_0_top1": (torch.tensor(2.0 / 3.0), 3),
            "val/rvq_codebook_1_ce": (torch.tensor(4.0), 3),
            "val/rvq_codebook_1_top1": (torch.tensor(2.0 / 3.0), 3),
            "val/flow_matching": (torch.tensor(1.5), 4),
            "val/repa": (torch.tensor(0.4), 2),
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
        rt = SimpleNamespace(
            layout=Layout(text=(0, 4), audio=(4, 9)),
            backbone=backbone,
            codec=_Codec(),
            audio_tokenizer=NativeAudioTokenizer(vocab_size=3),
        )
        model = TokenModel(
            Config(
                semantic_audio_adapter=None,
                semantic_audio_output_adapter=None,
            ),
            runtime=rt,
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
        model = TokenModel(
            Config(
                semantic_audio_adapter=None,
                semantic_audio_output_adapter=None,
            ),
            runtime=rt,
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
            TokenModel(
                Config(
                    semantic_audio_adapter=None,
                    semantic_audio_output_adapter=None,
                ),
                runtime=rt,
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

        item = AcousticFlowLoss()(
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
        training_details = outputs["rvq"].details
        validation_details = validation["rvq"].details
        if training_details is None or validation_details is None:
            self.fail("RVQ loss details are unavailable")
        self.assertNotIn("codebook_0_top1", training_details)
        self.assertIn("codebook_0_top1", validation_details)
        self.assertEqual(model.token_hidden_calls, 2)
        self.assertTrue(torch.equal(model.positions, positions))
        self.assertEqual(outputs["loss"].shape, ())
        self.assertTrue(torch.isfinite(outputs["loss"]))


def _checkpoint_module(config: LoraConfig) -> SpeechToSpeechModule[Any]:
    model = SimpleNamespace(
        runtime=SimpleNamespace(audio_route=None),
        lora_config=config,
    )
    return SpeechToSpeechModule(
        ModuleConfig(),
        model=cast(Any, model),
        objective=cast(Any, SimpleNamespace()),
    )


def _batch(
    task: Task,
    *,
    token_labels: Tensor,
    target_acoustic_codes: Tensor | None = None,
    target_audio_token_positions: Tensor | None = None,
) -> ModelBatch:
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
        tasks=[task],
        pad_token_id=99,
    )
    return batch


if __name__ == "__main__":
    unittest.main()
