from __future__ import annotations

import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from anytrain.tokenizer import CodecBPE
from anytrain.idspace import Modality
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast

from speech_to_speech.config import LoRAConfig, ModelConfig
from speech_to_speech.datamodule.batch_builder import CausalLMBatchBuilder
from speech_to_speech.model.orchestrator import AcousticSampler, Orchestrator
from speech_to_speech.model import orchestrator
from speech_to_speech.types import (
    AcousticCondition,
    AutoregressionExample,
    CausalLMBatch,
    IGNORE_INDEX,
    LongCatBatchSide,
    TranslationExample,
)
from helpers import MockQwen, MockTokenizer


class OrchestratorTest(unittest.TestCase):
    def test_forward_consumes_causal_lm_batch(self) -> None:
        tokenizer = MockTokenizer()
        model = Orchestrator(
            qwen3=MockQwen(),
            tokenizer=tokenizer,
            bpe_vocab_size=5,
            pretrained=False,
        )
        builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
        batch = builder.autoregression(
            AutoregressionExample(audio_ids=torch.tensor([0, 1, 2]))
        )

        output = model(batch)

        self.assertIsNotNone(output.loss)
        self.assertEqual(tuple(output.loss.shape), (1,))
        self.assertEqual(tuple(output.logits.shape), (5, 7))
        output.loss.mean().backward()

    def test_semantic_accuracy_uses_supervised_positions(self) -> None:
        model = Orchestrator(
            qwen3=MockQwen(),
            tokenizer=MockTokenizer(),
            bpe_vocab_size=5,
            pretrained=False,
        )
        batch = CausalLMBatch(
            input_ids=torch.zeros((2, 3), dtype=torch.long),
            attention_mask=torch.ones((2, 3), dtype=torch.long),
            labels=torch.tensor(
                [
                    [IGNORE_INDEX, 18, 19],
                    [20, IGNORE_INDEX, 21],
                ]
            ),
            logits_to_keep=torch.tensor(
                [
                    [0, 1],
                    [0, 2],
                    [1, 0],
                ]
            ),
        )
        logits = torch.full((3, model.lm_head.vocab_size), -1.0)
        logits[0, model.lm_head.to_head_ids(torch.tensor(18))] = 1.0
        logits[1, model.lm_head.to_head_ids(torch.tensor(16))] = 1.0
        logits[2, model.lm_head.to_head_ids(torch.tensor(20))] = 1.0
        output = CausalLMOutputWithPast(loss=torch.ones(2), logits=logits)

        accuracy = model.semantic_accuracy(batch, output)

        self.assertTrue(torch.equal(accuracy, torch.tensor(2.0 / 3.0)))

    def test_acoustic_condition_uses_target_label_shift(self) -> None:
        tokenizer = MockTokenizer()
        bpe = _bpe()
        model = Orchestrator(
            qwen3=MockQwen(),
            tokenizer=tokenizer,
            bpe_vocab_size=bpe.vocab_size,
            pretrained=False,
        )
        builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
        batch = builder.translation(
            TranslationExample(
                source_ids=torch.tensor([3]),
                target_ids=torch.tensor([4]),
            )
        )
        hidden_states = torch.arange(
            batch.input_ids.numel() * 4,
            dtype=torch.float,
        ).reshape(batch.input_ids.size(0), batch.input_ids.size(1), 4)

        condition = model.acoustic_condition(batch, bpe, hidden_states=hidden_states)

        self.assertEqual(condition.semantic_ids.tolist(), [[2, 1]])
        self.assertEqual(condition.mask.tolist(), [[True, True]])
        expected = hidden_states[:, 9:10].repeat_interleave(2, dim=1)
        self.assertTrue(torch.equal(condition.hidden_states, expected))

    def test_acoustic_flow_loss_uses_masked_velocity_target(self) -> None:
        tokenizer = MockTokenizer()
        bpe = _bpe()
        dit = MockDiT(hidden_size=4)
        model = Orchestrator(
            qwen3=MockQwen(),
            dit=dit,
            tokenizer=tokenizer,
            bpe_vocab_size=bpe.vocab_size,
            pretrained=False,
        )
        builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
        batch = builder.translation(
            TranslationExample(
                source_ids=torch.tensor([3]),
                target_ids=torch.tensor([4]),
            )
        )
        hidden_states = torch.zeros(batch.input_ids.size(0), batch.input_ids.size(1), 4)
        target_features = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]])
        noise = torch.zeros_like(target_features)
        timesteps = torch.tensor([0.25])

        loss = model.acoustic_flow_loss(
            batch,
            bpe,
            target_features,
            hidden_states=hidden_states,
            noise=noise,
            timesteps=timesteps,
        )

        expected = target_features.square().mean()
        self.assertTrue(torch.equal(loss, expected))
        self.assertEqual(dit.attention_mask.tolist(), [[1, 1]])
        self.assertEqual(tuple(dit.last_hidden_state.shape), (1, 2, 4))
        self.assertEqual(dit.timesteps.tolist(), [0.25])

    def test_acoustic_flow_loss_stats_returns_timesteps_and_row_loss(self) -> None:
        tokenizer = MockTokenizer()
        bpe = _bpe()
        dit = MockDiT(hidden_size=4)
        model = Orchestrator(
            qwen3=MockQwen(),
            dit=dit,
            tokenizer=tokenizer,
            bpe_vocab_size=bpe.vocab_size,
            pretrained=False,
        )
        builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
        batch = builder.translation(
            TranslationExample(
                source_ids=torch.tensor([3]),
                target_ids=torch.tensor([4]),
            )
        )
        hidden_states = torch.zeros(batch.input_ids.size(0), batch.input_ids.size(1), 4)
        target_features = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]])

        stats = model.acoustic_flow_loss_stats(
            batch,
            bpe,
            target_features,
            hidden_states=hidden_states,
            noise=torch.zeros_like(target_features),
            timesteps=torch.tensor([0.25]),
        )

        expected = target_features.square().mean()
        self.assertTrue(torch.equal(stats.loss, expected))
        self.assertTrue(torch.equal(stats.row_loss, expected.reshape(1)))
        self.assertTrue(torch.equal(stats.row_weight, torch.tensor([8.0])))
        self.assertTrue(torch.equal(stats.timesteps, torch.tensor([0.25])))

    def test_acoustic_flow_loss_aligns_condition_dtype_to_dit(self) -> None:
        tokenizer = MockTokenizer()
        bpe = _bpe()
        dit = MockDiT(hidden_size=4).to(dtype=torch.float32)
        model = Orchestrator(
            qwen3=MockQwen(),
            dit=dit,
            tokenizer=tokenizer,
            bpe_vocab_size=bpe.vocab_size,
            pretrained=False,
        )
        builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
        batch = builder.translation(
            TranslationExample(
                source_ids=torch.tensor([3]),
                target_ids=torch.tensor([4]),
            )
        )
        hidden_states = torch.zeros(
            batch.input_ids.size(0),
            batch.input_ids.size(1),
            4,
            dtype=torch.bfloat16,
        )
        target_features = torch.ones((1, 2, 4), dtype=torch.float32)

        model.acoustic_flow_loss(
            batch,
            bpe,
            target_features,
            hidden_states=hidden_states,
            noise=torch.zeros_like(target_features),
            timesteps=torch.tensor([0.5]),
        )

        self.assertIsNotNone(dit.last_hidden_state)
        self.assertEqual(dit.last_hidden_state.dtype, torch.float32)

    def test_acoustic_flow_loss_uses_pooled_source_acoustic_condition(self) -> None:
        tokenizer = MockTokenizer()
        bpe = _bpe()
        dit = MockDiT(hidden_size=4)
        model = Orchestrator(
            qwen3=MockQwen(),
            dit=dit,
            tokenizer=tokenizer,
            bpe_vocab_size=bpe.vocab_size,
            pretrained=False,
        )
        builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
        batch = builder.translation(
            TranslationExample(
                source_ids=torch.tensor([3]),
                target_ids=torch.tensor([4]),
            )
        )
        batch.source_audio = LongCatBatchSide(
            semantic_ids=torch.tensor([[1, 2, 0]]),
            semantic_mask=torch.tensor([[True, True, False]]),
            acoustic_ids=torch.zeros((1, 2, 3), dtype=torch.long),
            acoustic_mask=torch.tensor([[True, True, False]]),
        )
        hidden_states = torch.zeros(batch.input_ids.size(0), batch.input_ids.size(1), 4)

        model.acoustic_flow_loss(
            batch,
            bpe,
            torch.ones((1, 2, 4)),
            hidden_states=hidden_states,
            noise=torch.zeros((1, 2, 4)),
            timesteps=torch.tensor([0.5]),
            source_feature_extractor=FakeFeatureExtractor(
                torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 9.0, 9.0, 9.0]]])
            ),
        )

        self.assertIsNotNone(dit.acoustic_condition)
        self.assertTrue(
            torch.equal(
                dit.acoustic_condition,
                torch.tensor([[3.0, 4.0, 5.0, 6.0]]),
            )
        )

    def test_acoustic_condition_dropout_only_runs_in_training(self) -> None:
        tokenizer = MockTokenizer()
        bpe = _bpe()
        dit = MockDiT(hidden_size=4)
        model = Orchestrator(
            qwen3=MockQwen(),
            dit=dit,
            tokenizer=tokenizer,
            bpe_vocab_size=bpe.vocab_size,
            model_config=ModelConfig(acoustic_condition_dropout=1.0),
            pretrained=False,
        )
        builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
        batch = builder.translation(
            TranslationExample(
                source_ids=torch.tensor([3]),
                target_ids=torch.tensor([4]),
            )
        )
        batch.source_audio = LongCatBatchSide(
            semantic_ids=torch.tensor([[1, 2]]),
            semantic_mask=torch.tensor([[True, True]]),
            acoustic_ids=torch.zeros((1, 2, 2), dtype=torch.long),
            acoustic_mask=torch.tensor([[True, True]]),
        )
        hidden_states = torch.zeros(batch.input_ids.size(0), batch.input_ids.size(1), 4)
        feature_extractor = FakeFeatureExtractor(
            torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
        )

        model.eval()
        model.acoustic_flow_loss(
            batch,
            bpe,
            torch.ones((1, 2, 4)),
            hidden_states=hidden_states,
            noise=torch.zeros((1, 2, 4)),
            timesteps=torch.tensor([0.5]),
            source_feature_extractor=feature_extractor,
        )
        self.assertIsNotNone(dit.acoustic_condition)
        self.assertTrue(torch.equal(dit.acoustic_condition, torch.tensor([[3.0, 4.0, 5.0, 6.0]])))

        model.train()
        model.acoustic_flow_loss(
            batch,
            bpe,
            torch.ones((1, 2, 4)),
            hidden_states=hidden_states,
            noise=torch.zeros((1, 2, 4)),
            timesteps=torch.tensor([0.5]),
            source_feature_extractor=feature_extractor,
        )
        self.assertTrue(torch.equal(dit.acoustic_condition, torch.zeros((1, 4))))

    def test_forward_keeps_tail_supervised_logits(self) -> None:
        tokenizer = MockTokenizer()
        model = Orchestrator(
            qwen3=MockQwen(),
            tokenizer=tokenizer,
            bpe_vocab_size=5,
            pretrained=False,
        )
        builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
        batch = builder.autoregression(
            [
                AutoregressionExample(audio_ids=torch.tensor([0, 1, 2])),
                AutoregressionExample(audio_ids=torch.tensor([3])),
            ]
        )

        output = model(batch)

        self.assertEqual(batch.logits_to_keep, 5)
        self.assertEqual(tuple(output.logits.shape), (8, 7))
        self.assertIsNotNone(output.loss)
        self.assertEqual(tuple(output.loss.shape), (2,))

    def test_generate_acoustic_condition_returns_hidden_condition(self) -> None:
        tokenizer = MockTokenizer()
        qwen = ScriptedQwen(
            next_hidden_values=(
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                torch.tensor([0.0, 1.0, 0.0, 0.0]),
            )
        )
        model = Orchestrator(
            qwen3=qwen,
            tokenizer=tokenizer,
            bpe_vocab_size=5,
            pretrained=False,
        )
        _set_lm_head_weights(
            model,
            {
                19: torch.tensor([1.0, 0.0, 0.0, 0.0]),
                17: torch.tensor([0.0, 1.0, 0.0, 0.0]),
            },
        )
        builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
        batch = builder.translation_generation(torch.tensor([0]))

        condition = model.generate_acoustic_condition(
            batch,
            max_new_tokens=2,
            return_token_ids=True,
        )

        self.assertEqual(condition.token_ids.tolist(), [[19, 17]])
        self.assertEqual(condition.mask.tolist(), [[True]])
        self.assertEqual(tuple(condition.hidden_states.shape), (1, 1, 4))
        self.assertEqual(qwen.input_lengths, [batch.input_ids.size(1), 1])
        self.assertTrue(
            torch.equal(condition.hidden_states[0, 0], torch.tensor([0.0, 1.0, 0.0, 0.0]))
        )

    def test_generate_semantic_expands_bpe_tokens(self) -> None:
        tokenizer = MockTokenizer()
        qwen = ScriptedQwen(
            next_hidden_values=(
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                torch.tensor([0.0, 1.0, 0.0, 0.0]),
                torch.tensor([0.0, 0.0, 1.0, 0.0]),
            )
        )
        model = Orchestrator(
            qwen3=qwen,
            tokenizer=tokenizer,
            bpe_vocab_size=5,
            pretrained=False,
        )
        _set_lm_head_weights(
            model,
            {
                19: torch.tensor([1.0, 0.0, 0.0, 0.0]),
                20: torch.tensor([0.0, 1.0, 0.0, 0.0]),
                17: torch.tensor([0.0, 0.0, 1.0, 0.0]),
            },
        )
        builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
        batch = builder.translation_generation(torch.tensor([0]))

        generation = model.generate_semantic(
            batch,
            bpe=FakeBPE({1: [7, 8], 2: [9]}),
            max_new_tokens=3,
        )

        self.assertEqual(generation.token_ids.tolist(), [[19, 20, 17]])
        self.assertEqual(generation.semantic_ids.tolist(), [[7, 8, 9]])
        self.assertEqual(generation.semantic_mask.tolist(), [[True, True, True]])

    def test_generate_waveform_runs_full_sequence_decode_contract(self) -> None:
        tokenizer = MockTokenizer()
        qwen = ScriptedQwen(
            next_hidden_values=(
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                torch.tensor([0.0, 1.0, 0.0, 0.0]),
            )
        )
        model = Orchestrator(
            qwen3=qwen,
            tokenizer=tokenizer,
            bpe_vocab_size=5,
            pretrained=False,
        )
        _set_lm_head_weights(
            model,
            {
                19: torch.tensor([1.0, 0.0, 0.0, 0.0]),
                17: torch.tensor([0.0, 1.0, 0.0, 0.0]),
            },
        )
        builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
        batch = builder.translation_generation(torch.tensor([0]))
        codec = FakeCodec()
        acoustic = FakeAcousticGenerator(dim=4)

        generation = model.generate_waveform(
            batch,
            bpe=FakeBPE({1: [7, 8]}),
            codec=codec,
            acoustic_generator=acoustic,
            max_new_tokens=2,
        )

        self.assertEqual(generation.token_ids.tolist(), [[19, 17]])
        self.assertEqual(generation.semantic_ids.tolist(), [[7, 8]])
        self.assertEqual(generation.semantic_mask.tolist(), [[True, True]])
        self.assertEqual(tuple(generation.condition_hidden_states.shape), (1, 2, 4))
        self.assertEqual(tuple(generation.acoustic_features.shape), (1, 2, 4))
        self.assertEqual(tuple(generation.audio.shape), (1, 1, 6))
        self.assertEqual(codec.semantic_ids.tolist(), [[7, 8]])
        self.assertEqual(codec.acoustic_features.shape, (1, 2, 4))
        self.assertEqual(acoustic.condition.semantic_ids.tolist(), [[7, 8]])

    def test_generate_waveform_requires_acoustic_generator(self) -> None:
        tokenizer = MockTokenizer()
        model = Orchestrator(
            qwen3=ScriptedQwen(next_hidden_values=(torch.tensor([1.0, 0.0, 0.0, 0.0]),)),
            tokenizer=tokenizer,
            bpe_vocab_size=5,
            pretrained=False,
        )
        builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
        batch = builder.translation_generation(torch.tensor([0]))

        with self.assertRaisesRegex(RuntimeError, "requires an acoustic feature generator"):
            model.generate_waveform(
                batch,
                bpe=FakeBPE({}),
                codec=FakeCodec(),
                acoustic_generator=None,
                max_new_tokens=1,
            )

    def test_teacher_forced_waveform_decodes_from_label_hidden_condition(self) -> None:
        tokenizer = MockTokenizer()
        bpe = _bpe()
        model = Orchestrator(
            qwen3=MockQwen(),
            dit=MockDiT(hidden_size=4),
            tokenizer=tokenizer,
            bpe_vocab_size=bpe.vocab_size,
            pretrained=False,
        )
        builder = CausalLMBatchBuilder(model.embed_tokens, tokenizer=tokenizer)
        batch = builder.translation(
            TranslationExample(
                source_ids=torch.tensor([3]),
                target_ids=torch.tensor([4]),
            )
        )
        acoustic = FakeAcousticGenerator(dim=4)
        codec = FakeCodec()

        generation = model.teacher_forced_waveform(
            batch,
            bpe=bpe,
            codec=codec,
            acoustic_generator=acoustic,
        )

        self.assertEqual(generation.semantic_ids.tolist(), [[2, 1]])
        self.assertEqual(generation.semantic_mask.tolist(), [[True, True]])
        self.assertEqual(tuple(generation.acoustic_features.shape), (1, 2, 4))
        self.assertEqual(tuple(generation.audio.shape), (1, 1, 6))
        self.assertEqual(codec.semantic_ids.tolist(), [[2, 1]])
        self.assertEqual(acoustic.condition.semantic_ids.tolist(), [[2, 1]])

    def test_acoustic_feature_generator_runs_default_diagonal_sampler(self) -> None:
        tokenizer = MockTokenizer()
        dit = ConstantDiT(hidden_size=4)
        model = Orchestrator(
            qwen3=MockQwen(),
            dit=dit,
            tokenizer=tokenizer,
            bpe_vocab_size=5,
            pretrained=False,
        )
        condition = AcousticCondition(
            hidden_states=torch.ones((1, 3, 4)),
            semantic_ids=torch.tensor([[1, 2, 3]]),
            mask=torch.tensor([[True, True, False]]),
        )

        features = model.acoustic_feature_generator(num_steps=2, chunk_size=2)(condition)

        self.assertEqual(tuple(features.shape), (1, 3, 4))
        self.assertTrue(torch.equal(features[:, :2], torch.ones((1, 2, 4))))
        self.assertTrue(torch.equal(features[:, 2:], torch.zeros((1, 1, 4))))
        self.assertEqual(dit.forward_count, 3)

    def test_acoustic_feature_generator_can_use_serial_sampler(self) -> None:
        tokenizer = MockTokenizer()
        dit = ConstantDiT(hidden_size=4)
        model = Orchestrator(
            qwen3=MockQwen(),
            dit=dit,
            tokenizer=tokenizer,
            bpe_vocab_size=5,
            pretrained=False,
        )
        condition = AcousticCondition(
            hidden_states=torch.ones((1, 3, 4)),
            semantic_ids=torch.tensor([[1, 2, 3]]),
            mask=torch.tensor([[True, True, False]]),
        )

        features = model.acoustic_feature_generator(
            num_steps=2,
            chunk_size=2,
            sampler=AcousticSampler.SERIAL,
        )(condition)

        self.assertEqual(tuple(features.shape), (1, 3, 4))
        self.assertTrue(torch.equal(features[:, :2], torch.ones((1, 2, 4))))
        self.assertTrue(torch.equal(features[:, 2:], torch.zeros((1, 1, 4))))
        self.assertEqual(dit.forward_count, 4)

    def test_acoustic_feature_generator_uses_explicit_acoustic_condition(self) -> None:
        tokenizer = MockTokenizer()
        dit = ConditionDiT(hidden_size=4)
        model = Orchestrator(
            qwen3=MockQwen(),
            dit=dit,
            tokenizer=tokenizer,
            bpe_vocab_size=5,
            pretrained=False,
        )
        condition = AcousticCondition(
            hidden_states=torch.ones((1, 2, 4)),
            semantic_ids=torch.tensor([[1, 2]]),
            mask=torch.tensor([[True, True]]),
        )

        model.acoustic_feature_generator(
            num_steps=1,
            acoustic_condition=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        )(condition)

        self.assertEqual(dit.conditions, [[1.0, 2.0, 3.0, 4.0]])

    def test_default_trainable_policy_freezes_text_and_trains_audio(self) -> None:
        tokenizer = MockTokenizer()
        dit = MockDiT(hidden_size=4)
        model = Orchestrator(
            qwen3=MockQwen(),
            dit=dit,
            tokenizer=tokenizer,
            bpe_vocab_size=5,
            pretrained=False,
        )

        embedding = model.embed_tokens

        self.assertFalse(embedding.modality_embeddings[Modality.TEXT.value].weight.requires_grad)
        self.assertTrue(embedding.modality_embeddings[Modality.AUDIO.value].weight.requires_grad)
        self.assertTrue(embedding.special_embeddings["boa"].requires_grad)
        self.assertTrue(embedding.special_embeddings["eoa"].requires_grad)
        self.assertNotIn("<|im_start|>", embedding.special_embeddings)
        self.assertTrue(
            torch.equal(
                embedding.weight[1],
                embedding.modality_embeddings[Modality.TEXT.value].weight[1],
            )
        )
        self.assertTrue(any(parameter.requires_grad for parameter in dit.parameters()))

    def test_lora_parameters_are_trainable_when_peft_is_applied(self) -> None:
        tokenizer = MockTokenizer()
        qwen = MockPeftQwen()
        fake_peft = SimpleNamespace(get_peft_model=lambda model, config: model)
        with patch.dict(sys.modules, {"peft": fake_peft}):
            Orchestrator(
                qwen3=qwen,
                tokenizer=tokenizer,
                bpe_vocab_size=5,
                pretrained=True,
                lora_config=object(),
            )

        self.assertFalse(qwen.proj.weight.requires_grad)
        self.assertTrue(qwen.lora_adapter.requires_grad)

    def test_pretrained_load_can_disable_4bit_quantization(self) -> None:
        calls: list[object] = []

        def from_pretrained(
            name: str,
            *,
            trust_remote_code: bool,
            quantization_config: object | None,
        ) -> MockQwen:
            del name, trust_remote_code
            calls.append(quantization_config)
            return MockQwen()

        with patch.object(orchestrator.Qwen3Model, "from_pretrained", from_pretrained):
            Orchestrator(
                tokenizer=MockTokenizer(),
                bpe_vocab_size=5,
                model_config=ModelConfig(
                    load_in_4bit=False,
                    lora=LoRAConfig(enabled=False),
                ),
            )

        self.assertEqual(calls, [None])


class MockDiT(nn.Module):
    def __init__(self, *, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.null_acoustic_condition = nn.Parameter(torch.zeros(1, hidden_size))
        self.attention_mask: torch.Tensor | None = None
        self.last_hidden_state: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self.acoustic_condition: torch.Tensor | None = None

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        last_hidden_state: torch.Tensor,
        timesteps: torch.Tensor,
        acoustic_condition: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> BaseModelOutputWithPast:
        self.attention_mask = attention_mask
        self.last_hidden_state = last_hidden_state
        self.timesteps = timesteps
        self.acoustic_condition = acoustic_condition
        return BaseModelOutputWithPast(last_hidden_state=torch.zeros_like(x_t))


class ConstantDiT(nn.Module):
    def __init__(self, *, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.null_acoustic_condition = nn.Parameter(torch.zeros(1, hidden_size))
        self.forward_count = 0

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        last_hidden_state: torch.Tensor,
        timesteps: torch.Tensor,
        acoustic_condition: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> BaseModelOutputWithPast:
        del last_hidden_state, timesteps, acoustic_condition
        self.forward_count += 1
        velocity = attention_mask.to(dtype=x_t.dtype).unsqueeze(-1).expand_as(x_t)
        return BaseModelOutputWithPast(last_hidden_state=velocity)


class ConditionDiT(nn.Module):
    def __init__(self, *, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.null_acoustic_condition = nn.Parameter(torch.zeros(1, hidden_size))
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
        self.conditions.append([float(value) for value in acoustic_condition[0].detach().cpu()])
        return BaseModelOutputWithPast(last_hidden_state=torch.zeros_like(x_t))


class ScriptedQwen(nn.Module):
    def __init__(self, *, next_hidden_values: tuple[torch.Tensor, ...]) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=16, hidden_size=4)
        self.embed_tokens = nn.Embedding(16, 4)
        self.next_hidden_values = next_hidden_values
        self.calls = 0
        self.input_lengths: list[int] = []
        self.cache = object()

    def forward(
        self,
        *,
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor,
        **kwargs: object,
    ):
        del kwargs
        self.input_lengths.append(inputs_embeds.size(1))
        hidden = torch.zeros_like(inputs_embeds)
        value = self.next_hidden_values[min(self.calls, len(self.next_hidden_values) - 1)]
        del attention_mask
        positions = torch.full(
            (inputs_embeds.size(0),),
            inputs_embeds.size(1) - 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        hidden[torch.arange(inputs_embeds.size(0)), positions] = value.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        self.calls += 1
        return BaseModelOutputWithPast(last_hidden_state=hidden, past_key_values=self.cache)


class MockPeftQwen(MockQwen):
    def __init__(self) -> None:
        super().__init__()
        self.lora_adapter = nn.Parameter(torch.ones(4, 4))

    def print_trainable_parameters(self) -> None:
        pass


class FakeBPE:
    def __init__(self, expansions: dict[int, list[int]]) -> None:
        self.expansions = expansions

    def expand_ids(self, ids: list[int]) -> list[tuple[int]]:
        values: list[tuple[int]] = []
        for token_id in ids:
            values.extend((value,) for value in self.expansions[int(token_id)])
        return values


class FakeAcousticGenerator:
    def __init__(self, *, dim: int) -> None:
        self.dim = dim
        self.condition = None

    def __call__(self, condition):
        self.condition = condition
        batch, time = condition.mask.shape
        return torch.arange(batch * time * self.dim, dtype=torch.float).reshape(
            batch,
            time,
            self.dim,
        )


class FakeFeatureExtractor:
    def __init__(self, features: torch.Tensor) -> None:
        self.features = features

    def acoustic_codes_to_features(self, acoustic_ids: torch.Tensor) -> torch.Tensor:
        del acoustic_ids
        return self.features


class FakeCodec:
    def __init__(self) -> None:
        self.semantic_ids: torch.Tensor | None = None
        self.acoustic_features: torch.Tensor | None = None

    def decode_features(
        self,
        semantic_codes: torch.Tensor,
        acoustic_features: torch.Tensor,
    ) -> torch.Tensor:
        self.semantic_ids = semantic_codes
        self.acoustic_features = acoustic_features
        return torch.ones((semantic_codes.size(0), 1, semantic_codes.size(1) * 3))


def _set_lm_head_weights(model: Orchestrator, weights: dict[int, torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter in model.embed_tokens.special_embeddings.values():
            parameter.zero_()
        audio = model.embed_tokens.modality_embeddings[Modality.AUDIO.value]
        audio.weight.zero_()
        special_name_by_id = {
            token_id: name
            for name, token_id in model.embed_tokens.space.special_token_ids.items()
        }
        for token_id, value in weights.items():
            special_name = special_name_by_id.get(token_id)
            if special_name is not None and special_name in model.embed_tokens.special_embeddings:
                model.embed_tokens.special_embeddings[special_name].copy_(value)
                continue
            block = model.embed_tokens.space.modality_block(Modality.AUDIO)
            audio.weight[token_id - block.start].copy_(value)


def _bpe() -> CodecBPE:
    return CodecBPE.from_dict(
        {
            "codebook_sizes": [16],
            "tokens": {
                "0": [[0]],
                "1": [[1]],
                "2": [[2]],
                "3": [[0], [1]],
                "4": [[2], [1]],
            },
            "merges": [
                {"left": 0, "right": 1, "token_id": 3},
                {"left": 2, "right": 1, "token_id": 4},
            ],
            "strict": True,
        }
    )


if __name__ == "__main__":
    unittest.main()
