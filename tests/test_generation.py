from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from anydataset.types import Modality
from anytrain.idspace import Layout
from torch import Tensor, nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM

from speech_to_speech.callback.logging.sample import SampleLogger
from speech_to_speech.datamodule.module import Config as DataConfig
from speech_to_speech.datamodule.module import DataModule
from speech_to_speech.datamodule.types import ACOUSTIC_PAD_ID, ModelBatch, Task
from speech_to_speech.model.acoustic import SpeechToSpeechFlowModel
from speech_to_speech.model.base import Config as ModelConfig
from speech_to_speech.model.base import SemanticModel
from speech_to_speech.pl_module.generation import (
    Request,
    Result,
    generate,
    requests_from_batch,
)
from speech_to_speech.runtime.audio_tokenizer import NativeAudioTokenizer


class _Codec:
    acoustic_feature_dim = 2

    def __init__(self) -> None:
        self.decode_calls = 0

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor:
        self.decode_calls += 1
        return semantic_codes[..., 0].to(acoustic_features) + acoustic_features[..., 0]


class _Runtime:
    def __init__(self) -> None:
        self.layout = Layout(text=(0, 4), audio=(4, 8))
        self.audio_tokenizer = NativeAudioTokenizer(vocab_size=2)
        self.codec = _Codec()
        self.eos_token_id = 3
        self.boa_token_id = 6
        self.eoa_token_id = 7

    @property
    def codec_audio_range(self) -> tuple[int, int]:
        return 4, 6

    @property
    def audio_generation_allowed_ids(self) -> tuple[int, ...]:
        return 4, 5, 7

    def generation_allowed_ids(self, modality: Modality) -> tuple[int, ...]:
        if modality is Modality.TEXT:
            return 0, 1, 2, 3
        return self.audio_generation_allowed_ids

    def is_codec_audio_id(self, token_id: int) -> bool:
        start, end = self.codec_audio_range
        return start <= token_id < end


class _TinyCodec:
    acoustic_feature_dim = 8
    semantic_codebook = torch.randn(2, 8)

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        values = acoustic_codes[..., :1].to(dtype=torch.float32)
        return values.expand(*values.shape[:-1], self.acoustic_feature_dim)


class _TinyRuntime(_Runtime):
    def __init__(self) -> None:
        super().__init__()
        self.layout = Layout(text=(0, 8), audio=(8, 12))
        self.audio_tokenizer = NativeAudioTokenizer(vocab_size=2)
        self.codec = _TinyCodec()
        self.backbone = Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size=8,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                max_position_embeddings=32,
            )
        )
        self.eos_token_id = 3
        self.boa_token_id = 10
        self.eoa_token_id = 11

    @property
    def codec_audio_range(self) -> tuple[int, int]:
        return 8, 10

    @property
    def audio_generation_allowed_ids(self) -> tuple[int, ...]:
        return 8, 9, 11


class _GenerationModel(SpeechToSpeechFlowModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.runtime = _Runtime()
        self.backbone = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty(0))
        )
        self.calls: list[tuple[int, bool, int]] = []
        self.condition: Tensor | None = None
        self.sample_calls = 0

    def forward(
        self,
        input_ids: Tensor,
        *,
        acoustic_input_ids: Tensor | None = None,
        output_hidden_states: bool = False,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        generation_token_ids = kwargs.pop("_generation_token_ids", None)
        del kwargs
        cached_length = 0 if past_key_values is None else past_key_values.length
        source = (
            int(acoustic_input_ids.sum().item())
            if acoustic_input_ids is not None
            else 0
            if past_key_values is None
            else past_key_values.source
        )
        length = cached_length + input_ids.size(1)
        self.calls.append((input_ids.size(1), acoustic_input_ids is not None, source))

        next_id = {2: 4, 3: 5}.get(length, self.runtime.eoa_token_id)
        logits = torch.full(
            (*input_ids.shape, self.runtime.layout.vocab_size), float("-inf")
        )
        logits[:, -1, next_id] = 0
        if generation_token_ids is not None:
            logits = logits.index_select(-1, generation_token_ids)
        hidden = torch.zeros(*input_ids.shape, 2)
        hidden[:, -1] = torch.tensor([source, length])
        cache = SimpleNamespace(length=length, source=source) if use_cache else None
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=cache,
            hidden_states=(hidden,) if output_hidden_states else None,
        )

    def sample_acoustic(self, condition: Tensor) -> Tensor:
        self.sample_calls += 1
        self.condition = condition.clone()
        return torch.zeros_like(condition)


class GenerationTest(unittest.TestCase):
    def test_forward_skips_the_backbone_lm_head(self):
        model = SemanticModel(
            ModelConfig(
                semantic_audio_adapter=None,
                semantic_audio_output_adapter=None,
                acoustic_prompt_adapter=None,
            ),
            runtime_snapshot=_TinyRuntime(),
        ).eval()

        with patch.object(
            model.backbone.lm_head,
            "forward",
            side_effect=AssertionError("backbone LM head should not run"),
        ):
            output = model(torch.tensor([[1, 2]]))

        self.assertEqual(tuple(output.logits.shape), (1, 2, 12))

    def test_generation_only_computes_the_allowed_output_head(self):
        model = SemanticModel(
            ModelConfig(
                semantic_audio_adapter=None,
                semantic_audio_output_adapter=None,
                acoustic_prompt_adapter=None,
            ),
            runtime_snapshot=_TinyRuntime(),
        ).eval()

        with (
            patch.object(
                model,
                "text_logits",
                side_effect=AssertionError("text head should not run"),
            ),
            patch.object(
                model,
                "semantic_audio_logits",
                wraps=model.semantic_audio_logits,
            ) as semantic_audio_logits,
        ):
            generated = model.generate_semantic(
                torch.tensor([[1, 2]]),
                max_new_tokens=1,
                allowed_token_ids=model.runtime.audio_generation_allowed_ids,
                do_sample=False,
                use_cache=False,
            )

        self.assertIn(int(generated[0, -1]), model.runtime.audio_generation_allowed_ids)
        self.assertEqual(semantic_audio_logits.call_args.args[0].size(1), 1)

    def test_generation_rejects_invalid_constraints(self):
        model = SemanticModel(
            ModelConfig(
                semantic_audio_adapter=None,
                semantic_audio_output_adapter=None,
                acoustic_prompt_adapter=None,
            ),
            runtime_snapshot=_TinyRuntime(),
        ).eval()

        with self.assertRaisesRegex(ValueError, "duplicates"):
            model.generate_semantic(
                torch.tensor([[1, 2]]),
                max_new_tokens=1,
                allowed_token_ids=(8, 8, 11),
            )

        request = _request()
        request["task"] = Task.T2TT
        with self.assertRaisesRegex(ValueError, "source acoustic prompt"):
            generate([request], _GenerationModel(), max_new_tokens=1)

    def test_acoustic_prompt_adapter_bias_only_affects_prompt_positions(self):
        rt = _TinyRuntime()
        model = SemanticModel(
            ModelConfig(
                semantic_audio_adapter=None,
                semantic_audio_output_adapter=None,
                acoustic_prompt_adapter="linear",
            ),
            runtime_snapshot=rt,
        ).eval()
        with torch.no_grad():
            model.acoustic_prompt_adapter.weight.zero_()
            model.acoustic_prompt_adapter.bias.fill_(0.25)
            model.acoustic_prompt_gate.fill_(1)

        acoustic = model._acoustic_prompt_embedding(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[[2], [1]]]),
            torch.tensor([[1, 1]]),
            None,
        )

        self.assertTrue(torch.equal(acoustic[0, 0], torch.zeros(8)))
        self.assertTrue(torch.equal(acoustic[0, 1], torch.full((8,), 0.25)))
        self.assertTrue(torch.equal(acoustic[0, 2], torch.zeros(8)))

    def test_tiny_qwen_cache_matches_full_recompute(self):
        torch.manual_seed(0)
        rt = _TinyRuntime()
        model = SemanticModel(
            ModelConfig(
                semantic_audio_adapter=None,
                semantic_audio_output_adapter=None,
                acoustic_prompt_adapter=None,
            ),
            runtime_snapshot=rt,
        ).eval()
        model.acoustic_prompt_gate.data.fill_(1)
        kwargs = {
            "max_new_tokens": 3,
            "acoustic_input_ids": torch.tensor([[[2]]]),
            "acoustic_input_positions": torch.tensor([[0]]),
            "allowed_token_ids": tuple(range(8)),
            "do_sample": False,
        }

        cached = model.generate_semantic(
            torch.tensor([[1, 2]]), use_cache=True, **kwargs
        )
        full = model.generate_semantic(
            torch.tensor([[1, 2]]), use_cache=False, **kwargs
        )

        self.assertTrue(torch.equal(cached, full))

    def test_cached_audio_generation_matches_full_recompute(self):
        cached_model = _GenerationModel()
        cached = generate(
            [_request()],
            cached_model,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )[0]
        full_model = _GenerationModel()
        full = generate(
            [_request()],
            full_model,
            max_new_tokens=3,
            do_sample=False,
            use_cache=False,
        )[0]

        self.assertTrue(torch.equal(cached["token_ids"], torch.tensor([4, 5])))
        self.assertTrue(torch.equal(cached["token_ids"], full["token_ids"]))
        cached_audio = cached["audio"]
        full_audio = full["audio"]
        self.assertIsNotNone(cached_audio)
        self.assertIsNotNone(full_audio)
        self.assertTrue(
            torch.equal(cached_audio["features"], full_audio["features"])
        )
        self.assertTrue(
            torch.equal(cached_audio["waveform"], full_audio["waveform"])
        )
        self.assertEqual([call[0] for call in cached_model.calls], [2, 1, 1])
        self.assertEqual([call[0] for call in full_model.calls], [2, 3, 4])

    def test_cache_preserves_source_condition_and_collects_hidden_online(self):
        model = _GenerationModel()

        result = generate(
            [_request()],
            model,
            max_new_tokens=3,
            do_sample=False,
        )[0]

        self.assertEqual([call[1] for call in model.calls], [True, False, False])
        self.assertEqual([call[2] for call in model.calls], [3, 3, 3])
        self.assertTrue(
            torch.equal(
                model.condition,
                torch.tensor([[[3.0, 2.0], [3.0, 3.0]]]),
            )
        )
        self.assertEqual(model.sample_calls, 1)
        self.assertEqual(model.runtime.codec.decode_calls, 1)
        self.assertIsNotNone(result["audio"])

    def test_teacher_forcing_adapter_removes_target_and_acoustic_padding(self):
        batch = ModelBatch(
            input_ids=torch.tensor([[1, 6, 4, 7], [2, 6, 5, 7]]),
            labels=torch.tensor([[-100, -100, 4, 7], [-100, -100, 5, 7]]),
            acoustic_input_ids=torch.tensor([[[3], [ACOUSTIC_PAD_ID]], [[2], [1]]]),
            acoustic_input_positions=torch.tensor([[0, ACOUSTIC_PAD_ID], [0, 1]]),
            semantic_frame_labels=None,
            acoustic_labels=None,
            acoustic_label_positions=None,
            tasks=[Task.S2ST, Task.S2ST],
        )

        requests = requests_from_batch(batch)

        self.assertTrue(torch.equal(requests[0]["prompt_ids"], torch.tensor([1, 6])))
        first_acoustic = requests[0]["acoustic_prompt"]
        second_acoustic = requests[1]["acoustic_prompt"]
        self.assertIsNotNone(first_acoustic)
        self.assertIsNotNone(second_acoustic)
        self.assertTrue(
            torch.equal(first_acoustic["ids"], torch.tensor([[3]]))
        )
        self.assertTrue(
            torch.equal(second_acoustic["ids"], torch.tensor([[2], [1]]))
        )

    def test_sample_logger_reuses_one_generation_result(self):
        batch = ModelBatch(
            input_ids=torch.tensor([[1, 6, 4, 7]]),
            labels=torch.tensor([[-100, -100, 4, 7]]),
            acoustic_input_ids=None,
            acoustic_input_positions=None,
            semantic_frame_labels=None,
            acoustic_labels=None,
            acoustic_label_positions=None,
            tasks=[Task.TTS],
        )
        result = Result(
            token_ids=torch.tensor([4]),
            audio={
                "features": torch.zeros(1, 2),
                "waveform": torch.zeros(1, 8),
            },
        )
        module = SimpleNamespace(generate=Mock(return_value=[result]))
        datamodule = SimpleNamespace(collator=Mock(return_value=batch))
        experiment = Mock()
        trainer = SimpleNamespace(
            global_step=0,
            logger=SimpleNamespace(experiment=experiment),
            datamodule=datamodule,
        )
        logger = SampleLogger([0], every_n_steps=1)
        logger.samples = [Mock()]
        trainer.is_global_zero = True

        logger.on_train_batch_start(trainer, module, None, 0)

        module.generate.assert_called_once()
        experiment.add_audio.assert_called_once()

    def test_sample_logger_loads_samples_from_real_datamodule(self):
        samples = [Mock(), Mock()]
        config = DataConfig(
            codec="longcat",
            dataloader={"batch_size": 1, "num_workers": 0},
        )
        datamodule = DataModule(config, {Task.TTS: 1.0})
        with patch(
            "zhuyin.datasets.wmt19_tts.wmt19_tts_codec",
            return_value=samples,
        ):
            datamodule.setup()
        trainer = SimpleNamespace(is_global_zero=True, datamodule=datamodule)
        logger = SampleLogger([1, 0], every_n_steps=1)

        logger.on_fit_start(trainer, SimpleNamespace())

        self.assertEqual(logger.samples, [samples[1], samples[0]])

    def test_sample_logger_skips_nonzero_ranks(self):
        module = SimpleNamespace(generate=Mock())
        trainer = SimpleNamespace(global_step=0, is_global_zero=False)
        logger = SampleLogger([0], every_n_steps=1)

        logger.on_train_batch_start(trainer, module, None, 0)

        module.generate.assert_not_called()


def _request() -> Request:
    return Request(
        prompt_ids=torch.tensor([1, 6]),
        task=Task.S2ST,
        acoustic_prompt={
            "ids": torch.tensor([[3]]),
            "positions": torch.tensor([0]),
        },
    )


if __name__ == "__main__":
    unittest.main()
