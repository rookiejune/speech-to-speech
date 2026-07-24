from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import torch
from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioView,
    Lang,
    Modality,
    Role,
    TextItem,
    TextMeta,
    TextView,
)
from anytrain.idspace import Layout
from torch import Tensor, nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from speech_to_speech.callback.logging.task_sample import TaskSampleLogger
from speech_to_speech.datamodule.module import Config as DataConfig
from speech_to_speech.datamodule.module import DataModule
from speech_to_speech.datamodule.types import ACOUSTIC_PAD_ID, ModelBatch
from speech_to_speech.model import ToyConfig
from speech_to_speech.model.acoustic import FlowModel
from speech_to_speech.model.base import Config as ModelConfig
from speech_to_speech.model.base import TokenModel
from speech_to_speech.generation import (
    Request,
    Result,
    decode_generated_audio,
    decode_generated_codes,
    generate_responses,
)
from speech_to_speech.generation.batch import requests_from_batch
from speech_to_speech.runtime.audio_tokenizer import (
    FlattenedAudioTokenizer,
    NativeAudioTokenizer,
)
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import Runtime
from speech_to_speech.task import Task


class _Codec:
    acoustic_feature_dim = 2
    acoustic_codebook_sizes = (8,)
    sample_rate = 16_000

    def __init__(self) -> None:
        self.decode_calls = 0

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor:
        self.decode_calls += 1
        return semantic_codes[..., 0].to(acoustic_features) + acoustic_features[..., 0]

    def decode(self, codes: Tensor) -> Tensor:
        self.decode_calls += 1
        return codes[..., 0].float()


class _UnifiedCodec(_Codec):
    acoustic_codebook_sizes = ()

    def decode(self, codes: Tensor) -> Tensor:
        self.decode_calls += 1
        return codes[..., 0].float()


class _Runtime:
    def __init__(self) -> None:
        self.layout = Layout(text=(0, 4), audio=(4, 8))
        self.audio_tokenizer = NativeAudioTokenizer(vocab_size=2)
        self.codec = _Codec()
        self.eos_token_id = 3
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.boa_token_id = 6
        self.eoa_token_id = 7

    @property
    def codec_audio_range(self) -> tuple[int, int]:
        return 4, 6

    @property
    def acoustic_side_channel(self) -> bool:
        return bool(self.codec.acoustic_codebook_sizes)

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
    acoustic_codebook_sizes = (8,)
    semantic_codebook = torch.randn(2, 8)
    sample_rate = 16_000

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        values = acoustic_codes[..., :1].to(dtype=torch.float32)
        return values.expand(*values.shape[:-1], self.acoustic_feature_dim)

    def decode(self, codes: Tensor) -> Tensor:
        return codes[..., 0].float()


class _TinyRuntime(_Runtime):
    def __init__(self) -> None:
        super().__init__()
        self.layout = Layout(text=(0, 8), audio=(8, 12))
        self.audio_tokenizer = NativeAudioTokenizer(vocab_size=2)
        self.codec = _TinyCodec()
        self.eos_token_id = 3
        self.boa_token_id = 10
        self.eoa_token_id = 11

    @property
    def codec_audio_range(self) -> tuple[int, int]:
        return 8, 10

    @property
    def audio_generation_allowed_ids(self) -> tuple[int, ...]:
        return 8, 9, 11


class _GenerationModel(FlowModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.runtime = _Runtime()
        self.layout = self.runtime.layout
        self.audio_token_frame_spans = torch.tensor([1, 1])
        self.backbone = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty(0))
        )
        self.calls: list[tuple[int, bool, int, int]] = []
        self.condition: Tensor | None = None
        self.sample_calls = 0

    def generation_step(
        self,
        input_ids: Tensor,
        *,
        acoustic_prompt_codes: Tensor | None = None,
        output_hidden_states: bool = False,
        past_key_values=None,
        use_cache: bool = False,
        token_ids: Tensor | None = None,
        modality: Modality | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del kwargs
        cached_length = 0 if past_key_values is None else past_key_values.length
        source = (
            int(acoustic_prompt_codes.sum().item())
            if acoustic_prompt_codes is not None
            else 0
            if past_key_values is None
            else past_key_values.source
        )
        length = cached_length + input_ids.size(1)
        self.calls.append(
            (
                input_ids.size(1),
                acoustic_prompt_codes is not None,
                source,
                input_ids.size(0),
            )
        )

        next_id = {2: 4, 3: 5}.get(length, self.runtime.eoa_token_id)
        logits = torch.full(
            (*input_ids.shape, self.runtime.layout.vocab_size), float("-inf")
        )
        logits[:, -1, next_id] = 0
        if token_ids is not None:
            logits = logits.index_select(-1, token_ids)
        elif modality is not None:
            start, end = self.layout.blocks[modality.value]
            logits = logits[..., start:end]
        hidden = torch.zeros(*input_ids.shape, 2)
        hidden[:, -1] = torch.tensor([source, length])
        cache = SimpleNamespace(length=length, source=source) if use_cache else None
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=cache,
            hidden_states=(hidden,) if output_hidden_states else None,
        )

    def sample_acoustic_features(
        self,
        condition: Tensor,
        *,
        mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        del mask, generator
        self.sample_calls += 1
        self.condition = condition.clone()
        return torch.zeros_like(condition)


class _TokenGenerationModel(TokenModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.runtime = _Runtime()
        self.layout = self.runtime.layout
        self.audio_token_frame_spans = torch.tensor([1, 1])
        self.backbone = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty(0))
        )
        self.calls: list[tuple[int, bool, int, int]] = []

    generation_step = _GenerationModel.generation_step


class _UnifiedGenerationModel(_GenerationModel):
    def __init__(self) -> None:
        super().__init__()
        self.runtime.codec = _UnifiedCodec()


class _FullSequenceCodec:
    acoustic_codebook_sizes = (10,)
    sample_rate = 16_000

    def __init__(self) -> None:
        self.decode_calls = 0
        self.decoded_codes: Tensor | None = None

    def decode(self, codes: Tensor) -> Tensor:
        self.decode_calls += 1
        self.decoded_codes = codes.clone()
        return codes.sum(dim=-1).float()


class _FullSequenceRuntime:
    def __init__(self) -> None:
        self.audio_tokenizer = FlattenedAudioTokenizer(
            codebook_sizes=(4, 10),
            codec_name="longcat",
        )
        self.codec = _FullSequenceCodec()
        self.layout = Layout(text=(0, 4), audio=(4, 4 + self.audio_tokenizer.vocab_size + 2))
        self.pad_token_id = 0
        self.eos_token_id = 3
        self.bos_token_id = 1
        self.boa_token_id = self.codec_audio_range[1]
        self.eoa_token_id = self.boa_token_id + 1

    @property
    def acoustic_side_channel(self) -> bool:
        return False

    @property
    def codec_audio_range(self) -> tuple[int, int]:
        start, _ = self.layout.blocks["audio"]
        return start, start + self.audio_tokenizer.vocab_size

    @property
    def audio_generation_allowed_ids(self) -> tuple[int, ...]:
        start, end = self.codec_audio_range
        return (*range(start, end), self.eoa_token_id)

    def generation_allowed_ids(self, modality: Modality) -> tuple[int, ...]:
        if modality is Modality.TEXT:
            return 0, 1, 2, 3
        return self.audio_generation_allowed_ids

    def is_codec_audio_id(self, token_id: int) -> bool:
        start, end = self.codec_audio_range
        return start <= token_id < end


class _FullSequenceGenerationModel(TokenModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.runtime = _FullSequenceRuntime()
        self.layout = self.runtime.layout
        self.audio_token_frame_spans = torch.tensor(
            self.runtime.audio_tokenizer.frame_spans(
                range(self.runtime.audio_tokenizer.vocab_size)
            )
        )
        self.backbone = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty(0))
        )
        start, _ = self.runtime.codec_audio_range
        self._tokens = [
            start + token_id
            for token_id in self.runtime.audio_tokenizer.encode(
                torch.tensor([[1, 5], [2, 6]])
            ).tolist()
        ]
        self._tokens.append(self.runtime.eoa_token_id)
        self._step = 0
        self.sample_calls = 0

    def generation_step(
        self,
        input_ids: Tensor,
        *,
        output_hidden_states: bool = False,
        past_key_values=None,
        use_cache: bool = False,
        token_ids: Tensor | None = None,
        modality: Modality | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del kwargs, output_hidden_states, past_key_values, use_cache, token_ids
        next_id = self._tokens[min(self._step, len(self._tokens) - 1)]
        self._step += 1
        if modality is Modality.AUDIO:
            start, end = self.layout.blocks[Modality.AUDIO.value]
            logits = torch.full((input_ids.size(0), input_ids.size(1), end - start), float("-inf"))
            logits[:, -1, next_id - start] = 0
        else:
            logits = torch.full(
                (input_ids.size(0), input_ids.size(1), self.layout.vocab_size),
                float("-inf"),
            )
            logits[:, -1, next_id] = 0
        return CausalLMOutputWithPast(logits=logits)

    def generate_audio_features(self, **kwargs) -> None:
        del kwargs
        self.sample_calls += 1
        raise AssertionError("full codec sequence must not use acoustic generation")


class _RegisteredBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(1, 1)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


class _RegisteredGenerationModel(_GenerationModel):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _RegisteredBackbone()


class _VariableStopModel(_UnifiedGenerationModel):
    def __init__(self) -> None:
        super().__init__()
        self.step = 0
        self.batch_sizes: list[int] = []
        self.cache_selections: list[list[int]] = []

    def generation_step(self, input_ids: Tensor, **kwargs) -> CausalLMOutputWithPast:
        self.batch_sizes.append(input_ids.size(0))
        generation_token_ids = kwargs.get("token_ids")
        generation_modality = kwargs.get("modality")
        use_cache = kwargs["use_cache"]
        token_ids = (
            torch.tensor([self.runtime.eos_token_id, 1], device=input_ids.device)
            if self.step == 0
            else torch.full(
                (input_ids.size(0),),
                self.runtime.eos_token_id,
                device=input_ids.device,
            )
        )
        self.step += 1
        if generation_token_ids is not None:
            local = torch.stack(
                [
                    (generation_token_ids == token_id).nonzero()[0, 0]
                    for token_id in token_ids
                ]
            )
            output_size = generation_token_ids.numel()
        else:
            start, end = self.layout.blocks[generation_modality.value]
            local = token_ids - start
            output_size = end - start
        logits = torch.full(
            (input_ids.size(0), 1, output_size),
            float("-inf"),
            device=input_ids.device,
        )
        logits[torch.arange(input_ids.size(0)), 0, local] = 0
        cache = (
            SimpleNamespace(
                length=self.step,
                source=0,
                batch_select_indices=lambda indices: self.cache_selections.append(
                    indices.tolist()
                ),
            )
            if use_cache
            else None
        )
        return CausalLMOutputWithPast(logits=logits, past_key_values=cache)


class GenerationTest(unittest.TestCase):
    def test_frame_span_buffer_follows_the_backbone_device(self):
        runtime = _TinyRuntime()
        model = TokenModel(
            _model_config(),
            runtime=runtime,
        ).to(device="meta")

        self.assertEqual(model.audio_token_frame_spans.device.type, "meta")
        self.assertNotIn("audio_token_frame_spans", model.state_dict())

    def test_text_generation_excludes_padding_and_bos(self):
        rt = Runtime(RuntimeConfig())
        rt.__dict__["layout"] = Layout(text=(0, 4), audio=(4, 8))
        rt.__dict__["pad_token_id"] = 0
        rt.__dict__["bos_token_id"] = 1

        allowed = rt.generation_allowed_ids(Modality.TEXT)

        self.assertEqual(allowed, (2, 3))

    def test_modality_generation_masks_special_tokens(self):
        model = TokenModel(
            _model_config(),
            runtime=_TinyRuntime(),
        ).eval()

        def text_logits(hidden_state: Tensor, local_ids=None) -> Tensor:
            self.assertIsNone(local_ids)
            logits = hidden_state.new_zeros(*hidden_state.shape[:-1], 8)
            logits[..., 0] = 100
            logits[..., 1] = 90
            logits[..., 2] = 80
            return logits

        def audio_logits(hidden_state: Tensor, local_ids=None) -> Tensor:
            self.assertIsNone(local_ids)
            logits = hidden_state.new_zeros(*hidden_state.shape[:-1], 4)
            logits[..., 2] = 100
            logits[..., 0] = 90
            return logits

        with patch.object(model, "text_logits", side_effect=text_logits):
            text = model.generate_tokens(
                torch.tensor([[2, 3]]),
                max_new_tokens=1,
                generation_modality=Modality.TEXT,
                do_sample=False,
                use_cache=False,
            )
        with patch.object(model, "semantic_audio_logits", side_effect=audio_logits):
            audio = model.generate_tokens(
                torch.tensor([[2, 3]]),
                max_new_tokens=1,
                generation_modality=Modality.AUDIO,
                do_sample=False,
                use_cache=False,
            )

        self.assertEqual(int(text[0, -1]), 2)
        self.assertEqual(int(audio[0, -1]), 8)

    def test_forward_skips_the_backbone_lm_head(self):
        model = TokenModel(
            _model_config(),
            runtime=_TinyRuntime(),
        ).eval()

        with patch.object(
            model.backbone.lm_head,
            "forward",
            side_effect=AssertionError("backbone LM head should not run"),
        ):
            output = model(torch.tensor([[1, 2]]))

        self.assertEqual(tuple(output.logits.shape), (1, 2, 12))

    def test_generation_only_computes_the_allowed_output_head(self):
        model = TokenModel(
            _model_config(),
            runtime=_TinyRuntime(),
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
            generated = model.generate_tokens(
                torch.tensor([[1, 2]]),
                max_new_tokens=1,
                generation_modality=Modality.AUDIO,
                do_sample=False,
                use_cache=False,
            )

        self.assertIn(int(generated[0, -1]), model.runtime.audio_generation_allowed_ids)
        self.assertEqual(semantic_audio_logits.call_args.args[0].size(1), 1)

    def test_generation_rejects_invalid_constraints(self):
        model = TokenModel(
            _model_config(),
            runtime=_TinyRuntime(),
        ).eval()

        with self.assertRaisesRegex(ValueError, "duplicates"):
            model.generate_tokens(
                torch.tensor([[1, 2]]),
                max_new_tokens=1,
                allowed_token_ids=(8, 8, 11),
            )

        with self.assertRaisesRegex(ValueError, "unsupported generation modality"):
            model.generate_tokens(
                torch.tensor([[1, 2]]),
                max_new_tokens=0,
                generation_modality=Modality.IMAGE,
            )

        request = _request()
        request["task"] = cast(Task, "tts")
        with self.assertRaisesRegex(TypeError, "must be a Task"):
            generate_responses([request], _GenerationModel(), max_new_tokens=1)

    def test_token_only_generation_with_acoustic_codec_uses_semantic_decode(self):
        model = _TokenGenerationModel()

        result = generate_responses(
            [_request()],
            model,
            max_new_tokens=3,
            do_sample=False,
        )[0]

        self.assertEqual(model.runtime.codec.decode_calls, 1)
        self.assertIsNotNone(result["audio"])
        audio = result["audio"]
        if audio is None:
            self.fail("audio generation did not return an audio payload")
        self.assertIsNone(audio["features"])
        torch.testing.assert_close(audio["waveform"], torch.tensor([0.0, 1.0]))

    def test_generated_audio_decode_validates_token_ids_before_codec_work(self):
        codec = Mock()
        tokenizer = NativeAudioTokenizer(vocab_size=2)

        with self.assertRaisesRegex(TypeError, "integer ids"):
            decode_generated_audio(
                torch.tensor([[4.5]]),
                torch.zeros(1, 1, 2),
                codec=codec,
                audio_tokenizer=tokenizer,
                audio_token_range=(4, 6),
            )
        for dtype in (torch.uint16, torch.uint64):
            with (
                self.subTest(dtype=dtype),
                self.assertRaisesRegex(TypeError, "signed dtype"),
            ):
                decode_generated_audio(
                    torch.tensor([[4]], dtype=dtype),
                    torch.zeros(1, 1, 2),
                    codec=codec,
                    audio_tokenizer=tokenizer,
                    audio_token_range=(4, 6),
                )
        with self.assertRaisesRegex(ValueError, "shape"):
            decode_generated_audio(
                torch.tensor([4]),
                torch.zeros(1, 1, 2),
                codec=codec,
                audio_tokenizer=tokenizer,
                audio_token_range=(4, 6),
            )
        with self.assertRaisesRegex(ValueError, "codec-decodable"):
            decode_generated_codes(
                torch.tensor([[6]]),
                torch.zeros(1, 1, 1, dtype=torch.long),
                codec=codec,
                audio_tokenizer=tokenizer,
                audio_token_range=(4, 6),
            )
        codec.acoustic_codes_to_features.assert_not_called()

    def test_generation_validates_request_prompts_before_padding(self):
        invalid = (
            (torch.tensor([], dtype=torch.long), "at least one token"),
            (torch.tensor([[4, 6]]), "1 dimensions"),
            (torch.tensor([4.5, 6.0]), "integer ids"),
            (torch.tensor([4, 6], dtype=torch.uint64), "signed dtype"),
            (torch.tensor([4, 8]), "runtime layout"),
        )
        for prompt_ids, message in invalid:
            request = _request()
            request["prompt_ids"] = prompt_ids
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    generate_responses([request], _GenerationModel(), max_new_tokens=1)

    def test_acoustic_codec_without_audio_model_decodes_semantic_tokens(self):
        model = TokenModel(
            _model_config(),
            runtime=_TinyRuntime(),
        ).eval()
        sequence = torch.tensor([[1, 2, 8, 9, 11]])
        request = Request(
            prompt_ids=torch.tensor([1, 2]),
            task=Task.TTS,
        )

        with patch.object(model, "generate_tokens", return_value=sequence):
            result = generate_responses([request], model, max_new_tokens=3)[0]

        self.assertTrue(torch.equal(result["response_ids"], torch.tensor([8, 9])))
        self.assertIsNotNone(result["audio"])
        self.assertIsNone(result["audio"]["features"])
        self.assertTrue(
            torch.equal(result["audio"]["waveform"], torch.tensor([0.0, 1.0]))
        )

    def test_audio_generation_accepts_a_registered_module_backbone(self):
        model = _RegisteredGenerationModel()

        result = generate_responses(
            [_request()],
            model,
            max_new_tokens=3,
            do_sample=False,
        )[0]

        self.assertIn("backbone", model._modules)
        self.assertIsNotNone(result["audio"])
        self.assertEqual(model.sample_calls, 1)

    def test_tiny_qwen_cache_matches_full_recompute(self):
        torch.manual_seed(0)
        rt = _TinyRuntime()
        model = TokenModel(
            _model_config(),
            runtime=rt,
        ).eval()
        kwargs = {
            "max_new_tokens": 3,
            "allowed_token_ids": tuple(range(8)),
            "do_sample": False,
        }

        cached = model.generate_tokens(torch.tensor([[1, 2]]), use_cache=True, **kwargs)
        full = model.generate_tokens(torch.tensor([[1, 2]]), use_cache=False, **kwargs)

        self.assertTrue(torch.equal(cached, full))

    def test_tiny_qwen_cache_compacts_finished_rows(self):
        def generate(use_cache: bool) -> tuple[Tensor, list[int]]:
            model = TokenModel(
                _model_config(),
                runtime=_TinyRuntime(),
            ).eval()
            generation_step = model.generation_step
            batch_sizes: list[int] = []

            def variable_step(input_ids: Tensor, **kwargs) -> CausalLMOutputWithPast:
                batch_sizes.append(input_ids.size(0))
                output = generation_step(input_ids, **kwargs)
                next_ids = torch.where(
                    input_ids[:, -1].eq(4) | input_ids[:, -1].eq(2),
                    model.runtime.eos_token_id,
                    2,
                )
                logits = torch.full_like(output.logits, float("-inf"))
                logits[torch.arange(input_ids.size(0)), 0, next_ids] = 0
                output.logits = logits
                return output

            with patch.object(model, "generation_step", side_effect=variable_step):
                generated = model.generate_tokens(
                    torch.tensor([[1, 4], [1, 5]]),
                    max_new_tokens=2,
                    stop_token_id=model.runtime.eos_token_id,
                    generation_modality=Modality.TEXT,
                    do_sample=False,
                    use_cache=use_cache,
                )
            return generated, batch_sizes

        cached, cached_batch_sizes = generate(True)
        full, full_batch_sizes = generate(False)

        self.assertTrue(torch.equal(cached, full))
        self.assertTrue(torch.equal(cached, torch.tensor([[1, 4, 3, 3], [1, 5, 2, 3]])))
        self.assertEqual(cached_batch_sizes, [2, 1])
        self.assertEqual(full_batch_sizes, [2, 1])

    def test_cached_audio_generation_matches_full_recompute(self):
        cached_model = _GenerationModel()
        cached = generate_responses(
            [_request()],
            cached_model,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )[0]
        full_model = _GenerationModel()
        full = generate_responses(
            [_request()],
            full_model,
            max_new_tokens=3,
            do_sample=False,
            use_cache=False,
        )[0]

        self.assertTrue(torch.equal(cached["response_ids"], torch.tensor([4, 5])))
        self.assertTrue(torch.equal(cached["response_ids"], full["response_ids"]))
        cached_audio = cached["audio"]
        full_audio = full["audio"]
        self.assertIsNotNone(cached_audio)
        self.assertIsNotNone(full_audio)
        self.assertTrue(torch.equal(cached_audio["features"], full_audio["features"]))
        self.assertTrue(torch.equal(cached_audio["waveform"], full_audio["waveform"]))
        self.assertEqual([call[0] for call in cached_model.calls], [2, 1, 1])
        self.assertEqual([call[0] for call in full_model.calls], [2, 3, 4])

    def test_unified_audio_generation_decodes_semantic_tokens_directly(self):
        model = _UnifiedGenerationModel()
        result = generate_responses(
            [_request()],
            model,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
        )[0]

        self.assertTrue(torch.equal(result["response_ids"], torch.tensor([4, 5])))
        self.assertIsNotNone(result["audio"])
        self.assertIsNone(result["audio"]["features"])
        self.assertEqual(model.sample_calls, 0)
        self.assertEqual(model.runtime.codec.decode_calls, 1)

    def test_full_codec_sequence_generation_decodes_complete_codes(self):
        model = _FullSequenceGenerationModel()
        request = Request(prompt_ids=torch.tensor([1]), task=Task.TTS)

        result = generate_responses(
            [request],
            model,
            max_new_tokens=8,
            do_sample=False,
            use_cache=False,
        )[0]

        start, _ = model.runtime.codec_audio_range
        expected_local = model.runtime.audio_tokenizer.encode(
            torch.tensor([[1, 5], [2, 6]])
        )
        expected_response = expected_local + start
        self.assertTrue(torch.equal(result["response_ids"], expected_response))
        self.assertIsNotNone(result["audio"])
        audio = result["audio"]
        if audio is None:
            self.fail("audio generation did not return an audio payload")
        self.assertIsNone(audio["features"])
        self.assertEqual(model.sample_calls, 0)
        self.assertEqual(model.runtime.codec.decode_calls, 1)
        self.assertTrue(
            torch.equal(
                model.runtime.codec.decoded_codes,
                torch.tensor([[[1, 5], [2, 6]]]),
            )
        )
        self.assertTrue(torch.equal(audio["waveform"], torch.tensor([6.0, 8.0])))

    def test_generation_batches_variable_length_requests(self):
        model = _UnifiedGenerationModel()
        frame_spans = model.runtime.audio_tokenizer.frame_spans
        model.runtime.audio_tokenizer.frame_spans = Mock(wraps=frame_spans)
        first = _request()
        second = _request()
        second["prompt_ids"] = torch.tensor([2, 1, 6])

        results = generate_responses(
            [first, second], model, max_new_tokens=3, do_sample=False
        )

        self.assertEqual(len(results), 2)
        self.assertEqual([call[3] for call in model.calls], [2, 2])
        self.assertEqual(model.runtime.audio_tokenizer.frame_spans.call_count, 0)
        self.assertEqual(model.runtime.codec.decode_calls, 1)

    def test_generation_reuses_frame_counts_and_batches_acoustic_decode(self):
        model = _GenerationModel()
        model.runtime.audio_tokenizer.frame_spans = Mock(
            side_effect=AssertionError("service must reuse model frame counts")
        )

        results = generate_responses(
            [_request(), _request()],
            model,
            max_new_tokens=3,
            do_sample=False,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(model.runtime.codec.decode_calls, 1)
        for result in results:
            audio = result["audio"]
            self.assertIsNotNone(audio)
            self.assertEqual(audio["features"].size(0), 2)

    def test_batch_generation_tracks_stop_per_row(self):
        requests = [
            Request(prompt_ids=torch.tensor([1]), task=Task.T2TT),
            Request(prompt_ids=torch.tensor([2, 1]), task=Task.T2TT),
        ]

        for use_cache in (False, True):
            with self.subTest(use_cache=use_cache):
                model = _VariableStopModel()
                results = generate_responses(
                    requests,
                    model,
                    max_new_tokens=3,
                    do_sample=False,
                    use_cache=use_cache,
                )

                self.assertEqual(results[0]["response_ids"].numel(), 0)
                self.assertTrue(
                    torch.equal(results[1]["response_ids"], torch.tensor([1]))
                )
                self.assertEqual(model.batch_sizes, [2, 1])
                self.assertEqual(model.cache_selections, [[1]] if use_cache else [])

    def test_cache_preserves_source_condition_and_collects_hidden_online(self):
        model = _GenerationModel()

        result = generate_responses(
            [_request()],
            model,
            max_new_tokens=3,
            do_sample=False,
        )[0]

        self.assertEqual([call[1] for call in model.calls], [False, False, False])
        self.assertEqual([call[2] for call in model.calls], [0, 0, 0])
        self.assertTrue(
            torch.equal(
                model.condition,
                torch.tensor([[[0.0, 2.0], [0.0, 3.0]]]),
            )
        )
        self.assertEqual(model.sample_calls, 1)
        self.assertEqual(model.runtime.codec.decode_calls, 1)
        self.assertIsNotNone(result["audio"])

    def test_teacher_forcing_adapter_removes_target_padding(self):
        batch = ModelBatch(
            input_ids=torch.tensor([[1, 6, 4, 7], [2, 6, 5, 7]]),
            token_labels=torch.tensor([[-100, -100, 4, 7], [-100, -100, 5, 7]]),
            acoustic_target=None,
            tasks=[Task.S2ST, Task.S2ST],
            pad_token_id=0,
        )

        requests = requests_from_batch(batch)

        self.assertTrue(torch.equal(requests[0]["prompt_ids"], torch.tensor([1, 6])))
        self.assertTrue(torch.equal(requests[1]["prompt_ids"], torch.tensor([2, 6])))

    def test_task_sample_logger_reuses_one_generation_result(self):
        batch = ModelBatch(
            input_ids=torch.tensor([[1, 6, 4, 7]]),
            token_labels=torch.tensor([[-100, -100, 4, 7]]),
            acoustic_target=None,
            tasks=[Task.TTS],
            pad_token_id=0,
        )
        result = Result(
            response_ids=torch.tensor([4]),
            audio={
                "features": torch.zeros(1, 2),
                "waveform": torch.zeros(1, 8),
                "sample_rate": 16_000,
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
        logger = TaskSampleLogger([0], every_n_steps=1)
        logger.samples = [
            {
                (Role.SOURCE, Modality.TEXT): TextItem(
                    views={TextView.TEXT: "source"},
                    meta={TextMeta.LANG: Lang.ZH},
                ),
                (Role.TARGET, Modality.TEXT): TextItem(
                    views={TextView.TEXT: "target"},
                    meta={TextMeta.LANG: Lang.EN},
                ),
                (Role.TARGET, Modality.AUDIO): AudioItem(
                    views={AudioView.LONGCAT: torch.zeros(1, 2, dtype=torch.long)},
                    meta={AudioMeta.DURATION: 0.02},
                ),
            }
        ]
        trainer.is_global_zero = True

        logger.on_train_batch_start(trainer, module, None, 0)

        module.generate.assert_called_once()
        experiment.add_audio.assert_called_once()
        audio_call = experiment.add_audio.call_args
        self.assertEqual(audio_call.args[0], "task_sample/0/generated")
        self.assertTrue(torch.equal(audio_call.args[1], result["audio"]["waveform"]))
        self.assertEqual(audio_call.args[2], 0)
        self.assertEqual(audio_call.kwargs, {"sample_rate": 16_000})
        experiment.add_text.assert_called_once()
        metadata_call = experiment.add_text.call_args
        self.assertEqual(metadata_call.args[0], "task_sample/0/metadata")
        self.assertIn('"task": "tts"', metadata_call.args[1])
        self.assertIn('"dataset_index": 0', metadata_call.args[1])
        self.assertIn('"duration_seconds": 0.0005', metadata_call.args[1])
        self.assertIn('"status": "ok"', metadata_call.args[1])
        self.assertIn('"waveform_finite": true', metadata_call.args[1])

    def test_task_sample_logger_loads_samples_from_real_datamodule(self):
        samples = [Mock(), Mock()]
        config = DataConfig(
            codec="longcat",
            dataloader={"batch_size": 1, "num_workers": 0},
        )
        datamodule = DataModule(
            config,
            SimpleNamespace(codec_name="longcat"),
            {Task.TTS: 1.0},
        )
        with patch("speech_to_speech.datamodule.module.load_dataset", return_value=samples):
            datamodule.setup()
        trainer = SimpleNamespace(is_global_zero=True, datamodule=datamodule)
        logger = TaskSampleLogger([1, 0], every_n_steps=1)

        logger.on_fit_start(trainer, SimpleNamespace())

        self.assertEqual(logger.samples, [samples[1], samples[0]])

    def test_task_sample_logger_logs_generation_failure(self):
        batch = ModelBatch(
            input_ids=torch.tensor([[1, 2]]),
            token_labels=torch.tensor([[-100, 2]]),
            acoustic_target=None,
            tasks=[Task.T2TT],
            pad_token_id=0,
        )
        module = SimpleNamespace(generate=Mock(side_effect=RuntimeError("boom")))
        experiment = SimpleNamespace(add_text=Mock())
        trainer = SimpleNamespace(
            global_step=0,
            is_global_zero=True,
            logger=SimpleNamespace(experiment=experiment),
            datamodule=SimpleNamespace(collator=Mock(return_value=batch)),
        )
        logger = TaskSampleLogger([0], every_n_steps=1)
        logger.samples = [_raw_sample()]

        with self.assertRaisesRegex(RuntimeError, "boom"):
            logger.on_train_batch_start(trainer, module, None, 0)

        experiment.add_text.assert_called_once()
        metadata = experiment.add_text.call_args.args[1]
        self.assertIn('"status": "failed"', metadata)
        self.assertIn('"type": "RuntimeError"', metadata)

    def test_task_sample_logger_logs_row_count_mismatch(self):
        batch = ModelBatch(
            input_ids=torch.tensor([[1, 2]]),
            token_labels=torch.tensor([[-100, 2]]),
            acoustic_target=None,
            tasks=[Task.T2TT],
            pad_token_id=0,
        )
        module = SimpleNamespace(generate=Mock(return_value=[]))
        experiment = SimpleNamespace(add_text=Mock())
        trainer = SimpleNamespace(
            global_step=0,
            is_global_zero=True,
            logger=SimpleNamespace(experiment=experiment),
            datamodule=SimpleNamespace(collator=Mock(return_value=batch)),
        )
        logger = TaskSampleLogger([0], every_n_steps=1)
        logger.samples = [_raw_sample()]

        with self.assertRaisesRegex(RuntimeError, "wrong row count"):
            logger.on_train_batch_start(trainer, module, None, 0)

        metadata = experiment.add_text.call_args.args[1]
        self.assertIn('"status": "failed"', metadata)
        self.assertIn("wrong row count", metadata)

    def test_task_sample_logger_skips_nonzero_ranks(self):
        module = SimpleNamespace(generate=Mock())
        trainer = SimpleNamespace(global_step=0, is_global_zero=False)
        logger = TaskSampleLogger([0], every_n_steps=1)

        logger.on_train_batch_start(trainer, module, None, 0)

        module.generate.assert_not_called()


def _model_config() -> ModelConfig:
    return ModelConfig(
        semantic_audio_adapter=None,
        semantic_audio_output_adapter=None,
        toy=ToyConfig(
            hidden_size=8,
            intermediate_size=16,
            layers=1,
            heads=2,
            max_position_embeddings=32,
        ),
    )


def _request() -> Request:
    return Request(
        prompt_ids=torch.tensor([4, 6]),
        task=Task.S2ST,
    )


def _raw_sample():
    def audio(offset: int) -> AudioItem:
        return AudioItem(
            views={
                AudioView.LONGCAT: torch.tensor(
                    [[offset, offset + 1], [offset + 2, offset + 3]]
                )
            },
            meta={AudioMeta.DURATION: 0.04},
        )

    return {
        (Role.SOURCE, Modality.AUDIO): audio(0),
        (Role.SOURCE, Modality.TEXT): TextItem(
            views={TextView.TEXT: "source text"},
            meta={TextMeta.LANG: Lang.ZH},
        ),
        (Role.TARGET, Modality.AUDIO): audio(4),
        (Role.TARGET, Modality.TEXT): TextItem(
            views={TextView.TEXT: "target text"},
            meta={TextMeta.LANG: Lang.EN},
        ),
    }


if __name__ == "__main__":
    unittest.main()
