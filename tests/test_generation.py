from __future__ import annotations

from collections.abc import Iterable, Sequence
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import torch
from anydataset.types import Modality
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from anytrain.module.idspace import Layout
from torch import Tensor, nn
from speech_to_speech.datamodule.batch import ModelBatch
from speech_to_speech.model import (
    AdapterType,
    AudioOutputAdapter,
    AudioInputAdapterConfig,
    AudioInputAdapterType,
    AudioOutputAdapterConfig,
    AudioOutputAdapterType,
    ToyConfig,
)
from speech_to_speech.model.acoustic.flow import FlowModel
from speech_to_speech.model.base import Config as ModelConfig
from speech_to_speech.model.base import Model
from speech_to_speech.model.generation import (
    GenerationStepResult,
    _sampling_logits,
    _stop_logit_index,
)
from speech_to_speech.generation import (
    Request,
    Result,
    decode_generated_audio,
    decode_generated_codes,
    decode_generated_frame_codes,
    decode_generated_semantic,
    generate_responses,
)
from speech_to_speech.generation.service import requests_from_batch
from speech_to_speech.generation.evaluation import evaluate_autoregressive
from speech_to_speech.runtime.audio_tokenizer import (
    FlattenedAudioTokenizer,
    NativeAudioTokenizer,
)
from speech_to_speech.runtime import (
    AudioSequenceLayout,
    Config as RuntimeConfig,
)
from speech_to_speech.runtime import Runtime
from speech_to_speech.runtime.codec_contract import supports_acoustic
from speech_to_speech.task import Task


def _has_gradient(parameters: Iterable[nn.Parameter]) -> bool:
    return any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    )


class _Codec:
    acoustic_feature_dim = 2
    acoustic_codebook_sizes = (8,)
    acoustic_layout = AcousticLayout.FRAME_ALIGNED
    acoustic_unit_length = None
    semantic_codebook = torch.randn(2, 2)
    semantic_codebook_sizes = (2,)
    sample_rate = 16_000
    frame_rate = 50.0

    def __init__(self) -> None:
        self.decode_calls = 0

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor:
        self.decode_calls += 1
        return semantic_codes[..., 0].to(acoustic_features) + acoustic_features[..., 0]

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        values = acoustic_codes[..., :1].float()
        return values.expand(*values.shape[:-1], self.acoustic_feature_dim)

    def decode(self, codes: Tensor) -> Tensor:
        self.decode_calls += 1
        return codes[..., 0].float()

    def tokenize(self, audio: Tensor, sample_rate: int) -> SemanticAcousticCodes:
        del sample_rate
        shape = (audio.size(0), 1, 1)
        semantic = torch.zeros(shape, dtype=torch.long, device=audio.device)
        acoustic = torch.zeros(shape, dtype=torch.long, device=audio.device)
        return SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)

    def detokenize(self, codes: SemanticAcousticCodes) -> Tensor:
        return self.decode_features(
            codes.semantic,
            self.acoustic_codes_to_features(codes.acoustic),
        )


class _UnifiedCodec:
    sample_rate = 16_000
    frame_rate = 50.0
    codebook_sizes = (2,)

    def __init__(self) -> None:
        self.decode_calls = 0
        self.decode_call_args: list[
            tuple[
                Tensor,
                Tensor | None,
                Tensor | None,
                Tensor | None,
                torch.Generator | None,
            ]
        ] = []

    def decode(
        self,
        codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        self.decode_calls += 1
        self.decode_call_args.append(
            (
                codes.clone(),
                None if mask is None else mask.clone(),
                None if reference_features is None else reference_features.clone(),
                None if reference_mask is None else reference_mask.clone(),
                generator,
            )
        )
        return codes[..., 0].float()

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        del audio, sample_rate
        raise NotImplementedError


class _RowFailingSemanticCodec(_UnifiedCodec):
    def __init__(self) -> None:
        super().__init__()
        self.decode_batch_sizes: list[int] = []

    def decode(
        self,
        codes: Tensor,
        *,
        mask: Tensor | None = None,
        reference_features: Tensor | None = None,
        reference_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        del mask, reference_features, reference_mask, generator
        self.decode_calls += 1
        self.decode_batch_sizes.append(codes.size(0))
        if bool(codes[..., 0].eq(1).any()):
            raise ValueError("invalid semantic row")
        return codes[..., 0].float()


class _RowFailingAcousticCodec(_Codec):
    def __init__(self) -> None:
        super().__init__()
        self.decode_batch_sizes: list[int] = []

    def decode_features(
        self,
        semantic_codes: Tensor,
        acoustic_features: Tensor,
    ) -> Tensor:
        self.decode_calls += 1
        self.decode_batch_sizes.append(semantic_codes.size(0))
        if bool(semantic_codes[..., 0].eq(1).any()):
            raise ValueError("invalid acoustic row")
        return semantic_codes[..., 0].to(acoustic_features) + acoustic_features[..., 0]


class _Runtime:
    gradient_checkpointing = False

    def __init__(self) -> None:
        self.audio_sequence_layout = AudioSequenceLayout.SEMANTIC
        self.layout = Layout(text=(0, 4), audio=(4, 9))
        self.audio_tokenizer = NativeAudioTokenizer(vocab_size=2)
        self.codec = _Codec()
        self.eos_token_id = 3
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.boa_token_id = 6
        self.eoa_token_id = 7
        self.mask_token_id = 8
        self.structured_full_sequence = False

    @property
    def semantic_codec(self):
        try:
            return self._semantic_codec
        except AttributeError as exc:
            raise RuntimeError("test runtime requires an explicit semantic codec") from exc

    @property
    def codec_audio_range(self) -> tuple[int, int]:
        return 4, 6

    @property
    def acoustic_side_channel(self) -> bool:
        return supports_acoustic(self.codec)

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


class _TinyCodec(_Codec):
    acoustic_feature_dim = 8
    acoustic_codebook_sizes = (8,)
    semantic_codebook = torch.randn(2, 8)

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        values = acoustic_codes[..., :1].to(dtype=torch.float32)
        return values.expand(*values.shape[:-1], self.acoustic_feature_dim)

    def decode_features(
        self,
        semantic_codes: Tensor,
        acoustic_features: Tensor,
    ) -> Tensor:
        return semantic_codes[..., 0].to(acoustic_features) + acoustic_features[..., 0]

    def decode(self, codes: Tensor) -> Tensor:
        return codes[..., 0].float()


class _TinyRuntime(_Runtime):
    def __init__(self) -> None:
        super().__init__()
        self.layout = Layout(text=(0, 8), audio=(8, 13))
        self.audio_tokenizer = NativeAudioTokenizer(vocab_size=2)
        self.codec = _TinyCodec()
        self.eos_token_id = 3
        self.boa_token_id = 10
        self.eoa_token_id = 11
        self.mask_token_id = 12

    @property
    def codec_audio_range(self) -> tuple[int, int]:
        return 8, 10

    @property
    def audio_generation_allowed_ids(self) -> tuple[int, ...]:
        return 8, 9, 11


class _UnifiedRuntime(_Runtime):
    def __init__(self) -> None:
        super().__init__()
        self.codec = _UnifiedCodec()
        self.audio_sequence_layout = AudioSequenceLayout.FLATTENED
        self.audio_tokenizer = FlattenedAudioTokenizer(
            codebook_sizes=(2,),
            codec_name="unicodec",
        )
        self.layout = Layout(
            text=(0, 4),
            audio=(4, 4 + self.audio_tokenizer.vocab_size + 3),
        )
        self.boa_token_id = self.codec_audio_range[1]
        self.eoa_token_id = self.boa_token_id + 1
        self.mask_token_id = self.boa_token_id + 2

    @property
    def semantic_codec(self):
        raise RuntimeError("full codec sequence does not expose semantic-only decode")

    @property
    def acoustic_side_channel(self) -> bool:
        return False

    @property
    def codec_audio_range(self) -> tuple[int, int]:
        start, _ = self.layout.blocks[Modality.AUDIO.value]
        return start, start + self.audio_tokenizer.vocab_size

    @property
    def audio_generation_allowed_ids(self) -> tuple[int, ...]:
        start, end = self.codec_audio_range
        return (*range(start, end), self.eoa_token_id)


class _GenerationModel(FlowModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.runtime = _Runtime()
        self.layout = self.runtime.layout
        self.audio_token_frame_spans = torch.tensor([1, 1])
        self.backbone = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty(0))
        )
        self.acoustic_condition = nn.Identity()
        self.calls: list[tuple[int, int]] = []
        self.condition: Tensor | None = None
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
    ) -> GenerationStepResult:
        del kwargs
        cached_length = 0 if past_key_values is None else past_key_values.length
        source = 0 if past_key_values is None else past_key_values.source
        length = cached_length + input_ids.size(1)
        self.calls.append((input_ids.size(1), input_ids.size(0)))

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
        return GenerationStepResult(
            logits=logits,
            past_key_values=cache,
            audio_head_past=None,
            hidden_states=(hidden,) if output_hidden_states else None,
        )

    def select_audio_head_cache(self, past_key_values, indices):
        del indices
        return past_key_values

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


class _TokenGenerationModel(Model):
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


class _UnifiedGenerationModel(Model):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.runtime = _UnifiedRuntime()
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
        encoded = self.runtime.audio_tokenizer.encode(torch.tensor([[0], [1]]))
        self._tokens = [
            start + token_id
            for token_id in encoded.tolist()
        ]
        self._tokens.append(self.runtime.eoa_token_id)
        self._step = 0
        self.calls: list[tuple[int, int]] = []
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
    ) -> GenerationStepResult:
        del kwargs, output_hidden_states, past_key_values
        self.calls.append((input_ids.size(1), input_ids.size(0)))
        next_id = self._tokens[min(self._step, len(self._tokens) - 1)]
        self._step += 1
        if token_ids is not None:
            matches = (token_ids == next_id).nonzero()
            if matches.numel() == 0:
                raise AssertionError("generated id is outside allowed token_ids")
            logits = torch.full(
                (input_ids.size(0), input_ids.size(1), token_ids.numel()),
                float("-inf"),
            )
            logits[:, -1, int(matches[0, 0])] = 0
        elif modality is not None:
            start, end = self.layout.blocks[modality.value]
            logits = torch.full((input_ids.size(0), input_ids.size(1), end - start), float("-inf"))
            logits[:, -1, next_id - start] = 0
        else:
            logits = torch.full(
                (input_ids.size(0), input_ids.size(1), self.layout.vocab_size),
                float("-inf"),
            )
            logits[:, -1, next_id] = 0
        cache = (
            SimpleNamespace(
                length=self._step,
                source=0,
                batch_select_indices=lambda indices: None,
            )
            if use_cache
            else None
        )
        return GenerationStepResult(
            logits=logits,
            past_key_values=cache,
            audio_head_past=None,
        )


class _FullSequenceCodec:
    sample_rate = 16_000
    frame_rate = 50.0

    def __init__(self, codebook_sizes: tuple[int, ...] = (4, 10)) -> None:
        self.codebook_sizes = codebook_sizes
        self.decode_calls = 0
        self.decoded_codes: Tensor | None = None

    def decode(self, codes: Tensor) -> Tensor:
        self.decode_calls += 1
        self.decoded_codes = codes.clone()
        return codes.sum(dim=-1).float()

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        del audio, sample_rate
        raise NotImplementedError


class _FullSequenceRuntime:
    def __init__(self, codebook_sizes: tuple[int, ...] = (4, 10)) -> None:
        self.audio_sequence_layout = AudioSequenceLayout.FLATTENED
        self.audio_tokenizer = FlattenedAudioTokenizer(
            codebook_sizes=codebook_sizes,
            codec_name="frame-codec",
        )
        self.codec = _FullSequenceCodec(codebook_sizes)
        self.layout = Layout(text=(0, 4), audio=(4, 4 + self.audio_tokenizer.vocab_size + 3))
        self.pad_token_id = 0
        self.eos_token_id = 3
        self.bos_token_id = 1
        self.boa_token_id = self.codec_audio_range[1]
        self.eoa_token_id = self.boa_token_id + 1
        self.mask_token_id = self.boa_token_id + 2
        self.structured_full_sequence = False

    @property
    def semantic_codec(self):
        raise RuntimeError("full codec sequence does not expose semantic-only decode")

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


class _FullSequenceGenerationModel(Model):
    def __init__(
        self,
        codes: Tensor | None = None,
        *,
        codebook_sizes: tuple[int, ...] = (4, 10),
    ) -> None:
        nn.Module.__init__(self)
        if codes is None:
            codes = torch.tensor([[1, 5], [2, 6]])
        if codes.size(1) != len(codebook_sizes):
            raise ValueError("test codes must match configured codebooks.")
        self.runtime = _FullSequenceRuntime(codebook_sizes)
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
        encoded = self.runtime.audio_tokenizer.encode(codes)
        self._tokens = [
            start + token_id
            for token_id in encoded.tolist()
        ]
        self._tokens.append(self.runtime.eoa_token_id)
        self._step = 0
        self.sample_calls = 0
        self.generation_inputs: list[Tensor] = []
        self.allowed_token_ids: list[Tensor | None] = []

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
    ) -> GenerationStepResult:
        del kwargs, output_hidden_states, past_key_values
        self.generation_inputs.append(input_ids.clone())
        self.allowed_token_ids.append(None if token_ids is None else token_ids.clone())
        next_id = self._tokens[min(self._step, len(self._tokens) - 1)]
        self._step += 1
        if token_ids is not None:
            matches = (token_ids == next_id).nonzero()
            if matches.numel() == 0:
                raise AssertionError("generated id is outside allowed token_ids")
            logits = torch.full(
                (input_ids.size(0), input_ids.size(1), token_ids.numel()),
                -100.0,
            )
            logits[:, -1, int(matches[0, 0])] = 0
        elif modality is Modality.AUDIO:
            start, end = self.layout.blocks[Modality.AUDIO.value]
            logits = torch.full((input_ids.size(0), input_ids.size(1), end - start), float("-inf"))
            logits[:, -1, next_id - start] = 0
        else:
            logits = torch.full(
                (input_ids.size(0), input_ids.size(1), self.layout.vocab_size),
                float("-inf"),
            )
            logits[:, -1, next_id] = 0
        cache = (
            SimpleNamespace(batch_select_indices=lambda indices: None)
            if use_cache
            else None
        )
        return GenerationStepResult(
            logits=logits,
            past_key_values=cache,
            audio_head_past=None,
        )

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

    def generation_step(self, input_ids: Tensor, **kwargs) -> GenerationStepResult:
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
        return GenerationStepResult(logits=logits, past_key_values=cache, audio_head_past=None)

    def select_audio_head_cache(self, past_key_values, indices):
        del indices
        return past_key_values


class _LogprobGenerationModel(_UnifiedGenerationModel):
    def __init__(self) -> None:
        super().__init__()
        self.step_logits = torch.tensor([0.0, 1.0, 3.0])

    def generation_step(self, input_ids: Tensor, **kwargs) -> GenerationStepResult:
        token_ids = kwargs.get("token_ids")
        if token_ids is None:
            raise AssertionError("logprob test requires explicit allowed token ids")
        logits = self.step_logits.to(device=input_ids.device).view(1, 1, -1)
        logits = logits.expand(input_ids.size(0), 1, token_ids.numel()).clone()
        cache = (
            SimpleNamespace(batch_select_indices=lambda indices: None)
            if kwargs["use_cache"]
            else None
        )
        return GenerationStepResult(logits=logits, past_key_values=cache, audio_head_past=None)


class GenerationTest(unittest.TestCase):
    def test_autoregressive_evaluation_reports_generation_health(self):
        module = Mock()
        module.parameters.return_value = iter(
            [SimpleNamespace(device=torch.device("cpu"))]
        )
        module.generate.return_value = [
            Result(
                response_ids=torch.tensor([7, 8]),
                audio={
                    "features": torch.ones(1, 2, 3),
                    "codes": None,
                    "waveform": torch.ones(4),
                    "sample_rate": 4,
                },
            )
        ]
        requests = [Mock()]
        with (
            patch(
                "speech_to_speech.generation.evaluation.requests_from_batch",
                return_value=requests,
            ),
            patch(
                "speech_to_speech.generation.evaluation.time.perf_counter",
                side_effect=(1.0, 1.5),
            ),
        ):
            report = evaluate_autoregressive(module, Mock(), sample_rate=4)

        self.assertEqual(report["token_ids"], [7, 8])
        self.assertEqual(report["feature_shape"], [1, 2, 3])
        self.assertEqual(report["waveform_shape"], [4])
        self.assertEqual(report["duration_seconds"], 1.0)
        self.assertEqual(report["elapsed_seconds"], 0.5)
        self.assertEqual(report["rtf"], 0.5)
        self.assertTrue(report["finite"])

    def test_explicit_audio_head_is_shared_by_logits_paths(self):
        model = Model(
            ModelConfig(
                semantic_audio_adapter=None,
                audio_output_adapter=AudioOutputAdapterConfig(
                    type=AudioOutputAdapterType.MLP,
                ),
                toy=ToyConfig(
                    hidden_size=8,
                    intermediate_size=16,
                    layers=1,
                    heads=2,
                    max_position_embeddings=32,
                ),
            ),
            runtime=_TinyRuntime(),
        ).eval()

        self.assertIsInstance(model.tokens.audio_head, AudioOutputAdapter)
        hidden = torch.randn(1, 2, 8)
        logits = model.semantic_audio_logits(hidden)

        self.assertEqual(logits.shape[:2], (1, 2))
        self.assertTrue(torch.isfinite(logits).all())

    def test_runtime_gradient_checkpointing_enables_external_adapters(self):
        runtime = _TinyRuntime()
        runtime.gradient_checkpointing = True
        model = Model(
            ModelConfig(
                semantic_audio_adapter=None,
                audio_output_adapter=AudioOutputAdapterConfig(
                    type=AudioOutputAdapterType.MLP,
                ),
                audio_input_adapter=AudioInputAdapterConfig(
                    type=AudioInputAdapterType.MLP,
                ),
                toy=ToyConfig(
                    hidden_size=8,
                    intermediate_size=16,
                    layers=1,
                    heads=2,
                    max_position_embeddings=32,
                ),
            ),
            runtime=runtime,
        )

        audio_projection = model.tokens.audio_projection
        if model.source_audio_encoder is None:
            self.fail("source audio encoder was not constructed")
        self.assertTrue(audio_projection.gradient_checkpointing)
        self.assertTrue(model.source_audio_encoder.gradient_checkpointing)
        self.assertTrue(model.tokens.audio_head.gradient_checkpointing)
        self.assertTrue(model.ctc_decoders.source.gradient_checkpointing)
        self.assertTrue(model.ctc_decoders.target.gradient_checkpointing)

    def test_runtime_gradient_checkpointing_keeps_external_adapter_gradients(self):
        runtime = _TinyRuntime()
        runtime.gradient_checkpointing = True
        model = Model(
            ModelConfig(
                semantic_audio_adapter=AdapterType.MLP,
                audio_output_adapter=AudioOutputAdapterConfig(
                    type=AudioOutputAdapterType.MLP,
                ),
                audio_input_adapter=AudioInputAdapterConfig(
                    type=AudioInputAdapterType.MLP,
                ),
                toy=ToyConfig(
                    hidden_size=8,
                    intermediate_size=16,
                    layers=1,
                    heads=2,
                    max_position_embeddings=32,
                ),
            ),
            runtime=runtime,
        ).train()

        input_ids = torch.tensor([[1, 8, 9, 11]])
        output = model(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            audio_input_positions=torch.tensor([[1, 2]], dtype=torch.long),
        )
        loss = output.logits.float().sum()
        loss.backward()

        audio_projection = model.tokens.audio_projection
        if model.source_audio_encoder is None:
            self.fail("source audio encoder was not constructed")
        self.assertTrue(_has_gradient(audio_projection.parameters()))
        self.assertTrue(_has_gradient(model.source_audio_encoder.parameters()))
        self.assertTrue(_has_gradient(model.tokens.audio_head.parameters()))

    def test_source_audio_encoder_overlays_only_source_positions(self):
        model = Model(
            ModelConfig(
                semantic_audio_adapter=None,
                audio_output_adapter=AudioOutputAdapterConfig(
                    type=AudioOutputAdapterType.NONE,
                ),
                audio_input_adapter=AudioInputAdapterConfig(
                    type=AudioInputAdapterType.MLP,
                ),
                toy=ToyConfig(
                    hidden_size=8,
                    intermediate_size=16,
                    layers=1,
                    heads=2,
                    max_position_embeddings=32,
                ),
            ),
            runtime=_TinyRuntime(),
        ).eval()
        input_ids = torch.tensor([[1, 8, 9, 10, 11, 2]])
        positions = torch.tensor([[1, 2]], dtype=torch.long)
        base = model._input_embedding(input_ids)
        adapter = model.source_audio_encoder
        if adapter is None:
            self.fail("source audio encoder was not constructed")
        audio_start, _ = model.layout.blocks[Modality.AUDIO.value]
        source_ids = input_ids.gather(1, positions) - audio_start
        source_features = model.tokens.audio_rows(source_ids)
        source_values = adapter(
            source_features,
            mask=torch.ones_like(positions, dtype=torch.bool),
        )
        expected = base.clone()
        expected[0, positions[0]] = source_values[0].to(dtype=expected.dtype)

        ordinary_positions = torch.tensor([3, 4])
        ordinary_ids = input_ids[0, ordinary_positions] - audio_start
        ordinary_rows = model.tokens.audio_rows(ordinary_ids)
        projection = model.tokens.audio_projection
        with patch.object(
            projection,
            "forward",
            wraps=projection.forward,
        ) as project:
            adapted = model._input_embedding(input_ids, positions)

        self.assertEqual(project.call_count, 1)
        torch.testing.assert_close(project.call_args.args[0], ordinary_rows)
        torch.testing.assert_close(adapted, expected)

        with self.assertRaisesRegex(ValueError, "codec audio payload"):
            model._input_embedding(input_ids, torch.tensor([[3]]))

    def test_audio_input_positions_reject_invalid_indices_at_model_boundary(self):
        model = Model(_model_config(), runtime=_TinyRuntime()).eval()
        input_ids = torch.tensor([[1, 8, 9]])

        cases = (
            ("padding below -1", torch.tensor([[-2]]), "use -1 padding"),
            ("sequence upper bound", torch.tensor([[3]]), "valid sequence positions"),
            ("duplicate valid positions", torch.tensor([[1, 1]]), "must not repeat"),
        )
        for name, positions, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    model(input_ids, audio_input_positions=positions)

    def test_cached_generation_runs_source_audio_encoder_once(self):
        model = Model(
            ModelConfig(
                semantic_audio_adapter=None,
                audio_output_adapter=AudioOutputAdapterConfig(
                    type=AudioOutputAdapterType.NONE,
                ),
                audio_input_adapter=AudioInputAdapterConfig(
                    type=AudioInputAdapterType.MLP,
                ),
                toy=ToyConfig(
                    hidden_size=8,
                    intermediate_size=16,
                    layers=1,
                    heads=2,
                    max_position_embeddings=32,
                ),
            ),
            runtime=_TinyRuntime(),
        ).eval()
        adapter = model.source_audio_encoder
        if adapter is None:
            self.fail("source audio encoder was not constructed")
        with patch.object(adapter, "forward", wraps=adapter.forward) as forward:
            model.generate_tokens(
                torch.tensor([[1, 8, 9, 11]]),
                audio_input_positions=torch.tensor([[1, 2]]),
                max_new_tokens=2,
                generation_modality=Modality.TEXT,
                do_sample=False,
                use_cache=True,
            )
        self.assertEqual(forward.call_count, 1)

    def test_generation_validates_input_routing_only_for_the_prompt(self):
        prompt_ids = torch.tensor([[1, 8]])
        for use_cache in (False, True):
            with self.subTest(use_cache=use_cache):
                model = Model(_model_config(), runtime=_TinyRuntime()).eval()
                selected_modalities = model.tokens.selected_modalities
                selected_logits = model.tokens.selected_logits
                with (
                    patch.object(
                        model.tokens,
                        "selected_modalities",
                        wraps=selected_modalities,
                    ) as selected,
                    patch.object(
                        model.tokens,
                        "selected_logits",
                        wraps=selected_logits,
                    ) as logits,
                ):
                    generated = model.generate_tokens(
                        prompt_ids,
                        max_new_tokens=3,
                        allowed_token_ids=(2,),
                        do_sample=False,
                        use_cache=use_cache,
                    )

                self.assertEqual(tuple(generated.shape), (1, 5))
                self.assertEqual(selected.call_count, 1)
                torch.testing.assert_close(selected.call_args.args[0], prompt_ids)
                self.assertEqual(
                    [call.kwargs["validate"] for call in logits.call_args_list],
                    [True, False, False],
                )

    def test_mixed_generation_reuses_preclassified_routing(self):
        prompt_ids = torch.tensor([[1, 2]])
        scripted_ids = (2, 10, 8, 11, 3)
        for use_cache in (False, True):
            with self.subTest(use_cache=use_cache):
                runtime = _TinyRuntime()
                runtime._semantic_codec = _UnifiedCodec()
                model = Model(_model_config(), runtime=runtime).eval()
                script = iter(scripted_ids)
                token_kinds: list[str | None] = []

                def selected_logits(
                    text_embedding: nn.Embedding,
                    hidden_state: Tensor,
                    token_ids: Tensor,
                    **kwargs,
                ) -> tuple[Tensor, None]:
                    del text_embedding
                    token_kinds.append(kwargs.get("token_kind"))
                    next_id = next(script)
                    index = (token_ids == next_id).nonzero(as_tuple=False)
                    if index.numel() == 0:
                        raise AssertionError("scripted token is outside the mixed token set")
                    logits = hidden_state.new_full(
                        (*hidden_state.shape[:-1], token_ids.numel()),
                        float("-inf"),
                    )
                    logits[..., int(index[0, 0])] = 0
                    return logits, None

                selected_modalities = model.tokens.selected_modalities
                with (
                    patch.object(
                        model.tokens,
                        "selected_modalities",
                        wraps=selected_modalities,
                    ) as selected,
                    patch.object(
                        model.tokens,
                        "selected_logits",
                        side_effect=selected_logits,
                    ),
                ):
                    result = generate_responses(
                        [_mixed_request(Task.INTERLEAVED_AR)],
                        model,
                        max_new_tokens=len(scripted_ids),
                        do_sample=False,
                        use_cache=use_cache,
                    )[0]

                self.assertTrue(
                    torch.equal(result["response_ids"], torch.tensor(scripted_ids))
                )
                self.assertEqual(selected.call_count, 1)
                torch.testing.assert_close(selected.call_args.args[0], prompt_ids)
                self.assertEqual(token_kinds, ["mixed"] * len(scripted_ids))

    def test_frame_span_buffer_follows_the_backbone_device(self):
        runtime = _TinyRuntime()
        model = Model(
            _model_config(),
            runtime=runtime,
        ).to(device="meta")

        self.assertEqual(model.audio_token_frame_spans.device.type, "meta")
        self.assertNotIn("audio_token_frame_spans", model.state_dict())

    @patch("speech_to_speech._compat.nn.Buffer", new=None)
    def test_frame_span_buffer_supports_torch_without_nn_buffer(self):
        model = Model(
            _model_config(),
            runtime=_TinyRuntime(),
        )

        self.assertIs(
            model.audio_token_frame_spans,
            model._buffers["audio_token_frame_spans"],
        )
        self.assertNotIn("audio_token_frame_spans", model.state_dict())

    def test_text_generation_excludes_padding_and_bos(self):
        rt = Runtime(RuntimeConfig())
        rt.__dict__["layout"] = Layout(text=(0, 4), audio=(4, 8))
        rt.__dict__["pad_token_id"] = 0
        rt.__dict__["bos_token_id"] = 1

        allowed = rt.generation_allowed_ids(Modality.TEXT)

        self.assertEqual(allowed, (2, 3))

    def test_modality_generation_masks_special_tokens(self):
        model = Model(
            _model_config(),
            runtime=_TinyRuntime(),
        ).eval()

        def text_logits(
            text_embedding: nn.Embedding,
            hidden_state: Tensor,
            local_ids=None,
        ) -> Tensor:
            del text_embedding
            self.assertIsNone(local_ids)
            logits = hidden_state.new_zeros(*hidden_state.shape[:-1], 8)
            logits[..., 0] = 100
            logits[..., 1] = 90
            logits[..., 2] = 80
            return logits

        def audio_logits(hidden_state: Tensor, local_ids=None) -> Tensor:
            self.assertIsNone(local_ids)
            logits = hidden_state.new_zeros(*hidden_state.shape[:-1], 5)
            logits[..., 2] = 100
            logits[..., 0] = 90
            return logits

        with patch.object(model.tokens, "text_logits", side_effect=text_logits):
            text = model.generate_tokens(
                torch.tensor([[2, 3]]),
                max_new_tokens=1,
                generation_modality=Modality.TEXT,
                do_sample=False,
                use_cache=False,
            )
        with patch.object(model.tokens, "semantic_audio_logits", side_effect=audio_logits):
            audio = model.generate_tokens(
                torch.tensor([[2, 3]]),
                max_new_tokens=1,
                generation_modality=Modality.AUDIO,
                do_sample=False,
                use_cache=False,
            )

        self.assertEqual(int(text[0, -1]), 2)
        self.assertEqual(int(audio[0, -1]), 8)

    def test_generation_only_computes_the_allowed_output_head(self):
        model = Model(
            _model_config(),
            runtime=_TinyRuntime(),
        ).eval()

        with (
            patch.object(
                model.tokens,
                "text_logits",
                side_effect=AssertionError("text head should not run"),
            ),
            patch.object(
                model.tokens,
                "semantic_audio_logits",
                wraps=model.tokens.semantic_audio_logits,
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

    def test_transformer_audio_generation_step_encodes_the_full_prompt(self):
        torch.manual_seed(0)
        model = Model(
            _transformer_model_config(),
            runtime=_TinyRuntime(),
        ).eval()
        input_ids = torch.tensor([[1, 2, 8, 9]])
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        hidden_states = model.token_hidden_states(
            input_ids,
            attention_mask=attention_mask,
        )
        expected, _ = model.modality_logits(
            hidden_states,
            Modality.AUDIO,
            attention_mask=attention_mask,
        )

        output = model.generation_step(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            token_ids=None,
            modality=Modality.AUDIO,
            past_key_values=None,
            use_cache=True,
        )

        torch.testing.assert_close(output.logits, expected[:, -1:])
        self.assertIsNotNone(output.audio_head_past)

    def test_transformer_audio_generation_cache_matches_full_recompute(self):
        torch.manual_seed(0)
        model = Model(
            _transformer_model_config(),
            runtime=_TinyRuntime(),
        ).eval()
        generated = torch.tensor([[0, 0, 1, 2], [0, 1, 2, 8]])
        attention_mask = torch.tensor(
            [[False, False, True, True], [False, True, True, True]]
        )
        cached_input = generated
        backbone_past = None
        audio_head_past = None

        for next_ids in (torch.tensor([8, 9]), torch.tensor([9, 8])):
            cached = model.generation_step(
                cached_input,
                attention_mask=attention_mask,
                output_hidden_states=False,
                token_ids=None,
                modality=Modality.AUDIO,
                past_key_values=backbone_past,
                use_cache=True,
                audio_head_past=audio_head_past,
            )
            recomputed = model.generation_step(
                generated,
                attention_mask=attention_mask,
                output_hidden_states=False,
                token_ids=None,
                modality=Modality.AUDIO,
                past_key_values=None,
                use_cache=False,
            )

            torch.testing.assert_close(
                cached.logits,
                recomputed.logits,
                atol=1e-5,
                rtol=1e-5,
            )
            backbone_past = cached.past_key_values
            audio_head_past = cached.audio_head_past
            cached_input = next_ids[:, None]
            generated = torch.cat((generated, cached_input), dim=1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        attention_mask.size(0),
                        1,
                        dtype=torch.bool,
                        device=attention_mask.device,
                    ),
                ),
                dim=1,
            )

    def test_generate_tokens_with_logprobs_matches_public_generation(self):
        prompt_ids = torch.tensor([[1, 2]])
        allowed_token_ids = torch.tensor([2, 3, 4])
        kwargs = {
            "max_new_tokens": 2,
            "allowed_token_ids": allowed_token_ids,
            "do_sample": False,
            "use_cache": False,
        }

        output = _LogprobGenerationModel().generate_tokens_with_logprobs(
            prompt_ids,
            **kwargs,
        )
        generated = _LogprobGenerationModel().generate_tokens(prompt_ids, **kwargs)

        self.assertTrue(torch.equal(output.sequences, generated))
        self.assertTrue(torch.equal(output.sequences, torch.tensor([[1, 2, 4, 4]])))
        self.assertEqual(tuple(output.token_logprobs.shape), (1, 2))
        self.assertEqual(tuple(output.token_logprob_mask.shape), (1, 2))
        expected = torch.log_softmax(
            _LogprobGenerationModel().step_logits,
            dim=-1,
        )[2]
        torch.testing.assert_close(
            output.token_logprobs,
            expected.expand_as(output.token_logprobs),
        )
        self.assertTrue(
            torch.equal(output.token_logprob_mask, torch.ones(1, 2, dtype=torch.bool))
        )
        self.assertIsNone(output.audio_condition)
        self.assertIsNone(output.frame_spans)

    def test_generate_tokens_with_logprobs_masks_finished_rows(self):
        model = _VariableStopModel()

        output = model.generate_tokens_with_logprobs(
            torch.tensor([[1, 4], [1, 5]]),
            max_new_tokens=2,
            stop_token_id=model.runtime.eos_token_id,
            allowed_token_ids=(model.runtime.eos_token_id, 1),
            do_sample=False,
            use_cache=True,
        )

        self.assertTrue(
            torch.equal(output.sequences, torch.tensor([[1, 4, 3, 3], [1, 5, 1, 3]]))
        )
        self.assertEqual(tuple(output.token_logprobs.shape), (2, 2))
        self.assertTrue(
            torch.equal(
                output.token_logprob_mask,
                torch.tensor([[True, False], [True, True]]),
            )
        )
        torch.testing.assert_close(
            output.token_logprobs,
            torch.zeros_like(output.token_logprobs),
        )
        self.assertEqual(model.batch_sizes, [2, 2])
        self.assertEqual(model.cache_selections, [])

    def test_generation_rejects_invalid_constraints(self):
        model = Model(
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

    def test_stop_logit_index_is_resolved_before_generation(self):
        layout = Layout(text=(0, 4), audio=(4, 8))

        self.assertEqual(
            _stop_logit_index(7, torch.tensor([2, 7, 4]), None, layout),
            1,
        )
        self.assertIsNone(_stop_logit_index(7, torch.tensor([2, 4]), None, layout))
        self.assertEqual(
            _stop_logit_index(7, None, Modality.AUDIO, layout),
            3,
        )
        self.assertEqual(_stop_logit_index(7, None, None, layout), 7)
        with self.assertRaisesRegex(ValueError, "no non-stop token"):
            _stop_logit_index(7, torch.tensor([7]), None, layout)

    def test_minimum_length_rejects_first_step_without_non_stop_logit(self):
        logits = torch.tensor([[[0.0, float("-inf")]]])

        with self.assertRaisesRegex(ValueError, "no non-stop token"):
            _sampling_logits(
                logits,
                1.0,
                top_p=1.0,
                step=0,
                min_new_tokens=2,
                stop_logit_index=0,
            )

    def test_token_only_generation_requires_explicit_semantic_codec(self):
        model = _TokenGenerationModel()

        with self.assertRaisesRegex(RuntimeError, "semantic codec"):
            generate_responses(
                [_request()],
                model,
                max_new_tokens=3,
                do_sample=False,
            )

        self.assertEqual(model.runtime.codec.decode_calls, 0)

    def test_token_only_generation_uses_independent_semantic_codec(self):
        model = _TokenGenerationModel()
        semantic_codec = _UnifiedCodec()
        model.runtime._semantic_codec = semantic_codec

        result = generate_responses(
            [_request()],
            model,
            max_new_tokens=3,
            do_sample=False,
        )[0]

        self.assertEqual(semantic_codec.decode_calls, 1)
        self.assertEqual(model.runtime.codec.decode_calls, 0)
        self.assertEqual(result["audio"]["sample_rate"], semantic_codec.sample_rate)

    def test_decode_generated_semantic_passes_reference_and_generator(self):
        codec = _UnifiedCodec()
        tokenizer = NativeAudioTokenizer(vocab_size=2)
        reference_features = torch.randn(1, 2, 3)
        reference_mask = torch.tensor([[True, False]])
        generator = torch.Generator().manual_seed(1)

        decoded = decode_generated_semantic(
            torch.tensor([[4, 5]]),
            codec=codec,
            audio_tokenizer=tokenizer,
            audio_token_range=(4, 6),
            semantic_reference_features=reference_features,
            semantic_reference_mask=reference_mask,
            semantic_decode_generator=generator,
        )

        self.assertEqual(codec.decode_calls, 1)
        (
            _codes,
            _mask,
            actual_features,
            actual_reference_mask,
            actual_generator,
        ) = codec.decode_call_args[0]
        self.assertIsNotNone(actual_features)
        self.assertIsNotNone(actual_reference_mask)
        if actual_features is None or actual_reference_mask is None:
            self.fail("semantic decode reference arguments were not recorded")
        torch.testing.assert_close(actual_features, reference_features)
        torch.testing.assert_close(actual_reference_mask, reference_mask)
        self.assertIs(actual_generator, generator)
        torch.testing.assert_close(decoded, torch.tensor([[0.0, 1.0]]))

    def test_generate_responses_batches_semantic_reference_options(self):
        model = _TokenGenerationModel()
        semantic_codec = _UnifiedCodec()
        model.runtime._semantic_codec = semantic_codec
        reference_features = torch.randn(2, 3)
        reference_mask = torch.tensor([True, False])
        generator = torch.Generator().manual_seed(2)
        request = _request()
        request["semantic_reference_features"] = reference_features
        request["semantic_reference_mask"] = reference_mask
        request["semantic_decode_generator"] = generator

        result = generate_responses(
            [request],
            model,
            max_new_tokens=3,
            do_sample=False,
        )[0]

        self.assertIsNotNone(result["audio"])
        self.assertEqual(semantic_codec.decode_calls, 1)
        (
            _codes,
            _mask,
            actual_features,
            actual_reference_mask,
            actual_generator,
        ) = semantic_codec.decode_call_args[0]
        self.assertIsNotNone(actual_features)
        self.assertIsNotNone(actual_reference_mask)
        if actual_features is None or actual_reference_mask is None:
            self.fail("semantic decode reference arguments were not recorded")
        torch.testing.assert_close(actual_features, reference_features.unsqueeze(0))
        torch.testing.assert_close(
            actual_reference_mask,
            reference_mask.unsqueeze(0),
        )
        self.assertIs(actual_generator, generator)

    def test_semantic_reference_decode_isolates_invalid_row(self):
        model = _TokenGenerationModel()
        codec = _RowFailingSemanticCodec()
        model.runtime._semantic_codec = codec
        requests = [_request(), _request()]
        for request in requests:
            request["semantic_reference_features"] = torch.zeros(1, 2)
        sequence = torch.tensor(
            [
                [4, 6, 4, 4, 7],
                [4, 6, 5, 5, 7],
            ]
        )

        with (
            patch.object(model, "generate_tokens", return_value=sequence),
            self.assertWarnsRegex(RuntimeWarning, "invalid semantic row"),
        ):
            results = generate_responses(
                requests,
                model,
                max_new_tokens=3,
                do_sample=False,
            )

        self.assertIsNotNone(results[0]["audio"])
        self.assertIsNone(results[1]["audio"])
        self.assertEqual(codec.decode_batch_sizes, [1, 1])

    def test_full_codec_sequence_decodes_all_codebooks(self):
        tokenizer = FlattenedAudioTokenizer(
            codebook_sizes=(4, 10),
            codec_name="longcat",
        )
        frames = torch.tensor([[1, 5], [2, 6]], dtype=torch.long)
        codec = Mock()
        codec.decode.side_effect = lambda codes: codes
        start = 20

        decoded = decode_generated_frame_codes(
            tokenizer.encode(frames)[None] + start,
            codec=codec,
            audio_tokenizer=tokenizer,
            audio_token_range=(start, start + tokenizer.vocab_size),
        )

        torch.testing.assert_close(decoded, frames[None])
        codec.decode.assert_called_once()
        torch.testing.assert_close(codec.decode.call_args.args[0], frames[None])

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
            (torch.tensor([4, 9]), "runtime layout"),
        )
        for prompt_ids, message in invalid:
            request = _request()
            request["prompt_ids"] = prompt_ids
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    generate_responses([request], _GenerationModel(), max_new_tokens=1)

    def test_acoustic_codec_without_audio_model_decodes_semantic_tokens(self):
        runtime = _TinyRuntime()
        runtime._semantic_codec = _UnifiedCodec()
        model = Model(
            _model_config(),
            runtime=runtime,
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
        model = Model(
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

    def test_tiny_qwen_cache_keeps_finished_state_on_device(self):
        def generate(use_cache: bool) -> tuple[Tensor, list[int]]:
            model = Model(
                _model_config(),
                runtime=_TinyRuntime(),
            ).eval()
            generation_step = model.generation_step
            batch_sizes: list[int] = []

            def variable_step(input_ids: Tensor, **kwargs) -> GenerationStepResult:
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
        self.assertEqual(cached_batch_sizes, [2, 2])
        self.assertEqual(full_batch_sizes, [2, 2])

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

    def test_unified_audio_generation_decodes_full_frame_codes(self):
        model = _UnifiedGenerationModel()
        result = generate_responses(
            [_request()],
            model,
            max_new_tokens=6,
            do_sample=False,
            use_cache=True,
        )[0]

        start, _ = model.runtime.codec_audio_range
        expected_response = model.runtime.audio_tokenizer.encode(torch.tensor([[0], [1]])) + start
        self.assertTrue(torch.equal(result["response_ids"], expected_response))
        self.assertIsNotNone(result["audio"])
        self.assertIsNone(result["audio"]["features"])
        self.assertEqual(model.sample_calls, 0)
        self.assertEqual(model.runtime.codec.decode_calls, 1)

    def test_full_codec_sequence_generation_decodes_complete_codes(self):
        for use_cache in (False, True):
            with self.subTest(use_cache=use_cache):
                model = _FullSequenceGenerationModel()
                request = Request(prompt_ids=torch.tensor([1]), task=Task.TTS)

                result = generate_responses(
                    [request],
                    model,
                    max_new_tokens=8,
                    do_sample=False,
                    use_cache=use_cache,
                )[0]

                start, _ = model.runtime.codec_audio_range
                tokenizer = model.runtime.audio_tokenizer
                expected_local = tokenizer.encode(
                    torch.tensor([[1, 5], [2, 6]])
                )
                expected_response = expected_local + start
                self.assertTrue(
                    torch.equal(result["response_ids"], expected_response)
                )
                self.assertTrue(model.allowed_token_ids)
                self.assertTrue(all(ids is None for ids in model.allowed_token_ids))
                self.assertTrue(
                    torch.equal(model.generation_inputs[0], torch.tensor([[1]]))
                )
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
                self.assertTrue(
                    torch.equal(audio["waveform"], torch.tensor([6.0, 8.0]))
                )

    def test_single_codebook_generation_does_not_force_prefix_or_code_range(self):
        codes = torch.tensor([[1], [2]])
        for use_cache in (False, True):
            with self.subTest(use_cache=use_cache):
                model = _FullSequenceGenerationModel(codes, codebook_sizes=(4,))
                request = Request(prompt_ids=torch.tensor([1]), task=Task.TTS)

                result = generate_responses(
                    [request],
                    model,
                    max_new_tokens=5,
                    do_sample=False,
                    use_cache=use_cache,
                )[0]

                start, _ = model.runtime.codec_audio_range
                expected_local = model.runtime.audio_tokenizer.encode(codes)
                self.assertTrue(
                    torch.equal(result["response_ids"], expected_local + start)
                )
                self.assertTrue(
                    torch.equal(model.generation_inputs[0], torch.tensor([[1]]))
                )
                self.assertTrue(model.allowed_token_ids)
                self.assertTrue(all(ids is None for ids in model.allowed_token_ids))
                self.assertIsNotNone(result["audio"])
                self.assertTrue(
                    torch.equal(
                        model.runtime.codec.decoded_codes,
                        codes.unsqueeze(0),
                    )
                )

    def test_short_flattened_generation_warns_and_skips_invalid_decode(self):
        model = _FullSequenceGenerationModel(
            torch.tensor([[1, 5]]),
            codebook_sizes=(4, 10),
        )

        with self.assertWarnsRegex(RuntimeWarning, "skipping invalid"):
            result = generate_responses(
                [Request(prompt_ids=torch.tensor([1]), task=Task.TTS)],
                model,
                max_new_tokens=2,
                do_sample=False,
                use_cache=False,
            )[0]

        self.assertIsNone(result["audio"])
        self.assertIn("decode_error", result)

    def test_incomplete_multi_codebook_sequence_warns_and_skips_decode(self):
        model = _FullSequenceGenerationModel()
        start, _ = model.runtime.codec_audio_range
        tokenizer = model.runtime.audio_tokenizer
        model._tokens = [
            start + tokenizer.codebook_token_ids[0],
            *([start + 1] * 8),
        ]

        with self.assertWarnsRegex(RuntimeWarning, "skipping invalid"):
            result = generate_responses(
                [Request(prompt_ids=torch.tensor([1]), task=Task.TTS)],
                model,
                max_new_tokens=8,
                do_sample=False,
                use_cache=False,
            )[0]

        self.assertIsNone(result["audio"])
        self.assertIn("decode_error", result)

    def test_single_codebook_generation_truncates_and_skips_invalid_decode(self):
        model = _FullSequenceGenerationModel(
            torch.tensor([[1]]),
            codebook_sizes=(4,),
        )
        start, _ = model.runtime.codec_audio_range
        model._tokens = [start + 1] * 4

        with self.assertWarnsRegex(RuntimeWarning, "skipping invalid"):
            result = generate_responses(
                [Request(prompt_ids=torch.tensor([1]), task=Task.TTS)],
                model,
                max_new_tokens=4,
                do_sample=False,
                use_cache=False,
            )[0]

        self.assertIsNone(result["audio"])
        self.assertFalse(
            bool(result["response_ids"].eq(model.runtime.eoa_token_id).any())
        )
        self.assertEqual(int(result["response_ids"].numel()), 4)
        self.assertIn("decode_error", result)

    def test_single_codebook_generation_allows_immediate_eoa_and_skips_decode(self):
        model = _FullSequenceGenerationModel(
            torch.tensor([[1]]),
            codebook_sizes=(4,),
        )
        model._tokens = [model.runtime.eoa_token_id]

        with self.assertWarnsRegex(RuntimeWarning, "skipping invalid"):
            result = generate_responses(
                [Request(prompt_ids=torch.tensor([1]), task=Task.TTS)],
                model,
                max_new_tokens=4,
                do_sample=False,
                use_cache=False,
            )[0]

        self.assertEqual(int(result["response_ids"].numel()), 0)
        self.assertIsNone(result["audio"])
        self.assertIn("decode_error", result)

    def test_generation_batches_variable_length_requests(self):
        model = _UnifiedGenerationModel()
        frame_spans = model.runtime.audio_tokenizer.frame_spans
        model.runtime.audio_tokenizer.frame_spans = Mock(wraps=frame_spans)
        first = _request()
        second = _request()
        second["prompt_ids"] = torch.tensor([2, 1, 6])

        results = generate_responses(
            [first, second], model, max_new_tokens=6, do_sample=False
        )

        self.assertEqual(len(results), 2)
        self.assertEqual([call[1] for call in model.calls], [2, 2, 2, 2])
        self.assertEqual(model.runtime.audio_tokenizer.frame_spans.call_count, 0)
        self.assertEqual(model.runtime.codec.decode_calls, 2)

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

    def test_semantic_decode_falls_back_to_isolate_invalid_row(self):
        model = _TokenGenerationModel()
        codec = _RowFailingSemanticCodec()
        model.runtime._semantic_codec = codec
        sequence = torch.tensor(
            [
                [4, 6, 4, 4, 7],
                [4, 6, 5, 5, 7],
            ]
        )

        with (
            patch.object(model, "generate_tokens", return_value=sequence),
            self.assertWarnsRegex(RuntimeWarning, "invalid semantic row"),
        ):
            results = generate_responses(
                [_request(), _request()],
                model,
                max_new_tokens=3,
                do_sample=False,
            )

        self.assertIsNotNone(results[0]["audio"])
        self.assertIsNone(results[1]["audio"])
        self.assertEqual(
            results[1].get("decode_error", {}).get("message"), "invalid semantic row"
        )
        self.assertEqual(codec.decode_batch_sizes, [2, 1, 1])

    def test_acoustic_decode_falls_back_to_isolate_invalid_row(self):
        model = _GenerationModel()
        codec = _RowFailingAcousticCodec()
        model.runtime.codec = codec
        generated = {
            "sequence": torch.tensor(
                [
                    [4, 6, 4, 4, 7],
                    [4, 6, 5, 5, 7],
                ]
            ),
            "features": torch.zeros(2, 2, 2),
            "frame_counts": torch.tensor([2, 2]),
        }

        with (
            patch.object(model, "generate_audio_features", return_value=generated),
            self.assertWarnsRegex(RuntimeWarning, "invalid acoustic row"),
        ):
            results = generate_responses(
                [_request(), _request()],
                model,
                max_new_tokens=3,
                do_sample=False,
            )

        self.assertIsNotNone(results[0]["audio"])
        self.assertIsNone(results[1]["audio"])
        self.assertEqual(
            results[1].get("decode_error", {}).get("message"), "invalid acoustic row"
        )
        self.assertEqual(codec.decode_batch_sizes, [2, 1, 1])

    def test_grouped_decode_propagates_oom_without_row_fallback(self):
        model = _TokenGenerationModel()
        codec = _UnifiedCodec()
        model.runtime._semantic_codec = codec
        error = torch.OutOfMemoryError("codec allocation failed")

        with patch.object(codec, "decode", side_effect=error) as decode:
            with self.assertRaises(torch.OutOfMemoryError) as raised:
                generate_responses(
                    [_request(), _request()],
                    model,
                    max_new_tokens=3,
                    do_sample=False,
                )

        self.assertIs(raised.exception, error)
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(decode.call_args.args[0].size(0), 2)

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
                self.assertEqual(model.batch_sizes, [2, 2])
                self.assertEqual(model.cache_selections, [])

    def test_sampling_skips_finished_rows_without_compacting_forward(self):
        model = _VariableStopModel()

        with patch("torch.multinomial", wraps=torch.multinomial) as multinomial:
            generated = model.generate_tokens(
                torch.tensor([[1], [2]]),
                max_new_tokens=3,
                stop_token_id=model.runtime.eos_token_id,
                allowed_token_ids=(model.runtime.eos_token_id, 1),
                do_sample=True,
                use_cache=True,
            )

        self.assertTrue(
            torch.equal(generated, torch.tensor([[1, 3, 3], [2, 1, 3]]))
        )
        self.assertEqual(model.batch_sizes, [2, 2])
        self.assertEqual(
            [call.args[0].size(0) for call in multinomial.call_args_list],
            [2, 1],
        )

    def test_cache_collects_audio_condition_online(self):
        model = _GenerationModel()

        result = generate_responses(
            [_request()],
            model,
            max_new_tokens=3,
            do_sample=False,
        )[0]

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
            predictions=[Task.S2ST.prediction_modality, Task.S2ST.prediction_modality],
            pad_token_id=0,
        )

        requests = requests_from_batch(batch)

        self.assertTrue(torch.equal(requests[0]["prompt_ids"], torch.tensor([1, 6])))
        self.assertTrue(torch.equal(requests[1]["prompt_ids"], torch.tensor([2, 6])))

    def test_parallel_mixed_generation_decodes_audio_span(self):
        # TEXT token, EOS, then forced BOA, codec token, EOA.
        model = _MixedScriptModel([2, 3, 2, 4, 7])
        result = generate_responses(
            [_mixed_request(Task.PARALLEL_AR)],
            model,
            max_new_tokens=8,
            do_sample=False,
        )[0]
        self.assertTrue(torch.equal(result["response_ids"], torch.tensor([2, 3, 6, 4, 7])))
        self.assertIsNotNone(result["audio"])
        audio = cast(dict, result["audio"])
        self.assertEqual(audio["sample_rate"], 16_000)
        self.assertEqual(model.runtime._semantic_codec.decode_calls, 1)

    def test_parallel_mixed_generation_preserves_audio_decode_error(self):
        model = _MixedScriptModel([2, 3, 2, 5, 7])
        model.runtime._semantic_codec = _RowFailingSemanticCodec()

        with self.assertWarnsRegex(RuntimeWarning, "invalid semantic row"):
            result = generate_responses(
                [_mixed_request(Task.PARALLEL_AR)],
                model,
                max_new_tokens=8,
                do_sample=False,
            )[0]

        self.assertTrue(
            torch.equal(result["response_ids"], torch.tensor([2, 3, 6, 5, 7]))
        )
        self.assertIsNone(result["audio"])
        self.assertEqual(
            result.get("decode_error", {}).get("message"),
            "invalid semantic row",
        )

    def test_parallel_mixed_sampling_skips_forced_boa_row(self):
        model = _MixedScriptModel([2, 3, 2, 4, 7])

        with patch("torch.multinomial", wraps=torch.multinomial) as multinomial:
            result = generate_responses(
                [_mixed_request(Task.PARALLEL_AR)],
                model,
                max_new_tokens=8,
                top_p=0.9,
                do_sample=True,
            )[0]

        self.assertTrue(
            torch.equal(result["response_ids"], torch.tensor([2, 3, 6, 4, 7]))
        )
        self.assertEqual(
            [call.args[0].size(0) for call in multinomial.call_args_list],
            [1, 1, 0, 1, 1],
        )

    def test_mixed_sampling_skips_done_rows_without_compacting_forward(self):
        model = _MixedScriptModel([[3, 2], [2, 3]])

        with patch("torch.multinomial", wraps=torch.multinomial) as multinomial:
            results = generate_responses(
                [
                    _mixed_request(Task.INTERLEAVED_AR),
                    _mixed_request(Task.INTERLEAVED_AR),
                ],
                model,
                max_new_tokens=3,
                do_sample=True,
                use_cache=False,
            )

        self.assertTrue(torch.equal(results[0]["response_ids"], torch.tensor([3])))
        self.assertTrue(
            torch.equal(results[1]["response_ids"], torch.tensor([2, 3]))
        )
        self.assertEqual(model.batch_sizes, [2, 2])
        self.assertEqual(
            [call.args[0].size(0) for call in multinomial.call_args_list],
            [2, 1],
        )

    def test_interleaved_mixed_generation_decodes_audio_span(self):
        # TEXT, BOA, codec, EOA, EOS.
        model = _MixedScriptModel([2, 6, 4, 7, 3])
        result = generate_responses(
            [_mixed_request(Task.INTERLEAVED_AR)],
            model,
            max_new_tokens=8,
            do_sample=False,
        )[0]
        self.assertTrue(
            torch.equal(result["response_ids"], torch.tensor([2, 6, 4, 7, 3]))
        )
        self.assertIsNotNone(result["audio"])
        self.assertEqual(model.runtime._semantic_codec.decode_calls, 1)

    def test_mixed_generation_rejects_acoustic_feature_side_channel(self):
        model = _MixedAcousticModel([2, 3, 4, 7])
        with self.assertRaisesRegex(ValueError, "acoustic feature side channel"):
            generate_responses(
                [_mixed_request(Task.PARALLEL_AR)],
                model,
                max_new_tokens=8,
                do_sample=False,
            )


class _MixedScriptModel(Model):
    def __init__(self, script: Sequence[int | Sequence[int]]) -> None:
        nn.Module.__init__(self)
        self.runtime = _Runtime()
        self.runtime._semantic_codec = _UnifiedCodec()
        self.layout = self.runtime.layout
        self.audio_token_frame_spans = torch.tensor([1, 1])
        self.backbone = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty(0))
        )
        self._script = list(script)
        self._step = 0
        self.batch_sizes: list[int] = []

    def generation_step(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor,
        output_hidden_states: bool,
        token_ids: Tensor | None,
        token_kind: str | None = None,
        modality: Modality | None,
        past_key_values=None,
        use_cache: bool = False,
        audio_input_positions: Tensor | None = None,
        audio_head_past: object | None = None,
        input_modalities: frozenset[Modality] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
    ) -> GenerationStepResult:
        del (
            attention_mask,
            token_kind,
            modality,
            audio_input_positions,
            audio_head_past,
            input_modalities,
            validate_input,
            validate_audio_input_positions,
        )
        if self._step >= len(self._script):
            raise RuntimeError("mixed script exhausted.")
        scripted_ids = self._script[self._step]
        self._step += 1
        self.batch_sizes.append(input_ids.size(0))
        next_ids = (
            input_ids.new_full((input_ids.size(0),), scripted_ids)
            if isinstance(scripted_ids, int)
            else input_ids.new_tensor(scripted_ids)
        )
        if next_ids.shape != (input_ids.size(0),):
            raise ValueError("mixed script step must provide one token per row.")
        logits = torch.full(
            (*input_ids.shape, self.runtime.layout.vocab_size),
            float("-inf"),
        )
        logits[
            torch.arange(input_ids.size(0), device=input_ids.device),
            -1,
            next_ids,
        ] = 0.0
        if token_ids is not None:
            logits = logits.index_select(-1, token_ids)
        cache = SimpleNamespace(length=input_ids.size(1)) if use_cache else None
        return GenerationStepResult(
            logits=logits,
            past_key_values=cache,
            audio_head_past=None,
            hidden_states=(torch.zeros(*input_ids.shape, 2),)
            if output_hidden_states
            else None,
        )

    def select_audio_head_cache(self, past_key_values, indices):
        del indices
        return past_key_values


class _MixedAcousticModel(FlowModel):
    def __init__(self, script: list[int]) -> None:
        nn.Module.__init__(self)
        self.runtime = _Runtime()
        self.layout = self.runtime.layout
        self.audio_token_frame_spans = torch.tensor([1, 1])
        self.backbone = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty(0))
        )
        self.acoustic_condition = nn.Identity()
        self._script = list(script)
        self._step = 0

    def generation_step(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor,
        output_hidden_states: bool,
        token_ids: Tensor | None,
        token_kind: str | None = None,
        modality: Modality | None,
        past_key_values=None,
        use_cache: bool = False,
        audio_input_positions: Tensor | None = None,
        audio_head_past: object | None = None,
        input_modalities: frozenset[Modality] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
    ) -> GenerationStepResult:
        del (
            attention_mask,
            token_kind,
            modality,
            audio_input_positions,
            past_key_values,
            audio_head_past,
            input_modalities,
            validate_input,
            validate_audio_input_positions,
        )
        next_id = self._script[self._step]
        self._step += 1
        logits = torch.full(
            (*input_ids.shape, self.runtime.layout.vocab_size),
            float("-inf"),
        )
        logits[:, -1, next_id] = 0.0
        if token_ids is not None:
            logits = logits.index_select(-1, token_ids)
        cache = (
            SimpleNamespace(batch_select_indices=lambda indices: None)
            if use_cache
            else None
        )
        return GenerationStepResult(
            logits=logits,
            past_key_values=cache,
            audio_head_past=None,
            hidden_states=(torch.zeros(*input_ids.shape, 2),)
            if output_hidden_states
            else None,
        )

    def select_audio_head_cache(self, past_key_values, indices):
        del indices
        return past_key_values


def _mixed_request(task: Task) -> Request:
    return Request(
        prompt_ids=torch.tensor([1, 2]),
        task=task,
        audio_input_positions=None,
    )


def _model_config() -> ModelConfig:
    return ModelConfig(
        semantic_audio_adapter=None,
        audio_output_adapter=AudioOutputAdapterConfig(
            type=AudioOutputAdapterType.NONE,
        ),
        toy=ToyConfig(
            hidden_size=8,
            intermediate_size=16,
            layers=1,
            heads=2,
            max_position_embeddings=32,
        ),
    )


def _transformer_model_config() -> ModelConfig:
    return ModelConfig(
        semantic_audio_adapter=None,
        audio_output_adapter=AudioOutputAdapterConfig(
            type=AudioOutputAdapterType.TRANSFORMER,
            layers=2,
            heads=2,
            ffn_ratio=2.0,
        ),
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


if __name__ == "__main__":
    unittest.main()
