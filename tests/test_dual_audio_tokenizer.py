from __future__ import annotations

import unittest
from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
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
from anytrain.codec import (
    AcousticLayout,
    AudioCodeSchema,
    AudioCodeSpec,
    SemanticGlobalCodes,
)

from speech_to_speech.audio import AudioCodes
from speech_to_speech.callback import OnDeviceCodecMaterializer
from speech_to_speech.datamodule.collate import Collator
from speech_to_speech.datamodule.builder import build_speech_sample
from speech_to_speech.datamodule.batch import ModelBatch, ModelSample
from speech_to_speech.datamodule.parse import parse_sample
from speech_to_speech.datamodule.sample import AudioContextSample, RawSpeechBatch
from speech_to_speech.generation import chat as chat_adapter
from speech_to_speech.model import (
    AdapterType,
    AudioInputAdapterConfig,
    AudioInputAdapterType,
    AudioOutputAdapterConfig,
    AudioOutputAdapterType,
    Config as ModelConfig,
    Model,
    ToyConfig,
)
from speech_to_speech.runtime import (
    AudioInputConfig,
    AudioOutputConfig,
    AudioSequenceLayout,
    Config,
    InputAudioConfig,
    Runtime,
)
from speech_to_speech.runtime.audio_tokenizer import (
    BiCodecAudioTokenizer,
    FlattenedAudioTokenizer,
    NativeAudioTokenizer,
)
from speech_to_speech.task import Task


class DualAudioTokenizerTest(unittest.TestCase):
    def test_runtime_allocates_distinct_input_and_output_blocks(self) -> None:
        runtime = _runtime()

        self.assertEqual(
            runtime.layout.block_names,
            ("text", "audio_input", "audio"),
        )
        _, text_end = runtime.layout.blocks[Modality.TEXT.value]
        input_start, input_end = runtime.layout.blocks["audio_input"]
        output_start, _ = runtime.layout.blocks[Modality.AUDIO.value]
        self.assertEqual(input_start, text_end)
        self.assertEqual(
            runtime.input_codec_audio_range,
            (input_start, input_start + 16_384),
        )
        self.assertEqual(runtime.input_boa_token_id, input_start + 16_384)
        self.assertEqual(runtime.input_eoa_token_id, runtime.input_boa_token_id + 1)
        self.assertLess(runtime.input_eoa_token_id, input_end)
        self.assertEqual(runtime.codec_audio_range, (output_start, output_start + 8))
        self.assertEqual(runtime.boa_token_id, output_start + 8)
        self.assertEqual(runtime.eoa_token_id, output_start + 9)
        self.assertEqual(runtime.mask_token_id, output_start + 10)
        self.assertEqual(
            runtime.audio_generation_allowed_ids,
            (
                runtime.boa_token_id,
                runtime.audio_schema_token_id,
                *range(output_start, runtime.boa_token_id),
                runtime.eoa_token_id,
            ),
        )
        self.assertNotIn(input_start, runtime.audio_generation_allowed_ids)

    def test_canonical_glm4_input_bicodec_output_keeps_legacy_checkpoint_schema(
        self,
    ) -> None:
        with self.assertWarns(FutureWarning):
            legacy = _bicodec_runtime(
                Config(
                    codec="bicodec",
                    input_audio=InputAudioConfig(
                        codec="glm4",
                        vocab_size=16_384,
                        frame_rate=12.5,
                    ),
                )
            )
        canonical = _bicodec_runtime(
            Config(
                audio_input=AudioInputConfig(
                    tokenizer="glm4",
                ),
                audio_output=AudioOutputConfig(
                    tokenizer="bicodec",
                    detokenizer="bicodec",
                ),
            )
        )

        legacy_model = _checkpoint_model(legacy)
        canonical_model = _checkpoint_model(canonical)

        self.assertEqual(
            legacy_model.checkpoint_contract.sha256,
            canonical_model.checkpoint_contract.sha256,
        )
        self.assertEqual(
            legacy_model.checkpoint_contract.checkpoint_payload(),
            canonical_model.checkpoint_contract.checkpoint_payload(),
        )
        self.assertEqual(
            _state_schema(legacy_model),
            _state_schema(canonical_model),
        )
        self.assertIn(
            "tokens.input_audio_embedding.weight",
            canonical_model.state_dict(),
        )
        self.assertFalse(
            any(
                name.startswith("tokens.output_audio_")
                for name in canonical_model.state_dict()
            )
        )

    def test_canonical_output_config_keeps_coupled_state_ownership(self) -> None:
        legacy_model = _checkpoint_model(
            _bicodec_runtime(Config(codec="bicodec"))
        )
        canonical_model = _checkpoint_model(
            _bicodec_runtime(
                Config(
                    audio_output=AudioOutputConfig(
                        tokenizer="bicodec",
                        detokenizer="bicodec",
                    )
                )
            )
        )

        self.assertEqual(
            legacy_model.checkpoint_contract.checkpoint_payload(),
            canonical_model.checkpoint_contract.checkpoint_payload(),
        )
        self.assertEqual(
            _state_schema(legacy_model),
            _state_schema(canonical_model),
        )
        self.assertNotIn(
            "tokens.input_audio_embedding.weight",
            canonical_model.state_dict(),
        )

    def test_explicit_same_pair_ties_backend_tokenizer_and_embedding(self) -> None:
        runtime = _bicodec_runtime(
            Config(
                audio_input=AudioInputConfig(tokenizer="bicodec"),
                audio_output=AudioOutputConfig(
                    tokenizer="bicodec",
                    detokenizer="bicodec",
                ),
            )
        )

        model = _checkpoint_model(runtime)
        backend = runtime.output_audio_tokenizer_backend

        self.assertIsNone(runtime.config.audio_input)
        self.assertIs(runtime.input_audio_tokenizer_backend, backend)
        self.assertIs(runtime.output_audio_detokenizer, backend)
        self.assertIs(runtime.input_codec, runtime.codec)
        self.assertIs(runtime.input_audio_tokenizer, runtime.audio_tokenizer)
        self.assertIsNone(model.tokens.input_audio_embedding)
        self.assertNotIn("audio_input", runtime.layout.blocks)

    def test_same_backend_different_bpe_reuses_backend_but_not_embedding(self) -> None:
        runtime = Runtime(
            Config(
                audio_input=AudioInputConfig(
                    tokenizer="longcat",
                    bpe="/tmp/input-bpe",
                ),
                audio_output=AudioOutputConfig(
                    tokenizer="longcat",
                    bpe="/tmp/output-bpe",
                ),
            )
        )
        runtime.__dict__["output_audio_tokenizer_backend"] = _codec()
        runtime.__dict__["output_audio_code_spec"] = _longcat_spec()
        runtime.__dict__["text_tokenizer"] = _TextTokenizer()

        with patch(
            "speech_to_speech.runtime.core.audio_tokenizer",
            side_effect=lambda _path: NativeAudioTokenizer(vocab_size=8),
        ):
            output_tokens = runtime.audio_tokenizer
            input_tokens = runtime.input_audio_tokenizer
            model = _checkpoint_model(runtime)

        self.assertTrue(runtime.input_audio_backend_shared)
        self.assertTrue(runtime.input_audio_decoupled)
        self.assertIs(
            runtime.input_audio_tokenizer_backend,
            runtime.output_audio_tokenizer_backend,
        )
        self.assertIs(
            runtime.output_audio_detokenizer,
            runtime.output_audio_tokenizer_backend,
        )
        self.assertIs(runtime.input_codec, runtime.codec)
        self.assertIsNot(input_tokens, output_tokens)
        self.assertIsNotNone(model.tokens.input_audio_embedding)
        self.assertIn("tokens.input_audio_embedding.weight", model.state_dict())
        runtime_contract = cast(
            Mapping[str, object],
            cast(Mapping[str, object], model.checkpoint_contract.components)["runtime"],
        )
        audio_codecs = cast(Mapping[str, object], runtime_contract["audio_codecs"])
        token_space = cast(Mapping[str, object], runtime_contract["token_space"])
        audio_schemas = cast(
            Mapping[str, object],
            token_space["audio_schemas"],
        )
        self.assertEqual(audio_codecs["sharing"], "shared")
        self.assertEqual(audio_schemas["sharing"], "independent")

    def test_distinct_loadable_backends_keep_waveform_fallback(self) -> None:
        runtime = Runtime(
            Config(
                audio_input=AudioInputConfig(tokenizer="unicodec"),
                audio_output=AudioOutputConfig(tokenizer="longcat"),
            )
        )
        output_codec = _codec()
        output_codec.encode.return_value = torch.tensor(
            [[[1, 2], [2, 3]]],
            dtype=torch.long,
        )
        input_codec = _frame_codec()
        runtime.__dict__["output_audio_tokenizer_backend"] = output_codec
        runtime.__dict__["input_audio_tokenizer_backend"] = input_codec
        runtime.__dict__["text_tokenizer"] = _TextTokenizer()

        raw = Collator(
            runtime,
            {Task.S2ST: 1.0},
            encode_missing_codes=True,
        )([_waveform_sample()])
        self.assertIsInstance(raw, RawSpeechBatch)

        batch = OnDeviceCodecMaterializer(runtime)(
            raw,
            device=torch.device("cpu"),
        )

        self.assertIsInstance(batch, ModelBatch)
        self.assertEqual(tuple(input_codec.encode.call_args.args[0].shape), (1, 1, 4))
        self.assertEqual(tuple(output_codec.encode.call_args.args[0].shape), (1, 1, 6))

    def test_glm4_waveform_fallback_still_requires_prepared_codes(self) -> None:
        with self.assertRaisesRegex(ValueError, "no runtime codec backend"):
            Collator(
                _runtime(),
                {Task.S2ST: 1.0},
                encode_missing_codes=True,
            )([_waveform_sample()])

        with self.assertRaisesRegex(ValueError, "no runtime codec backend"):
            chat_adapter._materialize_input_codes(
                {
                    "type": "audio",
                    "waveform": torch.ones(1, 4),
                    "sample_rate": 4,
                },
                _runtime(),
            )

    def test_loadable_input_chat_waveform_uses_input_backend(self) -> None:
        runtime = Runtime(
            Config(
                audio_input=AudioInputConfig(tokenizer="unicodec"),
                audio_output=AudioOutputConfig(tokenizer="longcat"),
            )
        )
        output_codec = _codec()
        input_codec = _frame_codec()
        runtime.__dict__["output_audio_tokenizer_backend"] = output_codec
        runtime.__dict__["input_audio_tokenizer_backend"] = input_codec

        codes = chat_adapter._materialize_input_codes(
            {
                "type": "audio",
                "waveform": torch.ones(1, 4),
                "sample_rate": 4,
            },
            runtime,
        )

        self.assertIsInstance(codes, torch.Tensor)
        input_codec.encode.assert_called_once()
        output_codec.encode.assert_not_called()

    def test_decoupled_bicodec_chat_codes_preserve_global_stream(self) -> None:
        runtime = Runtime(
            Config(
                audio_input=AudioInputConfig(tokenizer="bicodec"),
                audio_output=AudioOutputConfig(tokenizer="longcat"),
            ),
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
        )
        codes = AudioCodes(
            semantic_codes=torch.tensor([[1], [2]], dtype=torch.long),
            global_codes=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        )

        materialized = chat_adapter._materialize_input_codes(
            {
                "type": "codec_codes",
                "codec": "bicodec",
                "codes": codes,
            },
            runtime,
        )

        self.assertIs(materialized, codes)
        self.assertIsInstance(materialized, AudioCodes)
        assert isinstance(materialized, AudioCodes)
        self.assertIs(materialized.global_codes, codes.global_codes)

    def test_same_bicodec_backend_different_bpe_chat_serializes_full_input(
        self,
    ) -> None:
        runtime = Runtime(
            Config(
                audio_input=AudioInputConfig(
                    tokenizer="bicodec",
                    bpe="/tmp/input-bpe",
                ),
                audio_output=AudioOutputConfig(
                    tokenizer="bicodec",
                    bpe="/tmp/output-bpe",
                ),
            ),
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
        )
        runtime.__dict__["output_audio_tokenizer_backend"] = _bicodec_codec()
        runtime.__dict__["output_audio_code_spec"] = _bicodec_spec()
        runtime.__dict__["text_tokenizer"] = _TextTokenizer()
        codes = AudioCodes(
            semantic_codes=torch.tensor([[1], [2]], dtype=torch.long),
            global_codes=torch.tensor(
                [[0, 1], [2, 3], [4, 5]],
                dtype=torch.long,
            ),
        )

        with patch(
            "speech_to_speech.runtime.core.audio_tokenizer",
            side_effect=lambda _path: NativeAudioTokenizer(vocab_size=8),
        ):
            input_tokenizer = runtime.input_audio_tokenizer
            private = chat_adapter.to_request(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "codec_codes",
                                    "codec": "bicodec",
                                    "codes": codes,
                                }
                            ],
                        }
                    ],
                    "task": Task.ASR,
                },
                runtime,
            )

        positions = private["audio_input_positions"]
        self.assertIsNotNone(positions)
        assert positions is not None
        expected = runtime.layout.to_global(
            runtime.input_audio_block_name,
            input_tokenizer.encode_full(codes),
        )
        actual = private["prompt_ids"].index_select(0, positions)
        torch.testing.assert_close(actual, expected)

    def test_prepared_glm4_input_can_train_while_bicodec_output_is_encoded(self) -> None:
        runtime = _bicodec_runtime(
            Config(
                audio_input=AudioInputConfig(
                    tokenizer="glm4",
                ),
                audio_output=AudioOutputConfig(
                    tokenizer="bicodec",
                    detokenizer="bicodec",
                ),
            )
        )
        runtime.codec.tokenize.return_value = SemanticGlobalCodes(
            semantic=torch.tensor([[[1], [2], [3]]], dtype=torch.long),
            global_codes=torch.tensor(
                [[[0, 1], [2, 3], [4, 5]]],
                dtype=torch.long,
            ),
        )

        raw = Collator(
            runtime,
            {Task.S2ST: 1.0},
            encode_missing_codes=True,
        )([_prepared_input_raw_output_sample()])
        self.assertIsInstance(raw, RawSpeechBatch)

        batch = OnDeviceCodecMaterializer(runtime)(
            raw,
            device=torch.device("cpu"),
        )

        self.assertIsInstance(batch, ModelBatch)
        self.assertNotIn("input_audio_tokenizer_backend", runtime.__dict__)
        self.assertEqual(runtime.codec.tokenize.call_count, 1)
        self.assertEqual(
            tuple(runtime.codec.tokenize.call_args.args[0].shape),
            (1, 1, 6),
        )

    def test_raw_audio_context_uses_output_backend(self) -> None:
        runtime = _bicodec_runtime(
            Config(
                audio_input=AudioInputConfig(
                    tokenizer="glm4",
                ),
                audio_output=AudioOutputConfig(
                    tokenizer="bicodec",
                    detokenizer="bicodec",
                ),
            )
        )
        runtime.codec.tokenize.return_value = SemanticGlobalCodes(
            semantic=torch.tensor([[[1], [2], [3]]], dtype=torch.long),
            global_codes=torch.tensor(
                [[[0, 1], [2, 3], [4, 5]]],
                dtype=torch.long,
            ),
        )

        raw = Collator(
            runtime,
            {Task.S2ST: 1.0},
            encode_missing_codes=True,
        )([_prepared_pair_with_raw_context()])
        self.assertIsInstance(raw, RawSpeechBatch)

        batch = OnDeviceCodecMaterializer(runtime)(
            raw,
            device=torch.device("cpu"),
        )

        self.assertIsInstance(batch, ModelBatch)
        self.assertNotIn("input_audio_tokenizer_backend", runtime.__dict__)
        self.assertEqual(runtime.codec.tokenize.call_count, 1)
        self.assertEqual(
            tuple(runtime.codec.tokenize.call_args.args[0].shape),
            (1, 1, 5),
        )

    def test_bicodec_input_preserves_global_and_semantic_streams(self) -> None:
        runtime = Runtime(
            Config(
                audio_input=AudioInputConfig(tokenizer="bicodec"),
                audio_output=AudioOutputConfig(tokenizer="longcat"),
            )
        )
        runtime.__dict__["output_audio_tokenizer_backend"] = _codec()
        runtime.__dict__["input_audio_tokenizer_backend"] = _bicodec_codec()
        runtime.__dict__["output_audio_code_spec"] = _longcat_spec()
        runtime.__dict__["input_audio_code_spec"] = _bicodec_spec()
        runtime.__dict__["text_tokenizer"] = _TextTokenizer()
        semantic = torch.tensor([[1], [2], [3]], dtype=torch.long)
        global_codes = torch.tensor(
            [[0, 1], [2, 3], [4, 5]],
            dtype=torch.long,
        )
        sample = _sample()
        sample[Role.SOURCE, Modality.AUDIO] = AudioItem(
            views={
                AudioView.BICODEC: {
                    "semantic": semantic,
                    "global": global_codes,
                }
            },
            meta={AudioMeta.DURATION: 0.06},
        )

        source = parse_sample(sample, runtime).source

        tokenizer = runtime.input_audio_tokenizer
        self.assertIsInstance(tokenizer, BiCodecAudioTokenizer)
        assert isinstance(tokenizer, BiCodecAudioTokenizer)
        decoded = tokenizer.decode_full(source.audio_token_ids)
        torch.testing.assert_close(decoded.semantic_codes, semantic)
        torch.testing.assert_close(decoded.global_codes, global_codes)

    def test_flattened_longcat_input_preserves_all_codebooks(self) -> None:
        runtime = Runtime(
            Config(
                audio_input=AudioInputConfig(tokenizer="longcat"),
                audio_output=AudioOutputConfig(tokenizer="bicodec"),
            ),
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
        )
        runtime.__dict__["output_audio_tokenizer_backend"] = _bicodec_codec()
        runtime.__dict__["input_audio_tokenizer_backend"] = _codec()
        runtime.__dict__["output_audio_code_spec"] = _bicodec_spec()
        runtime.__dict__["input_audio_code_spec"] = _longcat_spec()
        runtime.__dict__["text_tokenizer"] = _TextTokenizer()
        source_codes = torch.tensor(
            [[1, 2], [2, 3], [3, 0]],
            dtype=torch.long,
        )
        sample = _sample()
        sample[Role.SOURCE, Modality.AUDIO] = AudioItem(
            views={AudioView.LONGCAT: source_codes},
            meta={AudioMeta.DURATION: 0.06},
        )
        sample[Role.TARGET, Modality.AUDIO] = AudioItem(
            views={
                AudioView.BICODEC: {
                    "semantic": torch.tensor([[1], [2], [3]]),
                    "global": torch.tensor([[0, 1], [2, 3], [4, 5]]),
                }
            },
            meta={AudioMeta.DURATION: 0.06},
        )

        source = parse_sample(sample, runtime).source

        tokenizer = runtime.input_audio_tokenizer
        self.assertIsInstance(tokenizer, FlattenedAudioTokenizer)
        assert isinstance(tokenizer, FlattenedAudioTokenizer)
        decoded = tokenizer.decode(source.audio_token_ids)
        assert isinstance(decoded, torch.Tensor)
        torch.testing.assert_close(decoded, source_codes)

    def test_parser_and_builder_route_glm4_source_to_input_block(self) -> None:
        runtime = _runtime()
        pair = parse_sample(_sample(), runtime)

        built = build_speech_sample(
            pair.source,
            pair.target,
            Task.S2ST,
            runtime,
            prompt="translate $$$PLACEHOLDER$$$ now",
        )

        positions = built.request["audio_input_positions"]
        self.assertIsNotNone(positions)
        assert positions is not None
        prompt = built.request["prompt_ids"]
        source_ids = prompt.index_select(0, positions)
        input_start, input_end = runtime.layout.blocks["audio_input"]
        self.assertTrue(bool(source_ids.ge(input_start).all()))
        self.assertTrue(bool(source_ids.lt(input_end).all()))
        codec_start, codec_end = runtime.input_codec_audio_range
        self.assertTrue(
            bool((source_ids.ge(codec_start) & source_ids.lt(codec_end)).any())
        )
        self.assertIn(runtime.input_boa_token_id, prompt.tolist())
        self.assertIn(runtime.input_eoa_token_id, prompt.tolist())
        self.assertIn(runtime.input_audio_schema_token_id, prompt.tolist())

        response = built.labels.response_ids
        output_start, output_end = runtime.codec_audio_range
        self.assertEqual(int(response[0]), runtime.boa_token_id)
        self.assertEqual(int(response[1]), runtime.audio_schema_token_id)
        self.assertEqual(int(response[-1]), runtime.eoa_token_id)
        self.assertTrue(bool(response[2:-1].ge(output_start).all()))
        self.assertTrue(bool(response[2:-1].lt(output_end).all()))

    def test_model_routes_input_embedding_but_never_predicts_input_ids(self) -> None:
        runtime = _runtime()
        model = Model(
            ModelConfig(
                semantic_audio_adapter=AdapterType.LINEAR,
                audio_input_adapter=AudioInputAdapterConfig(
                    type=AudioInputAdapterType.NONE,
                ),
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
            ),
            runtime=cast(Any, runtime),
        )
        input_embedding = model.tokens.input_audio_embedding
        self.assertIsInstance(input_embedding, torch.nn.Embedding)
        assert input_embedding is not None
        self.assertIn("tokens.input_audio_embedding.weight", model.state_dict())

        input_start, _ = runtime.input_codec_audio_range
        output_start, _ = runtime.codec_audio_range
        input_ids = torch.tensor([[1, input_start, output_start]], dtype=torch.long)
        modalities = frozenset({Modality.TEXT, Modality.AUDIO})
        embedded = model.tokens.embed(
            input_ids,
            model.text_embedding,
            input_modalities=modalities,
            validate=False,
        )
        self.assertEqual(tuple(embedded.shape), (1, 3, 8))

        model.zero_grad(set_to_none=True)
        model.tokens.embed(
            torch.tensor([[input_start]], dtype=torch.long),
            model.text_embedding,
            input_modalities=frozenset({Modality.AUDIO}),
            validate=False,
        ).sum().backward()
        self.assertIsNotNone(input_embedding.weight.grad)

        model.zero_grad(set_to_none=True)
        hidden = torch.randn(1, 1, 8, requires_grad=True)
        model.semantic_audio_logits(model.project_audio_hidden(hidden)[0]).sum().backward()
        self.assertIsNone(input_embedding.weight.grad)
        self.assertIsNotNone(model.tokens.audio_embedding.weight.grad)

        dense = model.token_logits(torch.randn(1, 2, 8))
        block_start, block_end = runtime.layout.blocks["audio_input"]
        self.assertTrue(bool(torch.isneginf(dense[..., block_start:block_end]).all()))
        with self.assertRaisesRegex(ValueError, "invalid vocabulary id"):
            model.selected_logits(
                torch.randn(1, 1, 8),
                torch.tensor([input_start]),
            )
        with self.assertRaisesRegex(ValueError, "invalid vocabulary id"):
            model.selected_logits(
                torch.randn(1, 1, 8),
                torch.tensor([input_start]),
                token_kind="audio",
                validate=False,
            )

        continuation = model.tokens.embed(
            torch.tensor([[output_start]], dtype=torch.long),
            model.text_embedding,
            input_modalities=frozenset({Modality.AUDIO}),
            validate=False,
        )
        expected = model.tokens.audio_projection(
            model.tokens.audio_rows(torch.tensor([0]))
        )
        torch.testing.assert_close(continuation[0, 0], expected[0])

    def test_batch_boundary_rejects_input_only_supervision(self) -> None:
        runtime = _runtime()
        pair = parse_sample(_sample(), runtime)
        built = build_speech_sample(
            pair.source,
            pair.target,
            Task.S2ST,
            runtime,
            prompt="translate $$$PLACEHOLDER$$$ now",
        )
        labels = built.labels.token_labels.clone()
        labels[-1] = runtime.input_codec_audio_range[0]
        invalid = ModelSample(
            request=built.request,
            labels=replace(built.labels, token_labels=labels),
        )

        with self.assertRaisesRegex(ValueError, "supervised layout blocks"):
            ModelBatch.from_samples(
                [invalid],
                pad_token_id=runtime.pad_token_id,
                layout=runtime.layout,
            )


def _runtime() -> Runtime:
    runtime = Runtime(
        Config(
            audio_input=AudioInputConfig(tokenizer="glm4"),
            audio_output=AudioOutputConfig(
                tokenizer="longcat",
                detokenizer="longcat",
            ),
        )
    )
    runtime.__dict__["output_audio_tokenizer_backend"] = _codec()
    runtime.__dict__["output_audio_code_spec"] = _longcat_spec()
    runtime.__dict__["text_tokenizer"] = _TextTokenizer()
    return runtime


def _bicodec_runtime(config: Config) -> Runtime:
    runtime = Runtime(
        config,
        audio_sequence_layout=AudioSequenceLayout.FLATTENED,
    )
    runtime.__dict__["output_audio_tokenizer_backend"] = _bicodec_codec()
    runtime.__dict__["output_audio_code_spec"] = _bicodec_spec()
    runtime.__dict__["text_tokenizer"] = _TextTokenizer()
    return runtime


def _checkpoint_model(runtime: Runtime) -> Model:
    return Model(
        ModelConfig(
            semantic_audio_adapter=AdapterType.LINEAR,
            audio_input_adapter=AudioInputAdapterConfig(
                type=AudioInputAdapterType.NONE,
            ),
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
        ),
        runtime=cast(Any, runtime),
    )


def _state_schema(model: Model) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(value.shape)
        for name, value in model.state_dict().items()
    }


def _codec() -> SimpleNamespace:
    return SimpleNamespace(
        sample_rate=16_000,
        frame_rate=50.0,
        codebook_sizes=(8, 4),
        semantic_codebook=torch.randn(8, 4),
        semantic_codebook_sizes=(8,),
        acoustic_codebook_sizes=(4,),
        acoustic_feature_dim=4,
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
        acoustic_unit_length=None,
        encode=Mock(),
        decode=Mock(),
        tokenize=Mock(),
        detokenize=Mock(),
        acoustic_codes_to_features=Mock(),
        decode_features=Mock(),
    )


def _bicodec_codec() -> SimpleNamespace:
    return SimpleNamespace(
        sample_rate=16_000,
        frame_rate=50.0,
        semantic_codebook=torch.randn(8, 4),
        semantic_codebook_sizes=(8,),
        global_codebook_sizes=(5, 7),
        global_feature_dim=4,
        global_unit_length=3,
        tokenize=Mock(),
        detokenize=Mock(),
        global_codes_to_features=Mock(),
        decode_features=Mock(),
    )


def _frame_codec() -> SimpleNamespace:
    return SimpleNamespace(
        sample_rate=16_000,
        frame_rate=75.0,
        codebook_sizes=(4,),
        encode=Mock(
            return_value=torch.tensor(
                [[[1], [2]]],
                dtype=torch.long,
            )
        ),
        decode=Mock(),
    )


def _longcat_spec() -> AudioCodeSpec:
    return AudioCodeSpec(
        view="longcat",
        schema=AudioCodeSchema.SEMANTIC_ACOUSTIC,
        sample_rate=16_000,
        frame_rate=50.0,
        frame_codebook_sizes=(8, 4),
        semantic_codebook_sizes=(8,),
        acoustic_codebook_sizes=(4,),
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
    )


def _bicodec_spec() -> AudioCodeSpec:
    return AudioCodeSpec(
        view="bicodec",
        schema=AudioCodeSchema.SEMANTIC_GLOBAL,
        sample_rate=16_000,
        frame_rate=50.0,
        semantic_codebook_sizes=(8,),
        global_codebook_sizes=(5, 7),
        global_unit_length=3,
    )


def _sample() -> dict[tuple[Role, Modality], object]:
    return {
        (Role.SOURCE, Modality.AUDIO): AudioItem(
            views={AudioView.GLM4: torch.tensor([[1], [2], [3]])},
            meta={AudioMeta.DURATION: 0.24},
        ),
        (Role.SOURCE, Modality.TEXT): TextItem(
            views={TextView.TEXT: "source"},
            meta={TextMeta.LANG: Lang.ZH},
        ),
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={
                AudioView.LONGCAT: torch.tensor(
                    [[1, 2], [2, 3], [3, 0]],
                    dtype=torch.long,
                )
            },
            meta={AudioMeta.DURATION: 0.06},
        ),
        (Role.TARGET, Modality.TEXT): TextItem(
            views={TextView.TEXT: "target"},
            meta={TextMeta.LANG: Lang.EN},
        ),
    }


def _waveform_sample() -> dict[tuple[Role, Modality], object]:
    return {
        (Role.SOURCE, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (torch.ones(1, 4), 4)},
            meta={},
        ),
        (Role.SOURCE, Modality.TEXT): TextItem(
            views={TextView.TEXT: "s"},
            meta={TextMeta.LANG: Lang.ZH},
        ),
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (torch.ones(1, 6), 4)},
            meta={},
        ),
        (Role.TARGET, Modality.TEXT): TextItem(
            views={TextView.TEXT: "t"},
            meta={TextMeta.LANG: Lang.EN},
        ),
    }


def _prepared_input_raw_output_sample() -> dict[tuple[Role, Modality], object]:
    return {
        (Role.SOURCE, Modality.AUDIO): AudioItem(
            views={AudioView.GLM4: torch.tensor([[1], [2], [3]])},
            meta={AudioMeta.DURATION: 0.24},
        ),
        (Role.SOURCE, Modality.TEXT): TextItem(
            views={TextView.TEXT: "s"},
            meta={TextMeta.LANG: Lang.ZH},
        ),
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (torch.ones(1, 6), 4)},
            meta={},
        ),
        (Role.TARGET, Modality.TEXT): TextItem(
            views={TextView.TEXT: "t"},
            meta={TextMeta.LANG: Lang.EN},
        ),
    }


def _prepared_pair_with_raw_context() -> AudioContextSample:
    sample = _sample()
    sample[Role.SOURCE, Modality.TEXT] = TextItem(
        views={TextView.TEXT: "s"},
        meta={TextMeta.LANG: Lang.ZH},
    )
    sample[Role.TARGET, Modality.AUDIO] = AudioItem(
        views={
            AudioView.BICODEC: {
                "semantic": torch.tensor([[1], [2], [3]]),
                "global": torch.tensor([[0, 1], [2, 3], [4, 5]]),
            }
        },
        meta={AudioMeta.DURATION: 0.06},
    )
    sample[Role.TARGET, Modality.TEXT] = TextItem(
        views={TextView.TEXT: "t"},
        meta={TextMeta.LANG: Lang.EN},
    )
    context = {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (torch.ones(1, 5), 4)},
            meta={},
        ),
        (Role.DEFAULT, Modality.TEXT): TextItem(
            views={TextView.TEXT: "r"},
            meta={TextMeta.LANG: Lang.EN},
        ),
    }
    return AudioContextSample(sample=sample, audio_context=context)


class _TextTokenizer:
    vocab_size = 8
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    special_tokens_map: dict[str, str] = {}

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [3 + index % 5 for index, _character in enumerate(text)]

    def apply_chat_template(self, conversation, **kwargs) -> str:
        del kwargs
        return f"<user>{conversation[0]['content']}</user><assistant>"

    def contract_state(self) -> dict[str, object]:
        return {
            "grammar": "dual-tokenizer-text-v1",
            "vocab": list(range(self.vocab_size)),
        }


if __name__ == "__main__":
    unittest.main()
