from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

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
from anytrain.codec import AcousticLayout

from speech_to_speech.datamodule.builder import build_speech_sample
from speech_to_speech.datamodule.batch import ModelBatch, ModelSample
from speech_to_speech.datamodule.parse import parse_sample
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
from speech_to_speech.runtime import Config, InputAudioConfig, Runtime
from speech_to_speech.task import Task


class DualAudioTokenizerTest(unittest.TestCase):
    def test_runtime_allocates_distinct_input_and_output_blocks(self) -> None:
        runtime = _runtime()

        self.assertEqual(
            runtime.layout.block_names,
            ("text", "audio_input", "audio"),
        )
        self.assertEqual(runtime.input_codec_audio_range, (8, 24))
        self.assertEqual(runtime.input_boa_token_id, 24)
        self.assertEqual(runtime.input_eoa_token_id, 25)
        self.assertEqual(runtime.codec_audio_range, (26, 34))
        self.assertEqual(runtime.boa_token_id, 34)
        self.assertEqual(runtime.eoa_token_id, 35)
        self.assertEqual(runtime.mask_token_id, 36)
        self.assertEqual(runtime.audio_generation_allowed_ids[0], 26)
        self.assertNotIn(8, runtime.audio_generation_allowed_ids)

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
        input_start, input_end = runtime.input_codec_audio_range
        self.assertTrue(bool(source_ids.ge(input_start).all()))
        self.assertTrue(bool(source_ids.lt(input_end).all()))
        self.assertIn(runtime.input_boa_token_id, prompt.tolist())
        self.assertIn(runtime.input_eoa_token_id, prompt.tolist())

        response = built.labels.response_ids
        output_start, output_end = runtime.codec_audio_range
        self.assertEqual(int(prompt[-1]), runtime.boa_token_id)
        self.assertEqual(int(response[-1]), runtime.eoa_token_id)
        self.assertTrue(bool(response[:-1].ge(output_start).all()))
        self.assertTrue(bool(response[:-1].lt(output_end).all()))

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
            codec="longcat",
            input_audio=InputAudioConfig(
                codec="glm4",
                vocab_size=16,
                frame_rate=12.5,
            ),
        )
    )
    runtime.__dict__["codec"] = _codec()
    runtime.__dict__["text_tokenizer"] = _TextTokenizer()
    return runtime


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


if __name__ == "__main__":
    unittest.main()
