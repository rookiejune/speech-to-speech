from __future__ import annotations

import unittest

import torch
from anytrain.module.idspace import Layout

from speech_to_speech.model import AdapterType
from speech_to_speech.model.audio_output import (
    AudioOutputAdapterConfig,
    AudioOutputAdapterType,
)
from speech_to_speech.model.base import Config, Model
from speech_to_speech.model.toy import ToyConfig
from speech_to_speech.runtime import AudioRepresentation
from speech_to_speech.runtime.audio_tokenizer import NativeAudioTokenizer


class _Codec:
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_codebook = torch.randn(2, 4)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes[..., 0].float()


class _Runtime:
    def __init__(self) -> None:
        self.layout = Layout(text=(0, 8), audio=(8, 13))
        self.audio_tokenizer = NativeAudioTokenizer(vocab_size=2)
        self.codec = _Codec()
        self.audio_route = None
        self.audio_representation = AudioRepresentation.DECOUPLED
        self.eos_token_id = 3
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.boa_token_id = 10
        self.eoa_token_id = 11
        self.mask_token_id = 12
        self.structured_full_sequence = False

    @property
    def codec_audio_range(self) -> tuple[int, int]:
        return 8, 10

    @property
    def acoustic_side_channel(self) -> bool:
        return False


class TokenEmbeddingStructureTest(unittest.TestCase):
    def test_idspace_routes_and_ties_heads(self) -> None:
        model = Model(
            Config(
                semantic_audio_adapter=AdapterType.LINEAR,
                audio_output_adapter=AudioOutputAdapterConfig(
                    type=AudioOutputAdapterType.LINEAR,
                ),
                toy=ToyConfig(
                    hidden_size=8,
                    intermediate_size=16,
                    layers=1,
                    heads=2,
                    max_position_embeddings=32,
                ),
            ),
            runtime=_Runtime(),
        ).eval()
        ids = torch.tensor([[1, 8, 2, 9]])
        embeds = model._input_embedding(ids)
        self.assertEqual(tuple(embeds.shape), (1, 4, 8))
        hidden = torch.randn(1, 4, 8)
        text = model.text_logits(hidden)
        adapted, _ = model.project_audio_hidden(hidden)
        audio = model.semantic_audio_logits(adapted)
        audio_table = model.token_embedding.embeddings["audio"]
        audio_adapter = getattr(
            model.token_embedding.adapters["audio"],
            "module",
            model.token_embedding.adapters["audio"],
        )
        self.assertIsInstance(audio_table, torch.nn.Embedding)
        self.assertIsInstance(audio_adapter, torch.nn.Linear)
        self.assertEqual(
            text.shape[-1],
            model.token_embedding.embeddings["text"].num_embeddings,
        )
        self.assertEqual(audio.shape[-1], audio_table.num_embeddings)
        with torch.no_grad():
            tied_text = torch.nn.functional.linear(
                hidden,
                model.token_embedding.embeddings["text"].weight,
            )
            tied_audio = torch.nn.functional.linear(
                adapted,
                audio_table.weight,
            )
        torch.testing.assert_close(text, tied_text)
        torch.testing.assert_close(audio, tied_audio)


if __name__ == "__main__":
    unittest.main()
