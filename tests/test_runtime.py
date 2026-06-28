from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from speech_to_speech.config import ModelConfig
from speech_to_speech import runtime


class RuntimeTest(unittest.TestCase):
    def test_qwen3_tokenizer_uses_model_config_cache_key(self) -> None:
        calls: list[tuple[str, bool]] = []

        def from_pretrained(name: str, *, trust_remote_code: bool) -> object:
            calls.append((name, trust_remote_code))
            return object()

        with patch.object(runtime, "_QWEN3_TOKENIZERS", {}):
            with patch("transformers.AutoTokenizer.from_pretrained", from_pretrained):
                first = runtime.qwen3_tokenizer(
                    ModelConfig(
                        model_name_or_path="Qwen/Qwen3-0.6B",
                        trust_remote_code=False,
                    )
                )
                second = runtime.qwen3_tokenizer(
                    ModelConfig(
                        model_name_or_path="Qwen/Qwen3-0.6B",
                        trust_remote_code=False,
                    )
                )
                third = runtime.qwen3_tokenizer(
                    ModelConfig(
                        model_name_or_path="Qwen/Qwen3-8B",
                        trust_remote_code=True,
                    )
                )

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(
            calls,
            [
                ("Qwen/Qwen3-0.6B", False),
                ("Qwen/Qwen3-8B", True),
            ],
        )

    def test_longcat_acoustic_features_uses_codec_feature_boundary(self) -> None:
        codec = FakeLongCatCodec()
        acoustic_codes = torch.tensor([[1, 2], [3, 4]])

        features = runtime.longcat_acoustic_features(
            acoustic_codes,
            codec=codec,
            decoder="24k_4codebooks",
        )

        self.assertTrue(torch.equal(features, codec.features))
        self.assertTrue(torch.equal(codec.acoustic_codes, acoustic_codes.unsqueeze(0)))
        self.assertEqual(codec.decoder, "24k_4codebooks")


class FakeLongCatCodec:
    def __init__(self) -> None:
        self.features = torch.ones((1, 2, 3))
        self.acoustic_codes: torch.Tensor | None = None
        self.decoder: str | None = None

    def acoustic_codes_to_features(
        self,
        acoustic_codes: torch.Tensor,
        *,
        decoder: str,
    ) -> torch.Tensor:
        self.acoustic_codes = acoustic_codes
        self.decoder = decoder
        return self.features


if __name__ == "__main__":
    unittest.main()
