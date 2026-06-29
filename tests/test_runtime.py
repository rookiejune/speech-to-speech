from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import torch

from speech_to_speech.config import BPEConfig, ModelConfig
from speech_to_speech import runtime
from speech_to_speech.types import SpeechPair


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

    def test_prepare_longcat_tokenizer_rejects_dataset_cache_mismatch(self) -> None:
        pair = SpeechPair(
            source_ids=torch.tensor([0, 1, 2]),
            target_ids=torch.tensor([2, 1, 0]),
        )
        config = BPEConfig(vocab_size=16, max_token_length=4)

        with TemporaryDirectory() as tmpdir:
            runtime.prepare_longcat_tokenizer(
                [pair],
                datasets=(_fake_dataset_meta("store://first"),),
                config=config,
                cache_dir=tmpdir,
            )
            with self.assertRaisesRegex(ValueError, "dataset mismatch"):
                runtime.prepare_longcat_tokenizer(
                    [pair],
                    datasets=(_fake_dataset_meta("store://second"),),
                    config=config,
                    cache_dir=tmpdir,
                )

    def test_prepare_longcat_tokenizer_accepts_matching_dataset_cache(self) -> None:
        pair = SpeechPair(
            source_ids=torch.tensor([0, 1, 2]),
            target_ids=torch.tensor([2, 1, 0]),
        )
        config = BPEConfig(vocab_size=16, max_token_length=4)

        with TemporaryDirectory() as tmpdir:
            first = runtime.prepare_longcat_tokenizer(
                [pair],
                datasets=(_fake_dataset_meta("store://same"),),
                config=config,
                cache_dir=tmpdir,
            )
            runtime._LONGCAT_TOKENIZERS.pop(
                Path(tmpdir) / "longcat" / "vocab_16_minfreq_0_maxlen_4_codes_8192"
            )
            second = runtime.prepare_longcat_tokenizer(
                [pair],
                datasets=(_fake_dataset_meta("store://same"),),
                config=config,
                cache_dir=tmpdir,
            )

        self.assertEqual(first.vocab_size, second.vocab_size)

    def test_prepare_longcat_tokenizer_writes_datasets_metadata(self) -> None:
        pair = SpeechPair(
            source_ids=torch.tensor([0, 1, 2]),
            target_ids=torch.tensor([2, 1, 0]),
        )
        config = BPEConfig(vocab_size=16, max_token_length=4)
        datasets = (_fake_dataset_meta("store://same"),)

        with TemporaryDirectory() as tmpdir:
            runtime.prepare_longcat_tokenizer(
                [pair],
                datasets=datasets,
                config=config,
                cache_dir=tmpdir,
            )
            meta_path = (
                Path(tmpdir)
                / "longcat"
                / "vocab_16_minfreq_0_maxlen_4_codes_8192"
                / "meta.json"
            )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

        self.assertEqual(meta["datasets"], list(datasets))
        self.assertNotIn("dataset_meta", meta)


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


def _fake_dataset_meta(dataset: str) -> dict[str, object]:
    return {"source": "store", "path": dataset}


if __name__ == "__main__":
    unittest.main()
