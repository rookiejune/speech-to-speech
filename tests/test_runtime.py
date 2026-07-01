from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import torch
from anytrain.idspace import Modality

from speech_to_speech.config import BPEConfig, ModelConfig, QwenBackboneConfig
from speech_to_speech import runtime
from speech_to_speech.types import AudioBoundary, SpecialToken, SpeechPair
from helpers import MockTokenizer


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
                        backbone=QwenBackboneConfig(
                            model_name_or_path="Qwen/Qwen3-0.6B",
                            trust_remote_code=False,
                        ),
                    )
                )
                second = runtime.qwen3_tokenizer(
                    ModelConfig(
                        backbone=QwenBackboneConfig(
                            model_name_or_path="Qwen/Qwen3-0.6B",
                            trust_remote_code=False,
                        ),
                    )
                )
                third = runtime.qwen3_tokenizer(
                    ModelConfig(
                        backbone=QwenBackboneConfig(
                            model_name_or_path="Qwen/Qwen3-8B",
                            trust_remote_code=True,
                        ),
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

    def test_qwen3_longcat_idspace_uses_tokenizer_and_bpe_vocab(self) -> None:
        tokenizer = MockTokenizer()
        tokenizer.vocab_size = 16
        tokenizer.__len__ = lambda: 16

        space = runtime.qwen3_longcat_idspace(
            tokenizer=tokenizer,
            bpe_vocab_size=5,
        )

        self.assertEqual(space.modality_block(Modality.TEXT).start, 0)
        self.assertEqual(space.modality_block(Modality.TEXT).vocab_size, 16)
        self.assertEqual(space.special_token_id(AudioBoundary.BOA), 16)
        self.assertEqual(space.special_token_id(AudioBoundary.EOA), 17)
        self.assertEqual(space.modality_block(Modality.AUDIO).start, 18)
        self.assertEqual(space.modality_block(Modality.AUDIO).vocab_size, 5)
        self.assertEqual(
            space.special_token_id(SpecialToken.PAD),
            tokenizer.convert_tokens_to_ids(SpecialToken.PAD.value),
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
        self.assertEqual(meta["requested_vocab_size"], config.vocab_size)
        self.assertIsInstance(meta["actual_vocab_size"], int)
        self.assertGreater(meta["actual_vocab_size"], 0)
        self.assertNotEqual(meta["actual_vocab_size"], config.vocab_size)
        self.assertNotIn("vocab_size", meta)

    def test_prepare_longcat_tokenizer_backfills_actual_vocab_size(self) -> None:
        pair = SpeechPair(
            source_ids=torch.tensor([0, 1, 2]),
            target_ids=torch.tensor([2, 1, 0]),
        )
        config = BPEConfig(vocab_size=16, max_token_length=4)

        with TemporaryDirectory() as tmpdir:
            runtime.prepare_longcat_tokenizer([pair], config=config, cache_dir=tmpdir)
            path = Path(tmpdir) / "longcat" / "vocab_16_minfreq_0_maxlen_4_codes_8192"
            meta_path = path / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            legacy_meta = {
                "codec_name": meta["codec_name"],
                "vocab_size": meta["requested_vocab_size"],
                "min_frequency": meta["min_frequency"],
                "max_token_length": meta["max_token_length"],
                "codebook_sizes": meta["codebook_sizes"],
                "datasets": meta["datasets"],
            }
            meta_path.write_text(
                json.dumps(legacy_meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            runtime._LONGCAT_TOKENIZERS.pop(path)

            bpe = runtime.longcat_tokenizer(config, cache_dir=tmpdir)
            updated = json.loads(meta_path.read_text(encoding="utf-8"))

        self.assertEqual(updated["requested_vocab_size"], config.vocab_size)
        self.assertEqual(updated["actual_vocab_size"], bpe.vocab_size)
        self.assertNotIn("vocab_size", updated)


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
