from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from anydataset.types import AudioView
from anytrain.codec import AcousticLayout

from speech_to_speech.runtime import (
    AudioSequenceLayout,
    Config,
    Runtime,
    migrate_config_fields,
    runtime_for_sequence_layout,
)
from speech_to_speech.runtime.audio_tokenizer import (
    BiCodecAudioTokenizer,
    FlattenedAudioTokenizer,
)
from speech_to_speech.runtime.codec import load_codec
from speech_to_speech.runtime.codec_contract import (
    frame_codec,
    supports_acoustic,
    supports_structured,
)


class RuntimeCodecTest(unittest.TestCase):
    def test_legacy_generator_artifact_config_field_is_migrated_explicitly(self) -> None:
        fields = {"semantic_codec_artifact": "/tmp/legacy-generator"}

        with self.assertWarns(FutureWarning):
            migrate_config_fields(fields)

        self.assertEqual(
            fields,
            {"acoustic_generator_artifact": "/tmp/legacy-generator"},
        )

    def test_legacy_generator_artifact_config_field_rejects_conflicts(self) -> None:
        fields = {
            "acoustic_generator_artifact": "/tmp/current-generator",
            "semantic_codec_artifact": "/tmp/legacy-generator",
        }

        with self.assertRaisesRegex(ValueError, "conflicting"):
            migrate_config_fields(fields)

    def test_acoustic_generator_artifact_file_digest_is_cached_per_runtime(self) -> None:
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "semantic-codec.pt"
            artifact.write_bytes(b"semantic-codec-v1")
            runtime = Runtime(
                Config(
                    codec="longcat",
                    acoustic_generator_artifact=str(artifact),
                )
            )

            first = runtime.acoustic_generator_artifact_sha256
            artifact.write_bytes(b"semantic-codec-v2")

            self.assertEqual(
                first,
                hashlib.sha256(b"semantic-codec-v1").hexdigest(),
            )
            self.assertEqual(runtime.acoustic_generator_artifact_sha256, first)
            self.assertNotEqual(
                Runtime(runtime.config).acoustic_generator_artifact_sha256,
                first,
            )

    def test_acoustic_generator_artifact_directory_digest_uses_relative_content(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first" / "artifact"
            second = root / "second" / "artifact"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "nested").mkdir()
            (second / "nested").mkdir()
            (first / "weights.bin").write_bytes(b"weights")
            (first / "nested" / "config.json").write_text(
                '{"version": 1}',
                encoding="utf-8",
            )
            (second / "nested" / "config.json").write_text(
                '{"version": 1}',
                encoding="utf-8",
            )
            (second / "weights.bin").write_bytes(b"weights")

            first_digest = Runtime(
                Config(codec="longcat", acoustic_generator_artifact=str(first))
            ).acoustic_generator_artifact_sha256
            second_digest = Runtime(
                Config(codec="longcat", acoustic_generator_artifact=str(second))
            ).acoustic_generator_artifact_sha256

            self.assertEqual(first_digest, second_digest)
            (second / "nested" / "config.json").rename(second / "config.json")
            self.assertNotEqual(
                Runtime(
                    Config(codec="longcat", acoustic_generator_artifact=str(second))
                ).acoustic_generator_artifact_sha256,
                first_digest,
            )

    def test_runtime_without_acoustic_generator_artifact_has_no_digest(self) -> None:
        self.assertIsNone(Runtime(Config(codec="longcat")).acoustic_generator_artifact_sha256)

    def test_load_codec_longcat_uses_anytrain_backend(self) -> None:
        backend = SimpleNamespace(name="longcat")

        with patch(
            "speech_to_speech.runtime.codec.load_semantic_acoustic",
            return_value=backend,
        ) as load_codec_backend:
            codec = load_codec("longcat", device="cuda")

        self.assertIs(codec, backend)
        load_codec_backend.assert_called_once_with("longcat", device="cuda")

    def test_load_codec_stable_uses_frame_backend(self) -> None:
        backend = _StableSource()

        with patch(
            "speech_to_speech.runtime.codec.load_frame",
            return_value=backend,
        ) as load_codec_backend:
            codec = load_codec("stable_codec", device="cuda")

        self.assertEqual(codec.name, "stable_codec")
        self.assertEqual(codec.sample_rate, 16_000)
        self.assertEqual(codec.frame_rate, 25.0)
        self.assertEqual(codec.codebook_sizes, (46_656,))
        self.assertEqual(codec.semantic_feature_dim, 1)
        self.assertEqual(codec.fsq_levels, ((6, 6, 6, 6, 6, 6),))
        codes = codec.encode(torch.zeros(1, 1, 8), 16_000)
        torch.testing.assert_close(codes, backend.codes)
        waveform = codec.decode(codes)
        torch.testing.assert_close(waveform, backend.waveform)
        load_codec_backend.assert_called_once_with("stable_codec", device="cuda")

    def test_load_codec_unicodec_exposes_frame_capability(self) -> None:
        backend = _UnifiedSource()

        with patch(
            "speech_to_speech.runtime.codec.load_frame",
            return_value=backend,
        ) as load_codec_backend:
            codec = load_codec("unicodec", device="cuda")

        self.assertEqual(codec.name, "unicodec")
        self.assertEqual(codec.sample_rate, 24_000)
        self.assertEqual(codec.frame_rate, 75.0)
        self.assertEqual(codec.codebook_sizes, (4,))
        self.assertIs(frame_codec(codec), codec)
        codes = codec.encode(torch.zeros(1, 1, 8), 24_000)
        torch.testing.assert_close(codes, backend.codes)
        waveform = codec.decode(codes)
        torch.testing.assert_close(waveform, backend.waveform)
        load_codec_backend.assert_called_once_with("unicodec", device="cuda")

    def test_stable_runtime_uses_stable_audio_view(self) -> None:
        config = Config(codec="stable_codec")

        self.assertIs(config.audio_view, AudioView.STABLE)

    def test_frame_aligned_structured_codec_uses_frame_full_sequence(self) -> None:
        runtime = _runtime("longcat", _codec(AcousticLayout.FRAME_ALIGNED))

        self.assertFalse(runtime.structured_full_sequence)
        self.assertIsInstance(runtime.audio_tokenizer, FlattenedAudioTokenizer)

    def test_fixed_length_structured_codec_uses_structured_full_sequence(self) -> None:
        runtime = _runtime("bicodec", _codec(AcousticLayout.FIXED_LENGTH))

        self.assertTrue(runtime.structured_full_sequence)
        self.assertIsInstance(runtime.audio_tokenizer, BiCodecAudioTokenizer)

    def test_bicodec_runtime_nests_explicit_semantic_tokenizer(self) -> None:
        config = Config(
            codec="bicodec",
            audio_tokenizer="/tmp/bicodec-semantic-bpe",
        )
        runtime = Runtime(
            config,
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
        )
        runtime.__dict__["codec"] = _codec(AcousticLayout.FIXED_LENGTH)
        semantic_tokenizer = Mock()
        outer_tokenizer = Mock()

        with (
            patch(
                "speech_to_speech.runtime.core.audio_tokenizer",
                return_value=semantic_tokenizer,
            ) as load_tokenizer,
            patch(
                "speech_to_speech.runtime.core.BiCodecAudioTokenizer",
                return_value=outer_tokenizer,
            ) as build_tokenizer,
        ):
            loaded = runtime.audio_tokenizer

        self.assertIs(loaded, outer_tokenizer)
        load_tokenizer.assert_called_once_with("/tmp/bicodec-semantic-bpe")
        build_tokenizer.assert_called_once_with(
            semantic_codebook_size=8,
            global_codebook_sizes=(5, 7),
            global_unit_length=3,
            semantic_tokenizer=semantic_tokenizer,
        )

    def test_frame_codec_rejects_invalid_codebook_sizes(self) -> None:
        cases = (
            ([], TypeError, "tuple"),
            ((), ValueError, "non-empty"),
            ((True,), TypeError, "integer"),
            ((1.5,), TypeError, "integer"),
            ((0,), ValueError, "positive"),
        )
        for sizes, error, message in cases:
            with self.subTest(sizes=sizes), self.assertRaisesRegex(error, message):
                frame_codec(_codec(AcousticLayout.FRAME_ALIGNED, codebook_sizes=sizes))

    def test_frame_codec_rejects_invalid_rate_metadata(self) -> None:
        cases = (
            ({"sample_rate": True}, TypeError, "integer"),
            ({"sample_rate": 0}, ValueError, "positive"),
            ({"frame_rate": True}, TypeError, "number"),
            ({"frame_rate": float("nan")}, ValueError, "finite"),
            ({"frame_rate": 0.0}, ValueError, "positive"),
        )
        for overrides, error, message in cases:
            codec = _codec(AcousticLayout.FRAME_ALIGNED, **overrides)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(error, message):
                frame_codec(codec)

    def test_acoustic_capability_rejects_invalid_metadata(self) -> None:
        cases = (
            ({"sample_rate": True}, TypeError, "integer"),
            ({"frame_rate": float("inf")}, ValueError, "finite"),
            ({"acoustic_codebook_sizes": []}, TypeError, "tuple"),
            ({"acoustic_codebook_sizes": ()}, ValueError, "non-empty"),
            ({"acoustic_codebook_sizes": (True,)}, TypeError, "integer"),
            ({"acoustic_feature_dim": True}, TypeError, "integer"),
            ({"acoustic_feature_dim": 0}, ValueError, "positive"),
        )
        for overrides, error, message in cases:
            codec = _codec(AcousticLayout.FRAME_ALIGNED, **overrides)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(error, message):
                supports_acoustic(codec)

    def test_structured_capability_rejects_invalid_metadata(self) -> None:
        cases = (
            ({"semantic_codebook_sizes": []}, TypeError, "tuple"),
            ({"semantic_codebook_sizes": ()}, ValueError, "non-empty"),
            ({"semantic_codebook_sizes": (True,)}, TypeError, "integer"),
            ({"acoustic_codebook_sizes": (1.5,)}, TypeError, "integer"),
            ({"acoustic_layout": "fixed_length"}, TypeError, "AcousticLayout"),
            (
                {"acoustic_layout": AcousticLayout.FIXED_LENGTH, "acoustic_unit_length": None},
                TypeError,
                "integer",
            ),
            (
                {"acoustic_layout": AcousticLayout.FIXED_LENGTH, "acoustic_unit_length": True},
                TypeError,
                "integer",
            ),
            (
                {"acoustic_layout": AcousticLayout.FIXED_LENGTH, "acoustic_unit_length": 0},
                ValueError,
                "positive",
            ),
            ({"acoustic_unit_length": 2}, ValueError, "must be None"),
        )
        for overrides, error, message in cases:
            codec = _codec(AcousticLayout.FRAME_ALIGNED, **overrides)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(error, message):
                supports_structured(codec)


class RuntimeAudioSequenceLayoutTest(unittest.TestCase):
    def test_runtime_defaults_to_semantic_layout(self) -> None:
        runtime = Runtime(Config())

        self.assertIs(runtime.audio_sequence_layout, AudioSequenceLayout.SEMANTIC)

    def test_runtime_requires_sequence_layout_enum(self) -> None:
        with self.assertRaisesRegex(TypeError, "AudioSequenceLayout"):
            Runtime(Config(), audio_sequence_layout="semantic")

    def test_runtime_for_sequence_layout_sets_layout(self) -> None:
        config = Config(codec="bicodec")

        runtime = runtime_for_sequence_layout(
            config,
            AudioSequenceLayout.FLATTENED,
        )

        self.assertIs(runtime.audio_sequence_layout, AudioSequenceLayout.FLATTENED)
        self.assertIs(runtime.audio_view, AudioView.BICODEC)

    def test_bicodec_rejects_split_semantic_route(self) -> None:
        with self.assertRaisesRegex(ValueError, "self-describing structured sequence"):
            Runtime(
                Config(codec="bicodec"),
                audio_sequence_layout=AudioSequenceLayout.SEMANTIC,
            )

        with self.assertRaisesRegex(ValueError, "cannot use.*acoustic_generator_artifact"):
            Runtime(
                Config(
                    codec="bicodec",
                    acoustic_generator_artifact="/tmp/bicodec-semantic",
                ),
                audio_sequence_layout=AudioSequenceLayout.FLATTENED,
            )

    def test_runtime_for_sequence_layout_preserves_config(self) -> None:
        config = Config(codec="stable_codec")

        runtime = runtime_for_sequence_layout(
            config,
            AudioSequenceLayout.FLATTENED,
        )

        self.assertIs(runtime.config, config)
        self.assertIs(runtime.audio_view, AudioView.STABLE)


class _StableSource:
    sample_rate = 16_000
    frame_rate = 25.0
    codebook_sizes = (46_656,)

    def __init__(self) -> None:
        self.codes = torch.tensor([[[1], [2]]], dtype=torch.long)
        self.waveform = torch.zeros(1, 1, 8)

    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        del audio, sample_rate
        return self.codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        del codes
        return self.waveform


class _UnifiedSource:
    sample_rate = 24_000
    frame_rate = 75.0
    codebook_sizes = (4,)

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.model = SimpleNamespace()
        self.codes = torch.tensor([[[1], [2]]], dtype=torch.long)
        self.waveform = torch.zeros(1, 1, 8)
        self.features = torch.arange(12, dtype=torch.float32).view(4, 3)

    def codes_to_features(self, codes: torch.Tensor) -> torch.Tensor:
        return self.features[codes.squeeze(-1)]

    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        del audio, sample_rate
        return self.codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        del codes
        return self.waveform


def _runtime(codec_name: str, codec: object) -> Runtime:
    runtime = Runtime(
        Config(codec=codec_name),
        audio_sequence_layout=AudioSequenceLayout.FLATTENED,
    )
    runtime.__dict__["codec"] = codec
    return runtime


def _codec(layout: AcousticLayout, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "sample_rate": 16_000,
        "frame_rate": 50.0,
        "codebook_sizes": (8, 5, 7),
        "semantic_codebook": torch.zeros(8, 4),
        "semantic_codebook_sizes": (8,),
        "acoustic_codebook_sizes": (5, 7),
        "acoustic_feature_dim": 4,
        "acoustic_layout": layout,
        "acoustic_unit_length": (
            3 if layout is AcousticLayout.FIXED_LENGTH else None
        ),
        "encode": Mock(),
        "decode": Mock(),
        "tokenize": Mock(),
        "detokenize": Mock(),
        "acoustic_codes_to_features": Mock(),
        "decode_features": Mock(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


if __name__ == "__main__":
    unittest.main()
