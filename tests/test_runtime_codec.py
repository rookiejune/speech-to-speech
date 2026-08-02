from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from anydataset.types import AudioView
from anytrain.codec import AcousticLayout

from speech_to_speech.runtime import (
    AudioRepresentation,
    AudioSequenceLayout,
    Config,
    Runtime,
    runtime_for_sequence_layout,
)
from speech_to_speech.runtime.audio_tokenizer import (
    BiCodecAudioTokenizer,
    FlattenedAudioTokenizer,
)
from speech_to_speech.runtime.codec import load_codec
from speech_to_speech.runtime.types import (
    frame_codec,
    supports_acoustic,
    supports_structured,
)


class RuntimeCodecTest(unittest.TestCase):
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
        config = Config(
            codec="stable_codec",
            audio_representation=AudioRepresentation.FULL_CODEC_SEQUENCE,
        )

        self.assertIs(config.audio_view, AudioView.STABLE)

    def test_frame_aligned_structured_codec_uses_frame_full_sequence(self) -> None:
        runtime = _runtime("longcat", _codec(AcousticLayout.FRAME_ALIGNED))

        self.assertFalse(runtime.structured_full_sequence)
        self.assertIsInstance(runtime.audio_tokenizer, FlattenedAudioTokenizer)

    def test_fixed_length_structured_codec_uses_structured_full_sequence(self) -> None:
        runtime = _runtime("bicodec", _codec(AcousticLayout.FIXED_LENGTH))

        self.assertTrue(runtime.structured_full_sequence)
        self.assertIsInstance(runtime.audio_tokenizer, BiCodecAudioTokenizer)

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
        config = Config(
            codec="bicodec",
            audio_representation=AudioRepresentation.FULL_CODEC_SEQUENCE,
        )

        runtime = runtime_for_sequence_layout(
            config,
            AudioSequenceLayout.FLATTENED,
        )

        self.assertIs(runtime.audio_sequence_layout, AudioSequenceLayout.FLATTENED)
        self.assertIs(runtime.audio_representation, AudioRepresentation.FULL_CODEC_SEQUENCE)
        self.assertIs(runtime.audio_view, AudioView.BICODEC)

    def test_runtime_for_sequence_layout_preserves_config(self) -> None:
        config = Config(
            codec="stable_codec",
            audio_representation=AudioRepresentation.FULL_CODEC_SEQUENCE,
        )

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
        Config(
            codec=codec_name,
            audio_representation=AudioRepresentation.FULL_CODEC_SEQUENCE,
        )
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
