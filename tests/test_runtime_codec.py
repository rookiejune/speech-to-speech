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
    AudioInputConfig,
    AudioOutputConfig,
    AudioSequenceLayout,
    Config,
    InputAudioConfig,
    Runtime,
    migrate_config_fields,
    runtime_for_sequence_layout,
)
from speech_to_speech.runtime.audio_tokenizer import (
    BiCodecAudioTokenizer,
    FlattenedAudioTokenizer,
)
from speech_to_speech.runtime.codec import (
    has_audio_tokenizer_loader,
    has_codec_loader,
    load_codec,
)
from speech_to_speech.runtime.codec_contract import (
    frame_codec,
    frame_tokenizer,
    supports_acoustic,
    supports_global,
    supports_structured,
)


class RuntimeCodecTest(unittest.TestCase):
    def test_codec_loader_capability_matches_runtime_dispatch(self) -> None:
        for name in ("glm4", "longcat", "bicodec", "stable_codec", "unicodec"):
            with self.subTest(name=name):
                self.assertTrue(has_audio_tokenizer_loader(name))
        for name in ("dac",):
            with self.subTest(name=name):
                self.assertFalse(has_audio_tokenizer_loader(name))
        self.assertEqual(
            has_codec_loader("glm4"),
            has_audio_tokenizer_loader("glm4"),
        )

    def test_legacy_generator_artifact_config_field_is_migrated_explicitly(self) -> None:
        fields = {"semantic_codec_artifact": "/tmp/legacy-generator"}

        with self.assertWarns(FutureWarning):
            migrate_config_fields(fields)

        self.assertEqual(
            fields,
            {
                "audio_output": {
                    "acoustic_generator_artifact": "/tmp/legacy-generator"
                }
            },
        )

    def test_legacy_generator_artifact_config_field_rejects_conflicts(self) -> None:
        fields = {
            "acoustic_generator_artifact": "/tmp/current-generator",
            "semantic_codec_artifact": "/tmp/legacy-generator",
        }

        with self.assertRaisesRegex(ValueError, "conflicting"):
            migrate_config_fields(fields)

    def test_canonical_and_legacy_audio_configs_resolve_identically(self) -> None:
        with self.assertWarns(FutureWarning):
            legacy = Config(
                codec="bicodec",
                input_audio=InputAudioConfig(
                    codec="glm4",
                    vocab_size=16_384,
                    frame_rate=12.5,
                ),
            )
        canonical = Config(
            audio_input=AudioInputConfig(
                tokenizer="glm4",
            ),
            audio_output=AudioOutputConfig(
                tokenizer="bicodec",
                detokenizer="bicodec",
            ),
        )

        legacy_runtime = Runtime(
            legacy,
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
        )
        canonical_runtime = Runtime(
            canonical,
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
        )

        self.assertEqual(legacy_runtime.codec_name, canonical_runtime.codec_name)
        self.assertEqual(
            legacy_runtime.input_codec_name,
            canonical_runtime.input_codec_name,
        )
        self.assertIs(legacy_runtime.audio_view, canonical_runtime.audio_view)
        self.assertIs(
            legacy_runtime.input_audio_view,
            canonical_runtime.input_audio_view,
        )
        self.assertEqual(
            legacy_runtime.input_codec_frame_rate,
            canonical_runtime.input_codec_frame_rate,
        )

    def test_audio_view_is_derived_or_strictly_validated(self) -> None:
        self.assertIs(
            AudioOutputConfig(tokenizer="stable_codec").audio_view,
            AudioView.STABLE,
        )
        self.assertIs(
            AudioInputConfig(tokenizer="glm4").audio_view,
            AudioView.GLM4,
        )
        with self.assertRaisesRegex(ValueError, "must match tokenizer"):
            AudioOutputConfig(tokenizer="bicodec", view=AudioView.GLM4)
        with self.assertRaisesRegex(ValueError, "must match tokenizer"):
            AudioInputConfig(tokenizer="glm4", view=AudioView.BICODEC)

    def test_glm4_input_uses_preset_metadata_without_config_overrides(self) -> None:
        input_ = AudioInputConfig(tokenizer="glm4")
        runtime = Runtime(
            Config(
                audio_input=input_,
                audio_output=AudioOutputConfig(
                    tokenizer="bicodec",
                    detokenizer="bicodec",
                ),
            ),
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
        )

        self.assertIs(runtime.config.audio_input, input_)
        self.assertIs(runtime.input_audio_view, AudioView.GLM4)
        self.assertEqual(runtime.input_codec_frame_rate, 12.5)
        self.assertEqual(runtime.input_audio_tokenizer.vocab_size, 16_384)

    def test_glm4_codes_only_output_does_not_require_a_detokenizer(self) -> None:
        output = AudioOutputConfig(tokenizer="glm4", detokenizer=None)

        self.assertEqual(output.tokenizer, "glm4")
        self.assertIsNone(output.detokenizer)

    def test_bicodec_codes_only_output_does_not_load_a_detokenizer(self) -> None:
        runtime = Runtime(
            Config(
                audio_output=AudioOutputConfig(
                    tokenizer="bicodec",
                    detokenizer=None,
                )
            ),
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
        )

        self.assertIsNone(runtime.output_audio_detokenizer_name)
        self.assertIsNone(runtime.output_audio_detokenizer)
        self.assertNotIn("output_audio_tokenizer_backend", runtime.__dict__)

    def test_same_output_tokenizer_and_detokenizer_load_one_backend(self) -> None:
        backend = SimpleNamespace(name="bicodec")
        with (
            patch(
                "speech_to_speech.runtime.core.load_audio_tokenizer_backend",
                return_value=backend,
            ) as load_tokenizer,
            patch(
                "speech_to_speech.runtime.core.load_audio_detokenizer_backend"
            ) as load_detokenizer,
        ):
            runtime = Runtime(
                Config(
                    audio_output=AudioOutputConfig(
                        tokenizer="bicodec",
                        detokenizer="bicodec",
                    )
                ),
                audio_sequence_layout=AudioSequenceLayout.FLATTENED,
            )

            self.assertIs(runtime.output_audio_detokenizer, backend)
            self.assertIs(runtime.output_audio_tokenizer_backend, backend)

        load_tokenizer.assert_called_once_with("bicodec", None)
        load_detokenizer.assert_not_called()

    def test_explicit_same_token_space_input_collapses_to_shared(self) -> None:
        runtime = Runtime(
            Config(
                audio_input=AudioInputConfig(tokenizer="longcat"),
                audio_output=AudioOutputConfig(tokenizer="longcat"),
            )
        )

        self.assertIsNone(runtime.config.audio_input)
        self.assertFalse(runtime.input_audio_decoupled)
        self.assertEqual(runtime.input_codec_name, runtime.output_codec_name)
        self.assertIs(runtime.input_audio_view, runtime.output_audio_view)

    def test_distinct_input_backend_uses_registered_static_metadata(self) -> None:
        runtime = Runtime(
            Config(
                audio_input=AudioInputConfig(tokenizer="unicodec"),
                audio_output=AudioOutputConfig(tokenizer="longcat"),
            )
        )
        self.assertTrue(has_audio_tokenizer_loader("unicodec"))
        self.assertTrue(has_audio_tokenizer_loader("glm4"))
        self.assertEqual(runtime.input_codec_frame_rate, 75.0)
        self.assertEqual(runtime.input_audio_tokenizer.vocab_size, 16_384)
        self.assertNotIn("input_audio_tokenizer_backend", runtime.__dict__)

    def test_legacy_input_metadata_must_match_registered_preset(self) -> None:
        with self.assertWarns(FutureWarning), self.assertRaisesRegex(
            ValueError,
            "frame_rate does not match",
        ):
            AudioInputConfig(tokenizer="unicodec", frame_rate=50.0)

        with self.assertWarns(FutureWarning), self.assertRaisesRegex(
            ValueError,
            "vocab_size does not match",
        ):
            AudioInputConfig(tokenizer="unicodec", vocab_size=5)

    def test_shared_backend_frame_rate_resolves_to_backend_value(self) -> None:
        with self.assertWarns(FutureWarning):
            runtime = Runtime(
                Config(
                    audio_input=AudioInputConfig(
                        tokenizer="longcat",
                        bpe="/tmp/input-bpe",
                        frame_rate=16_000 / 960,
                    ),
                    audio_output=AudioOutputConfig(tokenizer="longcat"),
                )
            )
        self.assertEqual(runtime.input_codec_frame_rate, 16_000 / 960)

    def test_direct_config_rejects_canonical_legacy_conflicts(self) -> None:
        same = Config(
            codec="bicodec",
            audio_output=AudioOutputConfig(tokenizer="bicodec"),
        )
        self.assertEqual(same.audio_output.tokenizer, "bicodec")
        with self.assertRaisesRegex(ValueError, "conflicting audio_output.tokenizer"):
            Config(
                codec="longcat",
                audio_output=AudioOutputConfig(tokenizer="bicodec"),
            )

    def test_equal_codec_and_tokenizer_aliases_do_not_create_a_bpe(self) -> None:
        with self.assertWarns(FutureWarning):
            output = AudioOutputConfig(
                codec="bicodec",
                tokenizer="bicodec",
                detokenizer="bicodec",
            )
        with self.assertWarns(FutureWarning):
            input_ = AudioInputConfig(
                codec="glm4",
                tokenizer="glm4",
                vocab_size=16_384,
                frame_rate=12.5,
            )

        self.assertEqual(output.tokenizer, "bicodec")
        self.assertIsNone(output.bpe)
        self.assertEqual(input_.tokenizer, "glm4")
        self.assertIsNone(input_.bpe)

    def test_mapping_migration_accepts_equal_values_and_rejects_conflicts(self) -> None:
        fields = {
            "codec": "bicodec",
            "audio_output": {"codec": "bicodec"},
        }
        with self.assertWarns(FutureWarning):
            migrate_config_fields(fields)
        self.assertEqual(
            fields,
            {
                "audio_output": {
                    "tokenizer": "bicodec",
                    "detokenizer": "bicodec",
                }
            },
        )

        conflicting = {
            "codec": "longcat",
            "audio_output": {"codec": "bicodec"},
        }
        with self.assertRaisesRegex(ValueError, "conflicting audio_output.tokenizer"):
            migrate_config_fields(conflicting)

    def test_nested_codec_tokenizer_schema_migrates_to_tokenizer_bpe(self) -> None:
        fields = {
            "audio_input": {
                "codec": "glm4",
                "tokenizer": "/tmp/glm4-bpe",
                "vocab_size": 16,
                "frame_rate": 12.5,
            },
            "audio_output": {
                "codec": "bicodec",
                "tokenizer": "/tmp/bicodec-bpe",
            },
        }

        with self.assertWarns(FutureWarning):
            migrate_config_fields(fields)

        self.assertEqual(
            fields,
            {
                "audio_input": {
                    "tokenizer": "glm4",
                    "bpe": "/tmp/glm4-bpe",
                    "vocab_size": 16,
                    "frame_rate": 12.5,
                },
                "audio_output": {
                    "tokenizer": "bicodec",
                    "detokenizer": "bicodec",
                    "bpe": "/tmp/bicodec-bpe",
                },
            },
        )

    def test_nested_equal_codec_and_tokenizer_migrate_to_one_backend(self) -> None:
        fields = {
            "audio_input": {
                "codec": "glm4",
                "tokenizer": "glm4",
                "vocab_size": 16,
                "frame_rate": 12.5,
            },
            "audio_output": {
                "codec": "bicodec",
                "tokenizer": "bicodec",
            },
        }

        with self.assertWarns(FutureWarning):
            migrate_config_fields(fields)

        self.assertEqual(
            fields,
            {
                "audio_input": {
                    "tokenizer": "glm4",
                    "vocab_size": 16,
                    "frame_rate": 12.5,
                },
                "audio_output": {
                    "tokenizer": "bicodec",
                    "detokenizer": "bicodec",
                },
            },
        )

    def test_mapping_migration_treats_legacy_null_input_leaves_as_explicit(
        self,
    ) -> None:
        for name, canonical, conflict_name in (
            ("tokenizer", "/tmp/glm4-tokenizer", "bpe"),
            ("frame_rate", 12.5, "frame_rate"),
        ):
            fields = {
                "audio_input": {"codec": "glm4", name: canonical},
                "input_audio": {"codec": "glm4", name: None},
            }

            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError,
                f"conflicting audio_input.{conflict_name}",
            ):
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
            "speech_to_speech.runtime.codec.load_audio_tokenizer",
            return_value=SimpleNamespace(backend=backend),
        ) as load_codec_backend:
            codec = load_codec("longcat", device="cuda")

        self.assertIs(codec, backend)
        load_codec_backend.assert_called_once_with("longcat", device="cuda")

    def test_load_codec_bicodec_uses_anytrain_global_backend(self) -> None:
        backend = SimpleNamespace(name="bicodec")

        with patch(
            "speech_to_speech.runtime.codec.load_audio_tokenizer",
            return_value=SimpleNamespace(backend=backend),
        ) as load_codec_backend:
            codec = load_codec("bicodec", device="cuda")

        self.assertIs(codec, backend)
        load_codec_backend.assert_called_once_with("bicodec", device="cuda")

    def test_load_codec_stable_uses_frame_backend(self) -> None:
        backend = _StableSource()

        with patch(
            "speech_to_speech.runtime.codec.load_audio_tokenizer",
            return_value=SimpleNamespace(backend=backend),
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
            "speech_to_speech.runtime.codec.load_audio_tokenizer",
            return_value=SimpleNamespace(backend=backend),
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
        runtime = _runtime("longcat", _acoustic_codec())

        self.assertFalse(runtime.structured_full_sequence)
        self.assertIsInstance(runtime.audio_tokenizer, FlattenedAudioTokenizer)

    def test_global_codec_uses_structured_full_sequence(self) -> None:
        runtime = _runtime("bicodec", _global_codec())

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
        runtime.__dict__["output_audio_tokenizer_backend"] = _global_codec()
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
            semantic_codebook_size=8192,
            global_codebook_sizes=(4096,),
            global_unit_length=32,
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
                frame_codec(_acoustic_codec(codebook_sizes=sizes))

    def test_frame_codec_rejects_invalid_rate_metadata(self) -> None:
        cases = (
            ({"sample_rate": True}, TypeError, "integer"),
            ({"sample_rate": 0}, ValueError, "positive"),
            ({"frame_rate": True}, TypeError, "number"),
            ({"frame_rate": float("nan")}, ValueError, "finite"),
            ({"frame_rate": 0.0}, ValueError, "positive"),
        )
        for overrides, error, message in cases:
            codec = _acoustic_codec(**overrides)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(error, message):
                frame_codec(codec)

    def test_frame_tokenizer_accepts_tokenizer_only_backend(self) -> None:
        backend = SimpleNamespace(
            sample_rate=16_000,
            frame_rate=12.5,
            codebook_sizes=(16_384,),
            encode=Mock(),
        )

        self.assertIs(frame_tokenizer(backend), backend)
        with self.assertRaisesRegex(TypeError, "encoding and decoding"):
            frame_codec(backend)

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
            codec = _acoustic_codec(**overrides)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(error, message):
                supports_acoustic(codec)

    def test_structured_capability_rejects_invalid_metadata(self) -> None:
        cases = (
            ({"semantic_codebook_sizes": []}, TypeError, "tuple"),
            ({"semantic_codebook_sizes": ()}, ValueError, "non-empty"),
            ({"semantic_codebook_sizes": (True,)}, TypeError, "integer"),
            ({"acoustic_codebook_sizes": (1.5,)}, TypeError, "integer"),
            ({"acoustic_layout": "fixed_length"}, TypeError, "AcousticLayout"),
            ({"acoustic_unit_length": 2}, ValueError, "must be None"),
        )
        for overrides, error, message in cases:
            codec = _acoustic_codec(**overrides)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(error, message):
                supports_structured(codec)

    def test_global_capability_rejects_invalid_metadata(self) -> None:
        cases = (
            ({"semantic_codebook_sizes": []}, TypeError, "tuple"),
            ({"global_codebook_sizes": []}, TypeError, "tuple"),
            ({"global_codebook_sizes": ()}, ValueError, "non-empty"),
            ({"global_codebook_sizes": (True,)}, TypeError, "integer"),
            ({"global_codebook_sizes": (1.5,)}, TypeError, "integer"),
            ({"global_feature_dim": True}, TypeError, "integer"),
            ({"global_feature_dim": 0}, ValueError, "positive"),
            ({"global_unit_length": None}, TypeError, "integer"),
            ({"global_unit_length": True}, TypeError, "integer"),
            ({"global_unit_length": 0}, ValueError, "positive"),
        )
        for overrides, error, message in cases:
            codec = _global_codec(**overrides)
            with self.subTest(overrides=overrides), self.assertRaisesRegex(error, message):
                supports_global(codec)


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
    runtime.__dict__["output_audio_tokenizer_backend"] = codec
    return runtime


def _acoustic_codec(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "sample_rate": 16_000,
        "frame_rate": 50.0,
        "codebook_sizes": (8, 5, 7),
        "semantic_codebook": torch.zeros(8, 4),
        "semantic_codebook_sizes": (8,),
        "acoustic_codebook_sizes": (5, 7),
        "acoustic_feature_dim": 4,
        "acoustic_layout": AcousticLayout.FRAME_ALIGNED,
        "acoustic_unit_length": None,
        "encode": Mock(),
        "decode": Mock(),
        "tokenize": Mock(),
        "detokenize": Mock(),
        "acoustic_codes_to_features": Mock(),
        "decode_features": Mock(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _global_codec(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "sample_rate": 16_000,
        "frame_rate": 50.0,
        "semantic_codebook": torch.zeros(8, 4),
        "semantic_codebook_sizes": (8,),
        "global_codebook_sizes": (5, 7),
        "global_feature_dim": 4,
        "global_unit_length": 3,
        "tokenize": Mock(),
        "detokenize": Mock(),
        "global_codes_to_features": Mock(),
        "decode_features": Mock(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


if __name__ == "__main__":
    unittest.main()
