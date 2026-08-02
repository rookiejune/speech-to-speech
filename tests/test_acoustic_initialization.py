from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import cast

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodec
from semantic_acoustic_codec.config import DecoderConfig, Route
from semantic_acoustic_codec.runtime import (
    SemanticSupportConfig,
    build_support,
)
from semantic_acoustic_codec.runtime.artifact import save_artifact

from speech_to_speech.model.acoustic.initialization import load_acoustic_initialization


class _Backend:
    name = "fake"
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_frame_rate = 50.0
    semantic_codebook = torch.randn(8, 4)
    acoustic_feature_dim = 4
    acoustic_codebook_sizes = (5, 7)
    acoustic_layout = AcousticLayout.FRAME_ALIGNED
    acoustic_unit_length = None


class AcousticInitializationTest(unittest.TestCase):
    def test_loads_generator_with_matching_backend_and_route(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _save(path)

            artifact = load_acoustic_initialization(
                path,
                codec=_Backend(),
                route=Route.FM,
                device=torch.device("cpu"),
            )

        self.assertIs(artifact.spec.route, Route.FM)
        self.assertEqual(artifact.spec.condition_dim, 6)

    def test_rejects_route_and_backend_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _save(path)

            with self.assertRaisesRegex(ValueError, "route"):
                load_acoustic_initialization(
                    path,
                    codec=_Backend(),
                    route=Route.RVQ,
                    device=torch.device("cpu"),
                )

            backend = _Backend()
            backend.acoustic_feature_dim = 5
            with self.assertRaisesRegex(ValueError, "acoustic_feature_dim"):
                load_acoustic_initialization(
                    path,
                    codec=backend,
                    route=Route.FM,
                    device=torch.device("cpu"),
                )

    def test_semantic_codebook_is_not_a_joint_generator_constraint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            _save(path)
            backend = _Backend()
            backend.semantic_codebook = torch.randn(13, 9)

            artifact = load_acoustic_initialization(
                path,
                codec=backend,
                route=Route.FM,
                device=torch.device("cpu"),
            )

        self.assertEqual(artifact.spec.semantic_vocab_size, 8)
        self.assertEqual(artifact.spec.semantic_embedding_dim, 4)


def _save(path: Path) -> None:
    backend = _Backend()
    config = SemanticSupportConfig(
        route=Route.FM,
        condition_dim=6,
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
    )
    support = build_support(
        config,
        semantic_codebook=backend.semantic_codebook,
        acoustic_feature_dim=backend.acoustic_feature_dim,
        acoustic_codebook_sizes=backend.acoustic_codebook_sizes,
        acoustic_layout=backend.acoustic_layout,
        acoustic_unit_length=backend.acoustic_unit_length,
    )
    save_artifact(path, support, config, backend=cast(SemanticAcousticCodec, backend))


if __name__ == "__main__":
    unittest.main()
