from __future__ import annotations

import unittest
from types import SimpleNamespace
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
from anytrain.codec import AcousticLayout, SemanticAcousticCodes

from speech_to_speech.callback.logging.task_sample import (
    _log_target_audio,
    _request_metadata,
    _target_text,
)
from speech_to_speech.generation import Request, decode_reference_codes
from speech_to_speech.task import Task


class TaskSampleLoggingTest(unittest.TestCase):
    def test_single_sample_metadata_uses_default_role(self):
        sample = _sample()

        metadata = _request_metadata(
            3,
            sample,
            Request(prompt_ids=torch.tensor([1, 2]), task=Task.TTS),
        )

        self.assertEqual(metadata["source"]["role"], Role.DEFAULT.value)
        self.assertEqual(metadata["reference"]["role"], Role.DEFAULT.value)
        self.assertTrue(metadata["reference"]["structured"])

    def test_single_sample_text_target_uses_default_role(self):
        self.assertEqual(_target_text(_sample(), Task.ASR), "hello")

    def test_structured_target_audio_uses_detokenize(self):
        codec = _StructuredCodec()
        writer = Mock()
        datamodule = SimpleNamespace(
            runtime=SimpleNamespace(codec=codec, audio_view=AudioView.BICODEC)
        )

        _log_target_audio(writer, datamodule, _sample(), Task.TTS, "sample", 7)

        self.assertIsNotNone(codec.decoded)
        self.assertEqual(codec.decoded.semantic.shape, (1, 2, 1))
        self.assertEqual(codec.decoded.acoustic.shape, (1, 3, 2))
        writer.add_audio.assert_called_once()
        self.assertEqual(writer.add_audio.call_args.kwargs["sample_rate"], 16_000)

    def test_frame_reference_uses_complete_codec_decode(self):
        codec = _FrameCodec()
        codes = torch.tensor([[1, 2], [3, 4]])

        waveform = decode_reference_codes(codes, codec=codec)

        self.assertTrue(torch.equal(codec.decoded, codes.unsqueeze(0)))
        self.assertEqual(waveform.shape, (1, 2))


class _StructuredCodec:
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_codebook = torch.zeros(8, 4)
    semantic_codebook_sizes = (8,)
    acoustic_codebook_sizes = (5, 7)
    acoustic_feature_dim = 4
    acoustic_layout = AcousticLayout.FIXED_LENGTH
    acoustic_unit_length = 3

    def __init__(self) -> None:
        self.decoded: SemanticAcousticCodes | None = None

    def tokenize(self, audio: torch.Tensor, sample_rate: int) -> object:
        del audio, sample_rate
        raise NotImplementedError

    def detokenize(self, codes: object) -> torch.Tensor:
        if not isinstance(codes, SemanticAcousticCodes):
            raise TypeError("expected SemanticAcousticCodes")
        self.decoded = codes
        return torch.zeros(1, 1, 32)

    def acoustic_codes_to_features(self, codes: torch.Tensor) -> torch.Tensor:
        return torch.zeros(*codes.shape[:2], self.acoustic_feature_dim)

    def decode_features(
        self,
        semantic_codes: torch.Tensor,
        acoustic_features: torch.Tensor,
    ) -> torch.Tensor:
        del semantic_codes, acoustic_features
        return torch.zeros(1, 1, 32)


class _FrameCodec:
    sample_rate = 16_000
    frame_rate = 50.0
    codebook_sizes = (8, 8)

    def __init__(self) -> None:
        self.decoded: torch.Tensor | None = None

    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        del audio, sample_rate
        raise NotImplementedError

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        self.decoded = codes
        return codes[..., 0].float()


def _sample():
    return {
        (Role.DEFAULT, Modality.TEXT): TextItem(
            views={TextView.TEXT: "hello"},
            meta={TextMeta.LANG: Lang.EN},
        ),
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views={
                AudioView.BICODEC: {
                    "semantic": torch.tensor([[1], [2]]),
                    "acoustic": torch.tensor([[0, 1], [2, 3], [4, 5]]),
                }
            },
            meta={AudioMeta.DURATION: 0.04},
        ),
    }


if __name__ == "__main__":
    unittest.main()
