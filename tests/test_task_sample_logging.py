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
from anytrain.module.idspace import Layout

from speech_to_speech.callback.logging import TaskSampleLogger
from speech_to_speech.callback.logging.task_sample import (
    _log_source_audio,
    _log_target_audio,
    _request_metadata,
    _target_text,
)
from speech_to_speech.callback.logging._sample_metrics import audio_metrics, text_metrics
from speech_to_speech.generation import Request, decode_reference_codes
from speech_to_speech.datamodule import SampleSplit
from speech_to_speech.datamodule.types import ModelBatch
from speech_to_speech.task import Task


class TaskSampleLoggingTest(unittest.TestCase):
    def test_validation_asr_panel_logs_source_and_model_free_metrics(self):
        sample = _sample()
        batch = ModelBatch(
            input_ids=torch.tensor([[1, 2]]),
            token_labels=torch.tensor([[-100, 2]]),
            acoustic_target=None,
            tasks=[Task.ASR],
            pad_token_id=0,
        )
        codec = _StructuredCodec()
        datamodule = SimpleNamespace(
            runtime=SimpleNamespace(
                codec=codec,
                audio_view=AudioView.BICODEC,
                layout=Layout(text=(0, 10), audio=(10, 20)),
                text_tokenizer=SimpleNamespace(
                    decode=Mock(return_value="hello")
                ),
            ),
            diagnostic_samples=Mock(return_value=[sample]),
            diagnostic_collator=Mock(return_value=Mock(return_value=batch)),
        )
        experiment = Mock()
        trainer = SimpleNamespace(
            global_step=10,
            is_global_zero=True,
            logger=SimpleNamespace(experiment=experiment),
            datamodule=datamodule,
        )
        module = SimpleNamespace(
            model=Mock(),
            materialize_batch=Mock(side_effect=lambda value: value),
            generate=Mock(
                return_value=[
                    {"response_ids": torch.tensor([1]), "audio": None}
                ]
            ),
        )
        callback = TaskSampleLogger(
            [0],
            every_n_steps=1,
            loader_name="asr",
            split=SampleSplit.VALIDATION,
            task=Task.ASR,
            seed=7,
            do_sample=False,
        )

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_start(trainer, module, batch, 0)

        datamodule.diagnostic_samples.assert_called_once_with(
            [0], split=SampleSplit.VALIDATION, loader_name="asr"
        )
        datamodule.diagnostic_collator.assert_called_once_with(
            Task.ASR, split=SampleSplit.VALIDATION, loader_name="asr"
        )
        self.assertEqual(
            experiment.add_audio.call_args.args[0],
            "task_sample/validation/asr/asr/0/source",
        )
        scalar_tags = {call.args[0] for call in experiment.add_scalar.call_args_list}
        self.assertIn(
            "task_sample/validation/asr/asr/0/text/cer", scalar_tags
        )
        self.assertIn(
            "task_sample/validation/asr/asr/0/generation/response_tokens",
            scalar_tags,
        )
        metadata = experiment.add_text.call_args_list[0].args[1]
        self.assertIn('"seed": 7', metadata)
        self.assertIn("validation", callback.state_key)

    def test_text_metrics_are_model_free_and_normalized(self):
        metrics = text_metrics(" Ｈｅｌｌｏ  WORLD！ " , "hello world!")

        self.assertEqual(metrics["text/cer"], 0.0)
        self.assertEqual(metrics["text/exact_match"], 1.0)

    def test_audio_metrics_report_signal_health_and_duration_ratio(self):
        metrics = audio_metrics(
            torch.tensor([0.0, 0.5, 1.0, -1.0]),
            4,
            target_duration=2.0,
        )

        self.assertEqual(metrics["audio/duration_seconds"], 1.0)
        self.assertEqual(metrics["audio/duration_ratio"], 0.5)
        self.assertEqual(metrics["audio/silence_ratio"], 0.25)
        self.assertEqual(metrics["audio/clipping_ratio"], 0.5)

    def test_audio_source_is_logged_without_an_evaluator_model(self):
        codec = _StructuredCodec()
        writer = Mock()
        datamodule = SimpleNamespace(
            runtime=SimpleNamespace(codec=codec, audio_view=AudioView.BICODEC)
        )

        _log_source_audio(writer, datamodule, _sample(), Task.ASR, "sample", 7)

        writer.add_audio.assert_called_once()
        self.assertEqual(writer.add_audio.call_args.args[0], "sample/source")
        self.assertEqual(writer.add_audio.call_args.kwargs["sample_rate"], 16_000)

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

    def test_waveform_only_metadata_does_not_parse_audio_as_codes(self):
        sample = _sample()
        sample[(Role.DEFAULT, Modality.AUDIO)] = AudioItem(
            views={AudioView.WAVEFORM: (torch.zeros(1, 8), 8)},
            meta={AudioMeta.DURATION: 1.0},
        )

        metadata = _request_metadata(3, sample, Request(
            prompt_ids=torch.tensor([1, 2]),
            task=Task.ASR,
        ))

        self.assertEqual(metadata["source"]["view"], AudioView.WAVEFORM.value)
        self.assertEqual(metadata["source"]["sample_rate"], 8)
        self.assertEqual(metadata["source"]["duration_seconds"], 1.0)

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
