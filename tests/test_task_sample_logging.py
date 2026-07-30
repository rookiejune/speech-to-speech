from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
from speech_to_speech.datamodule.types import (
    Language,
    ModelBatch,
    RawSpeech,
    RawSpeechBatch,
    SpeechTaskSample,
    Text,
)
from speech_to_speech.task import Task


class TaskSampleLoggingTest(unittest.TestCase):
    @patch(
        "speech_to_speech.callback.logging.task_sample.report_oom",
        return_value=True,
    )
    def test_materialization_oom_records_the_diagnostic_batch(self, report_oom):
        sample = _sample()
        diagnostic_batch = RawSpeechBatch(
            samples=(
                SpeechTaskSample(
                    source=Text(torch.tensor([1, 2]), Language.EN),
                    target=RawSpeech(
                        text_token_ids=torch.tensor([3]),
                        waveform=torch.zeros(2, 160),
                        sample_rate=16_000,
                        language=Language.EN,
                    ),
                    task=Task.TTS,
                    prediction=Task.TTS.prediction_modality,
                ),
            ),
            pad_token_id=0,
        )
        datamodule = SimpleNamespace(
            runtime=SimpleNamespace(),
            diagnostic_samples=Mock(return_value=[sample]),
            diagnostic_collator=Mock(
                return_value=Mock(return_value=diagnostic_batch)
            ),
        )
        experiment = Mock()
        trainer = SimpleNamespace(
            global_step=10,
            is_global_zero=True,
            logger=SimpleNamespace(experiment=experiment),
            datamodule=datamodule,
        )
        error = torch.OutOfMemoryError("codec allocation failed")
        module = SimpleNamespace(
            materialize_batch=Mock(side_effect=error),
            generate=Mock(),
        )
        callback = TaskSampleLogger(
            [0],
            every_n_steps=1,
            loader_name="tts",
            task=Task.TTS,
        )

        callback.on_fit_start(trainer, module)
        with self.assertRaises(torch.OutOfMemoryError) as raised:
            callback.on_train_batch_start(trainer, module, diagnostic_batch, 0)

        self.assertIs(raised.exception, error)
        report_oom.assert_called_once()
        self.assertEqual(
            report_oom.call_args.kwargs["phase"],
            "task_sample_materialize",
        )
        inputs = report_oom.call_args.kwargs["inputs"]
        self.assertEqual(inputs["type"], "RawSpeechBatch")
        self.assertEqual(
            inputs["samples"][0]["target"]["waveform"]["shape"],
            [2, 160],
        )
        module.generate.assert_not_called()

    @patch(
        "speech_to_speech.callback.logging.task_sample.report_oom",
        return_value=True,
    )
    def test_generation_oom_records_the_fixed_sample_context(self, report_oom):
        sample = _sample()
        batch = ModelBatch(
            input_ids=torch.tensor([[1, 2]]),
            token_labels=torch.tensor([[-100, 2]]),
            acoustic_target=None,
            tasks=[Task.TTS],
            predictions=[Task.TTS.prediction_modality],
            pad_token_id=0,
        )
        datamodule = SimpleNamespace(
            runtime=SimpleNamespace(),
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
        error = torch.OutOfMemoryError("generation allocation failed")
        module = SimpleNamespace(
            materialize_batch=Mock(side_effect=lambda value: value),
            generate=Mock(side_effect=error),
        )
        callback = TaskSampleLogger(
            [0],
            every_n_steps=1,
            loader_name="tts",
            task=Task.TTS,
            max_new_tokens=32,
            do_sample=False,
        )

        callback.on_fit_start(trainer, module)
        with self.assertRaises(torch.OutOfMemoryError) as raised:
            callback.on_train_batch_start(trainer, module, batch, 0)

        self.assertIs(raised.exception, error)
        report_oom.assert_called_once()
        self.assertEqual(
            report_oom.call_args.kwargs["phase"],
            "task_sample_generation",
        )
        self.assertEqual(
            report_oom.call_args.kwargs["inputs"]["padded_prompt_shape"],
            [1, 1],
        )
        self.assertEqual(
            report_oom.call_args.kwargs["inputs"]["max_new_tokens"],
            32,
        )
        experiment.add_text.assert_not_called()

    def test_train_mt_panel_logs_translation_and_text_metrics(self):
        sample = {
            (Role.SOURCE, Modality.TEXT): TextItem(
                views={TextView.TEXT: "hello source"},
                meta={TextMeta.LANG: Lang.ZH},
            ),
            (Role.TARGET, Modality.TEXT): TextItem(
                views={TextView.TEXT: "hello"},
                meta={TextMeta.LANG: Lang.EN},
            ),
        }
        batch = ModelBatch(
            input_ids=torch.tensor([[1, 2]]),
            token_labels=torch.tensor([[-100, 2]]),
            acoustic_target=None,
            tasks=[Task.MT],
            predictions=[Task.MT.prediction_modality],
            pad_token_id=0,
        )
        datamodule = SimpleNamespace(
            runtime=SimpleNamespace(
                layout=Layout(text=(0, 10), audio=(10, 20)),
                text_tokenizer=SimpleNamespace(decode=Mock(return_value="hello")),
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
                return_value=[{"response_ids": torch.tensor([1]), "audio": None}]
            ),
        )
        callback = TaskSampleLogger(
            [0],
            every_n_steps=1,
            loader_name="mt",
            split=SampleSplit.TRAIN,
            task=Task.MT,
            do_sample=False,
        )

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_start(trainer, module, batch, 0)

        datamodule.diagnostic_samples.assert_called_once_with(
            [0], split=SampleSplit.TRAIN, loader_name="mt"
        )
        datamodule.diagnostic_collator.assert_called_once_with(
            Task.MT, split=SampleSplit.TRAIN, loader_name="mt"
        )
        experiment.add_audio.assert_not_called()
        text_tags = {call.args[0] for call in experiment.add_text.call_args_list}
        self.assertIn("sample/mt/0/target", text_tags)
        self.assertIn("sample/mt/0/generated", text_tags)
        scalar_tags = {call.args[0] for call in experiment.add_scalar.call_args_list}
        self.assertIn("sample/mt/0/text/cer", scalar_tags)

    def test_validation_asr_panel_logs_source_and_model_free_metrics(self):
        sample = _sample()
        batch = ModelBatch(
            input_ids=torch.tensor([[1, 2]]),
            token_labels=torch.tensor([[-100, 2]]),
            acoustic_target=None,
            tasks=[Task.ASR],
            predictions=[Task.ASR.prediction_modality],
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
            "sample/asr/0/source",
        )
        scalar_tags = {call.args[0] for call in experiment.add_scalar.call_args_list}
        self.assertIn(
            "sample/asr/0/text/cer", scalar_tags
        )
        self.assertIn(
            "sample/asr/0/generation/response_tokens",
            scalar_tags,
        )
        metadata = experiment.add_text.call_args_list[0].args[1]
        self.assertIn('"seed": 7', metadata)
        self.assertIn("validation", callback.state_key)

    def test_tts_decode_error_logs_partial_audio_metadata(self):
        sample = _sample()
        batch = ModelBatch(
            input_ids=torch.tensor([[1, 2, 3]]),
            token_labels=torch.tensor([[-100, 2, 3]]),
            acoustic_target=None,
            tasks=[Task.TTS],
            predictions=[Task.TTS.prediction_modality],
            pad_token_id=0,
        )
        codec = _StructuredCodec()
        datamodule = SimpleNamespace(
            runtime=SimpleNamespace(
                codec=codec,
                audio_view=AudioView.BICODEC,
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
                    {
                        "response_ids": torch.tensor([11, 12]),
                        "audio": None,
                        "decode_error": {
                            "type": "ValueError",
                            "message": "incomplete",
                        },
                    }
                ]
            ),
        )
        callback = TaskSampleLogger(
            [0],
            every_n_steps=1,
            loader_name="tts",
            task=Task.TTS,
            max_new_tokens=2,
            do_sample=False,
        )

        callback.on_fit_start(trainer, module)
        callback.on_train_batch_start(trainer, module, batch, 0)

        metadata_text = next(
            call.args[1]
            for call in experiment.add_text.call_args_list
            if call.args[0] == "sample/tts/0/metadata"
        )
        metadata = json.loads(metadata_text)
        generation = metadata["generation"]
        self.assertEqual(generation["status"], "partial")
        self.assertIs(generation["result"]["audio_decode_failed"], True)
        self.assertEqual(
            generation["result"]["audio_decode_error"]["type"],
            "ValueError",
        )
        self.assertIn("stopped_without_eoa", generation["result"])
        self.assertNotIn("stopped_without_eos", generation["result"])
        self.assertIn(
            "generation/audio_decode_failed",
            generation["metrics"],
        )
        scalar_tags = {call.args[0] for call in experiment.add_scalar.call_args_list}
        self.assertIn(
            "sample/tts/0/generation/audio_decode_failed",
            scalar_tags,
        )
        self.assertIn(
            "sample/tts/0/generation/stopped_without_eoa",
            scalar_tags,
        )
        self.assertNotIn(
            "sample/tts/0/generation/stopped_without_eos",
            scalar_tags,
        )
        text_tags = {call.args[0] for call in experiment.add_text.call_args_list}
        self.assertIn("sample/tts/0/generated_ids", text_tags)

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
