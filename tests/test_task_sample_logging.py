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
from speech_to_speech.callback.logging._sample_metrics import audio_metrics, text_metrics
from speech_to_speech.callback.logging.task_sample import (
    _log_source_audio,
    _log_target_audio,
    _request_metadata,
    _target_text,
)
from speech_to_speech.datamodule import SampleSplit
from speech_to_speech.datamodule.config import DataLoaderConfig, SpeechConfig
from speech_to_speech.datamodule.module import DataModule, LoaderSpec
from speech_to_speech.datamodule.types import (
    Language,
    ModelBatch,
    RawSpeech,
    RawSpeechBatch,
    SpeechTaskSample,
    Text,
)
from speech_to_speech.generation import Request, Result, decode_reference_codes
from speech_to_speech.prediction import PredictionModality
from speech_to_speech.task import Task


class TaskSampleLoggingTest(unittest.TestCase):
    @patch(
        "speech_to_speech.callback.logging.task_sample.report_oom",
        return_value=True,
    )
    def test_materialization_oom_records_the_diagnostic_batch(self, report_oom):
        diagnostic_batch = _raw_speech_batch()
        error = torch.OutOfMemoryError("codec allocation failed")
        ctx = _started_logger(
            diagnostic_batch,
            Task.TTS,
            sample=_sample(),
            materialize=error,
            loader_name="tts",
        )

        with self.assertRaises(torch.OutOfMemoryError) as raised:
            ctx.callback.on_train_batch_start(
                ctx.trainer, ctx.module, diagnostic_batch, 0
            )

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
        ctx.module.generate.assert_not_called()

    @patch(
        "speech_to_speech.callback.logging.task_sample.report_oom",
        return_value=True,
    )
    def test_generation_oom_records_the_fixed_sample_context(self, report_oom):
        batch = _batch(Task.TTS)
        error = torch.OutOfMemoryError("generation allocation failed")
        ctx = _started_logger(
            batch,
            Task.TTS,
            generate=error,
            loader_name="tts",
            max_new_tokens=32,
            do_sample=False,
        )

        with self.assertRaises(torch.OutOfMemoryError) as raised:
            ctx.callback.on_train_batch_start(ctx.trainer, ctx.module, batch, 0)

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
        ctx.experiment.add_text.assert_not_called()

    def test_train_mt_panel_logs_translation_and_text_metrics(self):
        batch = _batch(Task.MT)
        ctx = _started_logger(
            batch,
            Task.MT,
            sample=_text_sample("hello source", "hello"),
            runtime=_text_runtime(),
            results=[_text_result()],
            loader_name="mt",
            split=SampleSplit.TRAIN,
            do_sample=False,
        )

        ctx.callback.on_train_batch_start(ctx.trainer, ctx.module, batch, 0)

        ctx.datamodule.diagnostic_samples.assert_called_once_with(
            [0], split=SampleSplit.TRAIN, loader_name="mt"
        )
        ctx.datamodule.diagnostic_collator.assert_called_once_with(
            Task.MT, split=SampleSplit.TRAIN, loader_name="mt"
        )
        ctx.experiment.add_audio.assert_not_called()
        text_tags = _tags(ctx.experiment.add_text)
        self.assertIn("sample/mt/0/target", text_tags)
        self.assertIn("sample/mt/0/generated", text_tags)
        self.assertIn("sample/mt/0/text/cer", _tags(ctx.experiment.add_scalar))

    def test_validation_asr_panel_logs_source_and_model_free_metrics(self):
        batch = _batch(Task.ASR)
        codec = _StructuredCodec()
        ctx = _started_logger(
            batch,
            Task.ASR,
            runtime=_text_runtime(
                codec=codec,
                audio_view=AudioView.BICODEC,
            ),
            results=[_text_result()],
            loader_name="asr",
            split=SampleSplit.VALIDATION,
            seed=7,
            do_sample=False,
        )

        ctx.callback.on_train_batch_start(ctx.trainer, ctx.module, batch, 0)

        ctx.datamodule.diagnostic_samples.assert_called_once_with(
            [0], split=SampleSplit.VALIDATION, loader_name="asr"
        )
        ctx.datamodule.diagnostic_collator.assert_called_once_with(
            Task.ASR, split=SampleSplit.VALIDATION, loader_name="asr"
        )
        self.assertEqual(
            ctx.experiment.add_audio.call_args.args[0],
            "sample/asr/0/source",
        )
        scalar_tags = _tags(ctx.experiment.add_scalar)
        self.assertIn("sample/asr/0/text/cer", scalar_tags)
        self.assertIn(
            "sample/asr/0/generation/response_tokens",
            scalar_tags,
        )
        metadata = ctx.experiment.add_text.call_args_list[0].args[1]
        self.assertIn('"seed": 7', metadata)
        self.assertIn("validation", ctx.callback.state_key)

    def test_tts_decode_error_logs_partial_audio_metadata(self):
        batch = _batch(Task.TTS, input_ids=[1, 2, 3], token_labels=[-100, 2, 3])
        codec = _StructuredCodec()
        ctx = _started_logger(
            batch,
            Task.TTS,
            runtime=SimpleNamespace(
                codec=codec,
                audio_view=AudioView.BICODEC,
            ),
            results=[_decode_error_result()],
            loader_name="tts",
            max_new_tokens=2,
            do_sample=False,
        )

        ctx.callback.on_train_batch_start(ctx.trainer, ctx.module, batch, 0)

        metadata = _metadata(ctx.experiment, "sample/tts/0/metadata")
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
        scalar_tags = _tags(ctx.experiment.add_scalar)
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
        self.assertIn("sample/tts/0/generated_ids", _tags(ctx.experiment.add_text))

    def test_task_sample_logger_reuses_one_generation_result(self):
        batch = _batch(
            Task.TTS,
            input_ids=[1, 6, 4, 7],
            token_labels=[-100, -100, 4, 7],
        )
        result = Result(
            response_ids=torch.tensor([4]),
            audio={
                "features": torch.zeros(1, 2),
                "waveform": torch.zeros(1, 8),
                "sample_rate": 16_000,
            },
        )
        ctx = _started_logger(
            batch,
            Task.TTS,
            sample=_longcat_tts_sample(),
            runtime=_longcat_runtime(),
            results=[result],
            experiment=Mock(),
            global_step=1,
            loader_name="tts",
        )

        ctx.callback.on_train_batch_start(ctx.trainer, ctx.module, None, 0)

        ctx.module.generate.assert_called_once()
        self.assertEqual(ctx.experiment.add_audio.call_count, 2)
        audio_call = ctx.experiment.add_audio.call_args_list[1]
        self.assertEqual(audio_call.args[0], "sample/tts/0/generated")
        self.assertTrue(torch.equal(audio_call.args[1], result["audio"]["waveform"]))
        self.assertEqual(audio_call.args[2], 1)
        self.assertEqual(audio_call.kwargs, {"sample_rate": 16_000})
        ctx.experiment.add_text.assert_called_once()
        metadata = _metadata(ctx.experiment, "sample/tts/0/metadata")
        self.assertEqual(set(metadata), {"chat_template", "labels", "generation"})
        self.assertEqual(metadata["chat_template"]["task"], "tts")
        self.assertEqual(metadata["chat_template"]["dataset_index"], 0)
        self.assertEqual(metadata["chat_template"]["prompt_ids"]["ids"], [1, 6])
        self.assertEqual(metadata["chat_template"]["text"], "generated<audio>")
        self.assertEqual(metadata["labels"]["supervised_token_ids"]["ids"], [4, 7])
        self.assertEqual(metadata["generation"]["status"], "ok")
        self.assertEqual(metadata["generation"]["response_ids"]["ids"], [4])
        self.assertEqual(metadata["generation"]["result"]["duration_seconds"], 0.0005)
        self.assertTrue(metadata["generation"]["result"]["waveform_finite"])

    def test_parallel_override_logs_text_and_metrics_with_generated_audio(self):
        batch = _batch(Task.T2ST, prediction=PredictionModality.PARALLEL)
        result = Result(
            response_ids=torch.tensor([2, 6, 4, 7]),
            audio={
                "features": None,
                "codes": None,
                "waveform": torch.zeros(1, 8),
                "sample_rate": 16_000,
            },
        )
        ctx = _started_logger(
            batch,
            Task.T2ST,
            sample=_longcat_tts_sample(),
            runtime=_longcat_runtime(),
            results=[result],
            loader_name="t2st",
        )

        ctx.callback.on_train_batch_start(ctx.trainer, ctx.module, batch, 0)

        text_tags = _tags(ctx.experiment.add_text)
        self.assertIn("sample/t2st/0/target", text_tags)
        self.assertIn("sample/t2st/0/generated", text_tags)
        scalar_tags = _tags(ctx.experiment.add_scalar)
        self.assertIn("sample/t2st/0/text/cer", scalar_tags)
        self.assertIn("sample/t2st/0/text/exact_match", scalar_tags)
        metadata = _metadata(ctx.experiment, "sample/t2st/0/metadata")
        self.assertEqual(metadata["chat_template"]["prediction"], "parallel")
        self.assertEqual(metadata["generation"]["text"], "generated")

    def test_task_sample_logger_loads_samples_from_real_datamodule(self):
        samples = [Mock(), Mock()]
        config = SpeechConfig(
            codec="longcat",
            dataloader=DataLoaderConfig(batch_size=1, num_workers=0),
        )
        datamodule = DataModule(
            SimpleNamespace(codec_name="longcat"),
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )
        with patch(
            "speech_to_speech.datamodule.module.load_dataset",
            return_value=samples,
        ):
            datamodule.setup()
        trainer = SimpleNamespace(is_global_zero=True, datamodule=datamodule)
        callback = _logger(Task.TTS, loader_name="train", indices=[1, 0])

        callback.on_fit_start(trainer, SimpleNamespace())

        self.assertEqual(callback.samples, [samples[1], samples[0]])

    def test_task_sample_logger_state_key_distinguishes_fixed_loaders(self):
        asr = _logger(Task.ASR, loader_name="asr", every_n_steps=10)
        same_asr = _logger(Task.ASR, loader_name="asr", every_n_steps=10)
        tts = _logger(Task.TTS, loader_name="tts", every_n_steps=10)

        self.assertEqual(asr.state_key, same_asr.state_key)
        self.assertNotEqual(asr.state_key, tts.state_key)

    def test_task_sample_logger_logs_generation_failure(self):
        batch = _batch(Task.T2TT)
        ctx = _started_logger(batch, Task.T2TT, generate=RuntimeError("boom"))

        with self.assertRaisesRegex(RuntimeError, "boom"):
            ctx.callback.on_train_batch_start(ctx.trainer, ctx.module, None, 0)

        ctx.experiment.add_text.assert_called_once()
        metadata = _metadata(ctx.experiment)
        self.assertEqual(metadata["generation"]["status"], "failed")
        self.assertEqual(metadata["generation"]["error"]["type"], "RuntimeError")
        self.assertEqual(metadata["labels"]["supervised_token_ids"]["ids"], [2])

    def test_task_sample_logger_logs_row_count_mismatch(self):
        batch = _batch(Task.T2TT)
        ctx = _started_logger(batch, Task.T2TT, results=[])

        with self.assertRaisesRegex(RuntimeError, "wrong row count"):
            ctx.callback.on_train_batch_start(ctx.trainer, ctx.module, None, 0)

        metadata = _metadata(ctx.experiment)
        self.assertEqual(metadata["generation"]["status"], "failed")
        self.assertIn("wrong row count", metadata["generation"]["error"]["message"])

    def test_task_sample_logger_skips_nonzero_ranks(self):
        module = SimpleNamespace(generate=Mock())
        trainer = SimpleNamespace(global_step=1, is_global_zero=False)
        callback = _logger(Task.T2TT)

        callback.on_train_batch_start(trainer, module, None, 0)

        module.generate.assert_not_called()

    def test_text_metrics_are_model_free_and_normalized(self):
        metrics = text_metrics(" Ｈｅｌｌｏ  WORLD！ ", "hello world!")

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
        self.assertEqual(
            _target_text(_sample(), Task.ASR, PredictionModality.TEXT),
            "hello",
        )

    def test_waveform_only_metadata_does_not_parse_audio_as_codes(self):
        sample = _sample()
        sample[(Role.DEFAULT, Modality.AUDIO)] = AudioItem(
            views={AudioView.WAVEFORM: (torch.zeros(1, 8), 8)},
            meta={AudioMeta.DURATION: 1.0},
        )

        metadata = _request_metadata(
            3,
            sample,
            Request(
                prompt_ids=torch.tensor([1, 2]),
                task=Task.ASR,
            ),
        )

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


def _batch(
    task: Task,
    *,
    input_ids: list[int] | None = None,
    token_labels: list[int] | None = None,
    prediction: PredictionModality | None = None,
) -> ModelBatch:
    input_ids = [1, 2] if input_ids is None else input_ids
    token_labels = [-100, 2] if token_labels is None else token_labels
    return ModelBatch(
        input_ids=torch.tensor([input_ids]),
        token_labels=torch.tensor([token_labels]),
        acoustic_target=None,
        tasks=[task],
        predictions=[task.prediction_modality if prediction is None else prediction],
        pad_token_id=0,
    )


def _started_logger(
    batch: object,
    task: Task,
    *,
    sample: object | None = None,
    runtime: object | None = None,
    results: list[object] | None = None,
    generate: Exception | None = None,
    materialize: Exception | None = None,
    experiment: object | None = None,
    global_step: int = 10,
    loader_name: str = "train",
    split: SampleSplit = SampleSplit.TRAIN,
    **logger_kwargs: object,
):
    datamodule = _datamodule(batch, sample=sample, runtime=runtime)
    trainer, experiment = _trainer(
        datamodule,
        experiment=experiment,
        global_step=global_step,
    )
    module = _module(results, generate=generate, materialize=materialize)
    callback = _logger(
        task,
        loader_name=loader_name,
        split=split,
        **logger_kwargs,
    )
    callback.on_fit_start(trainer, module)
    return SimpleNamespace(
        callback=callback,
        datamodule=datamodule,
        experiment=experiment,
        module=module,
        trainer=trainer,
    )


def _raw_speech_batch() -> RawSpeechBatch:
    return RawSpeechBatch(
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


def _datamodule(
    batch: object,
    *,
    sample: object | None = None,
    runtime: object | None = None,
):
    return SimpleNamespace(
        runtime=SimpleNamespace() if runtime is None else runtime,
        diagnostic_samples=Mock(return_value=[_sample() if sample is None else sample]),
        diagnostic_collator=Mock(return_value=Mock(return_value=batch)),
    )


def _trainer(
    datamodule: object,
    *,
    experiment: object | None = None,
    global_step: int = 10,
    is_global_zero: bool = True,
):
    experiment = Mock() if experiment is None else experiment
    return (
        SimpleNamespace(
            global_step=global_step,
            is_global_zero=is_global_zero,
            logger=SimpleNamespace(experiment=experiment),
            datamodule=datamodule,
        ),
        experiment,
    )


def _module(
    results: list[object] | None = None,
    *,
    generate: Exception | None = None,
    materialize: Exception | None = None,
):
    return SimpleNamespace(
        model=Mock(),
        materialize_batch=Mock(
            side_effect=materialize if materialize is not None else lambda value: value
        ),
        generate=Mock(
            side_effect=generate,
            return_value=[] if results is None else results,
        ),
    )


def _logger(
    task: Task,
    *,
    indices: list[int] | None = None,
    every_n_steps: int = 1,
    loader_name: str = "train",
    split: SampleSplit = SampleSplit.TRAIN,
    **kwargs,
) -> TaskSampleLogger:
    return TaskSampleLogger(
        [0] if indices is None else indices,
        every_n_steps=every_n_steps,
        loader_name=loader_name,
        split=split,
        task=task,
        **kwargs,
    )


def _metadata(experiment: Mock, tag: str | None = None):
    if tag is None:
        return json.loads(experiment.add_text.call_args.args[1])
    text = next(
        call.args[1]
        for call in experiment.add_text.call_args_list
        if call.args[0] == tag
    )
    return json.loads(text)


def _tags(writer: Mock) -> set[str]:
    return {call.args[0] for call in writer.call_args_list}


def _longcat_runtime():
    return SimpleNamespace(
        codec=_FrameCodec(),
        audio_view=AudioView.LONGCAT,
        layout=Layout(text=(0, 4), audio=(4, 8)),
        text_tokenizer=SimpleNamespace(decode=Mock(return_value="generated")),
    )


def _text_runtime(decoded: str = "hello", **kwargs: object):
    values = {
        "layout": Layout(text=(0, 10), audio=(10, 20)),
        "text_tokenizer": SimpleNamespace(decode=Mock(return_value=decoded)),
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def _text_result(token_id: int = 1):
    return {"response_ids": torch.tensor([token_id]), "audio": None}


def _decode_error_result():
    return {
        "response_ids": torch.tensor([11, 12]),
        "audio": None,
        "decode_error": {
            "type": "ValueError",
            "message": "incomplete",
        },
    }


def _text_item(text: str, lang: Lang) -> TextItem:
    return TextItem(views={TextView.TEXT: text}, meta={TextMeta.LANG: lang})


def _text_sample(source: str, target: str):
    return {
        (Role.SOURCE, Modality.TEXT): _text_item(source, Lang.ZH),
        (Role.TARGET, Modality.TEXT): _text_item(target, Lang.EN),
    }


def _longcat_tts_sample():
    sample = _text_sample("source", "target")
    sample[(Role.TARGET, Modality.AUDIO)] = AudioItem(
        views={AudioView.LONGCAT: torch.zeros(1, 2, dtype=torch.long)},
        meta={AudioMeta.DURATION: 0.02},
    )
    return sample


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
