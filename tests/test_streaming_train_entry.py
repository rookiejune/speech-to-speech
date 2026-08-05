from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

from anydataset.types import Lang, Modality, Role, Sample, TextItem, TextMeta, TextView
from lightning.pytorch.callbacks import Callback

from _config_helpers import _train
from scripts import train as train_script
from speech_to_speech.callback import StreamingSynthesis, SynthesisSampleLogger
from speech_to_speech.datamodule.config import StreamingConfig
from speech_to_speech.datamodule.streaming import PublishedSample


class StreamingConfigTest(unittest.TestCase):
    def test_enabled_stream_requires_identity_and_expected_sample_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "stream_id"):
            StreamingConfig(enabled=True, expected_samples=1)
        with self.assertRaisesRegex(ValueError, "expected_samples"):
            StreamingConfig(enabled=True, stream_id="stream-v1")

    def test_streaming_values_are_strict_and_intervals_normalize_to_float(self) -> None:
        config = StreamingConfig(poll_seconds=2, status_seconds=3)

        self.assertEqual(config.poll_seconds, 2.0)
        self.assertEqual(config.status_seconds, 3.0)
        for kwargs, error, message in (
            ({"enabled": 1}, TypeError, "enabled"),
            ({"expected_samples": True}, TypeError, "expected_samples"),
            ({"poll_seconds": False}, TypeError, "poll_seconds"),
            ({"status_seconds": 0}, ValueError, "status_seconds"),
            ({"producer_options": []}, TypeError, "producer_options"),
            ({"producer_options": {"command": "[]"}}, ValueError, "producer_factory"),
            ({"producer_factory": "module"}, ValueError, "module:attribute"),
            ({"stream_id": ""}, ValueError, "stream_id"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(error, message):
                StreamingConfig(**cast(Any, kwargs))


@patch.dict(
    "os.environ",
    {
        "DYNAMIC_HOME": "/tmp/dynamic",
        "SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer",
    },
)
class StreamingTrainConfigTest(unittest.TestCase):
    def test_valid_streaming_train_config_uses_one_unbounded_epoch(self) -> None:
        config = _streaming_train()

        self.assertEqual(config.trainer.max_epochs, 1)
        self.assertEqual(config.train.max_steps, -1)
        self.assertTrue(config.train.auto_resume)
        self.assertEqual(set(config.loader_plan.loaders), {"s2st"})

    @patch.dict(
        "os.environ",
        {
            "SPEECH_TO_SPEECH_STREAM_ROOT": "/tmp/stream",
            "SPEECH_TO_SPEECH_STREAM_ID": "wmt19-bidirectional-v1",
            "SPEECH_TO_SPEECH_STREAM_EXPECTED_SAMPLES": "8",
            "SPEECH_TO_SPEECH_STREAM_PRODUCER_COMMAND": '["/bin/true"]',
        },
    )
    def test_formal_streaming_experiment_resolves_producer_and_parallel_labels(
        self,
    ) -> None:
        config = _train("experiment=train/streaming_s2st")

        self.assertEqual(config.datamodule.streaming.expected_samples, 8)
        self.assertEqual(
            config.datamodule.streaming.producer_factory,
            "speech_to_speech.synthesis.process:controller",
        )
        self.assertEqual(
            config.datamodule.streaming.producer_options,
            {"command": '["/bin/true"]'},
        )
        self.assertEqual(config.loader_plan.loaders["s2st"].prediction, "parallel")

    def test_streaming_requires_resume_checkpoint_or_auto_resume(self) -> None:
        with self.assertRaisesRegex(ValueError, "auto_resume"):
            _streaming_train("train.auto_resume=false")

        config = _streaming_train(
            "train.auto_resume=false",
            "train.ckpt_path=/tmp/stream.ckpt",
        )
        self.assertEqual(config.train.ckpt_path, "/tmp/stream.ckpt")

    def test_streaming_rejects_incompatible_formal_train_contracts(self) -> None:
        cases = (
            (("trainer.max_epochs=2",), "max_epochs=1"),
            (("train.max_steps=10",), "max_steps=-1"),
            (("trainer.enable_checkpointing=false",), "checkpointing"),
            (("callbacks.checkpoint.save_last=false",), "save_last"),
            (
                ("+loader_plan.loaders.mt={weight:0.5,task_weights:{mt:1.0}}",),
                "exactly one training loader",
            ),
            (
                (
                    "~loader_plan.loaders.s2st",
                    "+loader_plan.loaders.s2st={weight:1.0,task_weights:{mt:1.0},prediction:text}",
                ),
                "one speech loader",
            ),
            (
                ("validation.enabled=true",),
                "validation",
            ),
            (
                ("callbacks.task_sample.enabled=true",),
                "task_sample requires fixed samples",
            ),
            (
                ("loader_plan.loaders.s2st.prediction=audio",),
                "prediction=parallel",
            ),
            (
                (
                    "~loader_plan.loaders.s2st",
                    "+loader_plan.loaders.s2st={weight:1.0,task_weights:{tts:1.0}}",
                ),
                "exactly the s2st task",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                _streaming_train(
                    *overrides,
                    *(
                        (
                            "callbacks.task_sample.panels=[{split:train,loader:s2st,task:s2st,indices:[0]}]",
                        )
                        if overrides == ("callbacks.task_sample.enabled=true",)
                        else ()
                    ),
                )


@patch.dict(
    "os.environ",
    {
        "DYNAMIC_HOME": "/tmp/dynamic",
        "SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer",
    },
)
class StreamingTrainEntryTest(unittest.TestCase):
    def test_streaming_callbacks_run_before_schedule_and_sample_logging(self) -> None:
        config = _streaming_train(
            "callbacks.synthesis_sample.enabled=true",
            "callbacks.synthesis_sample.indices=[0]",
        )
        schedule_runtime = Mock()
        schedule_runtime.callbacks.return_value = []

        callbacks = train_script.training_callbacks(
            config,
            Path("/tmp/output"),
            Mock(spec=Callback),
            schedule_runtime=schedule_runtime,
        )

        streaming_index = next(
            index
            for index, callback in enumerate(callbacks)
            if isinstance(callback, StreamingSynthesis)
        )
        synthesis_index = next(
            index
            for index, callback in enumerate(callbacks)
            if isinstance(callback, SynthesisSampleLogger)
        )
        self.assertLess(streaming_index, synthesis_index)
        self.assertEqual(
            [type(callback) for callback in train_script._lifecycle_callbacks(config)],
            [StreamingSynthesis],
        )

    def test_streaming_disables_epoch_dataloader_reload(self) -> None:
        config = _streaming_train()
        with (
            patch("scripts.train.create_trainer") as create_trainer,
            patch("scripts.train.build_logger"),
        ):
            train_script.build_trainer(config, Path("/tmp/output"), [])

        self.assertEqual(
            create_trainer.call_args.kwargs["reload_dataloaders_every_n_epochs"],
            0,
        )

    def test_auto_resume_uses_last_checkpoint_only_when_present(self) -> None:
        config = _streaming_train()
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            self.assertIsNone(train_script._resume_checkpoint(config, output_dir))

            checkpoint = output_dir / "checkpoints" / "last.ckpt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            self.assertEqual(
                train_script._resume_checkpoint(config, output_dir),
                str(checkpoint),
            )

        explicit = _streaming_train("train.ckpt_path=/tmp/explicit.ckpt")
        self.assertEqual(
            train_script._resume_checkpoint(explicit, Path("/missing")),
            "/tmp/explicit.ckpt",
        )


class SynthesisSampleLoggerTest(unittest.TestCase):
    def test_logs_published_text_audio_tags_and_restores_logged_state(self) -> None:
        sample = cast(
            Sample,
            {
                (Role.SOURCE, Modality.TEXT): _text_item("source", Lang.ZH),
                (Role.TARGET, Modality.TEXT): _text_item("target", Lang.EN),
            },
        )
        published = PublishedSample(3, "snapshot-0007", sample)
        datamodule = SimpleNamespace(
            streaming_enabled=True,
            published_streaming_samples=Mock(return_value=[published]),
        )
        trainer = SimpleNamespace(
            datamodule=datamodule,
            global_step=4,
            is_global_zero=True,
        )
        audio_writer = Mock()
        text_writer = Mock()
        callback = SynthesisSampleLogger([3], 2, loader_name="s2st")

        with (
            patch(
                "speech_to_speech.callback.streaming.experiment.audio",
                return_value=audio_writer,
            ),
            patch(
                "speech_to_speech.callback.streaming.experiment.text",
                return_value=text_writer,
            ),
            patch(
                "speech_to_speech.callback.logging.sample_report.sample_audio",
                side_effect=[("source-wave", 16_000), ("target-wave", 24_000)],
            ),
        ):
            callback.on_train_batch_start(
                cast(Any, trainer), cast(Any, object()), None, 0
            )

        self.assertEqual(
            _tags(text_writer.add_text),
            {
                "synthesis/3/source_text",
                "synthesis/3/target_text",
                "synthesis/3/metadata",
            },
        )
        metadata = json.loads(
            next(
                call.args[1]
                for call in text_writer.add_text.call_args_list
                if call.args[0] == "synthesis/3/metadata"
            )
        )
        self.assertEqual(
            metadata,
            {
                "dataset_index": 3,
                "snapshot_id": "snapshot-0007",
                "source_text": "source",
                "target_text": "target",
            },
        )
        self.assertEqual(
            _tags(audio_writer.add_audio),
            {"synthesis/3/source_audio", "synthesis/3/target_audio"},
        )
        self.assertEqual(callback.state_dict(), {"interval": {"last_step": 4}, "logged": [3]})

        restored = SynthesisSampleLogger([3], 2, loader_name="s2st")
        restored.load_state_dict(callback.state_dict())
        restored.on_train_batch_start(cast(Any, trainer), cast(Any, object()), None, 0)
        datamodule.published_streaming_samples.assert_called_once_with([3], loader_name="s2st")


def _streaming_train(*overrides: str):
    return _train(
        "datamodule/dataset=streaming_s2st",
        "datamodule.streaming.enabled=true",
        "datamodule.streaming.stream_id=test-stream-v1",
        "datamodule.streaming.expected_samples=4",
        "datamodule.dataloader.num_workers=0",
        "datamodule.dataloader.persistent_workers=false",
        "datamodule.dataloader.costs.enabled=false",
        "trainer.max_epochs=1",
        "trainer.use_distributed_sampler=false",
        "train.max_steps=-1",
        "train.auto_resume=true",
        "~loader_plan.loaders.tts",
        "~loader_plan.loaders.mt",
        "+loader_plan.loaders.s2st={weight:1.0,task_weights:{s2st:1.0},prediction:parallel}",
        "loader_plan.accumulate_grad_batches=1",
        "loader_plan.step_mode=weighted_window",
        "callbacks.task_sample.enabled=false",
        "callbacks.text_retention.enabled=false",
        *overrides,
    )


def _text_item(text: str, language: Lang) -> TextItem:
    return TextItem(
        views={TextView.TEXT: text},
        meta={TextMeta.LANG: language},
    )


def _tags(writer: Mock) -> set[str]:
    return {call.args[0] for call in writer.call_args_list}


if __name__ == "__main__":
    unittest.main()
