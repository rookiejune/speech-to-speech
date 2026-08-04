from __future__ import annotations

import unittest
from collections import Counter
from itertools import cycle
from unittest.mock import patch

from _config_helpers import _train, build_train_datamodule
from speech_to_speech.datamodule.collate.joint import (
    LoaderSchedule,
    ScheduledDataLoader,
)
from speech_to_speech.datamodule.dataset.text import TextDatasetName
from speech_to_speech.loader_plan import ARFraming
from speech_to_speech.loader_step import LoaderStepMode
from speech_to_speech.model.audio_input import AudioInputAdapterType
from speech_to_speech.model.acoustic import AcousticType
from speech_to_speech.parameter_policy import ParameterPolicyName
from speech_to_speech.runtime import AudioSequenceLayout, BackboneType
from speech_to_speech.task import Task


@patch.dict(
    "os.environ",
    {
        "DYNAMIC_HOME": "/tmp/dynamic",
        "SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer",
    },
)
class KimiARPretrainConfigTest(unittest.TestCase):
    def test_experiment_composes_the_four_task_single_stream_baseline(self) -> None:
        config = _train("experiment=train/kimi_audio/ar_pretrain")

        self.assertIs(config.runtime.backbone_type, BackboneType.KIMI_AUDIO)
        self.assertEqual(config.runtime.codec, "bicodec")
        self.assertIs(config.audio_sequence_layout, AudioSequenceLayout.FLATTENED)
        self.assertEqual(config.model.acoustic.type, AcousticType.NONE.value)
        self.assertIs(config.callbacks.parameter_policy.name, ParameterPolicyName.FULL)
        self.assertIsNone(config.model.lora)
        self.assertEqual(config.pl_module.ctc.source_weight, 1.0)
        self.assertEqual(config.pl_module.ctc.target_weight, 1.0)
        self.assertIs(
            config.model.audio_input_adapter.type,
            AudioInputAdapterType.TRANSFORMER,
        )
        self.assertFalse(config.model.audio_input_adapter.causal)
        self.assertEqual(
            config.loader_plan.loader_weights(),
            {"text_ar": 7.0, "audio_ar": 1.0, "asr": 1.0, "tts": 1.0},
        )
        self.assertIs(config.loader_plan.mode, LoaderStepMode.TOKEN_WEIGHTED)
        self.assertEqual(config.loader_plan.accumulate_grad_batches, 10)
        self.assertFalse(config.loader_plan.fuse_loaders_per_step)
        self.assertEqual(
            config.loader_plan.loaders["text_ar"].tasks,
            {Task.TEXT_AR: 1.0},
        )
        self.assertEqual(
            config.loader_plan.loaders["audio_ar"].tasks,
            {Task.AUDIO_AR: 1.0},
        )
        self.assertIs(
            config.loader_plan.loaders["text_ar"].framing,
            ARFraming.PRETRAINING,
        )
        self.assertIs(
            config.loader_plan.loaders["audio_ar"].framing,
            ARFraming.PRETRAINING,
        )
        self.assertIs(config.loader_plan.loaders["asr"].framing, ARFraming.INSTRUCTION)
        self.assertIs(config.loader_plan.loaders["tts"].framing, ARFraming.INSTRUCTION)
        self.assertTrue(config.loader_plan.loaders["text_ar"].is_text)
        self.assertFalse(config.loader_plan.loaders["audio_ar"].is_text)
        self.assertFalse(config.callbacks.task_sample.enabled)
        self.assertIs(config.text_datamodule.dataset.name, TextDatasetName.GENERAL)
        self.assertEqual(config.text_datamodule.max_tokens, 4096)
        self.assertTrue(config.text_datamodule.pack_documents)

        datamodule = build_train_datamodule(config, object())
        self.assertIs(
            datamodule.loader_specs["text_ar"].ar_framing,
            ARFraming.PRETRAINING,
        )
        self.assertIs(
            datamodule.loader_specs["audio_ar"].ar_framing,
            ARFraming.PRETRAINING,
        )
        self.assertIs(datamodule.loader_specs["asr"].ar_framing, ARFraming.INSTRUCTION)
        self.assertIs(datamodule.loader_specs["tts"].ar_framing, ARFraming.INSTRUCTION)

    def test_token_weighted_realizes_the_declared_token_ratio(self) -> None:
        config = _train("experiment=train/kimi_audio/ar_pretrain")
        schedule = LoaderSchedule(
            config.loader_plan.loader_weights(),
            accumulate_grad_batches=config.loader_plan.accumulate_grad_batches,
            fuse_loaders_per_step=config.loader_plan.fuse_loaders_per_step,
            step_mode=config.loader_plan.step_mode,
        )
        loaders = {name: cycle((name,)) for name in config.loader_plan.loaders}
        token_counts = {name: 1 for name in loaders}

        batches = iter(
            ScheduledDataLoader(
                loaders,
                schedule,
                token_counter=lambda name: token_counts[name],
            )
        )

        self.assertEqual(
            Counter(next(batches) for _ in range(10)),
            Counter({"text_ar": 7, "audio_ar": 1, "asr": 1, "tts": 1}),
        )

    def test_pretraining_framing_rejects_non_ar_loaders(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "only supports AUDIO_AR and TEXT_AR",
        ):
            _train(
                "experiment=train/kimi_audio/ar_pretrain",
                "+loader_plan.loaders.asr.ar_framing=pretraining",
            )


if __name__ == "__main__":
    unittest.main()
