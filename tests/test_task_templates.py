from __future__ import annotations

import re
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from speech_to_speech.datamodule.config import DataLoaderConfig, SpeechConfig
from speech_to_speech.datamodule.dataset.text import TextConfig
from speech_to_speech.prediction import PredictionModality
from speech_to_speech.source import SourceLayout
from speech_to_speech.task import Task
from speech_to_speech.templates import (
    CANONICAL_TEMPLATES,
    CANONICAL_TEMPLATES_PER_TASK,
    TEMPLATES,
    TEMPLATES_PER_TASK,
    evaluation_template_index,
    format_instruction,
    select_template,
)

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
_REQUIRED = {
    Task.AUDIO_AR: {"language"},
    Task.ASR: {"language", "source"},
    Task.INTERLEAVED_AR: {"language"},
    Task.MASKED_AR: {"language"},
    Task.MT: {"language", "source"},
    Task.PARALLEL_AR: {"language"},
    Task.S2ST: {"language", "source"},
    Task.S2TT: {"language", "source"},
    Task.TEXT_AR: set(),
    Task.T2ST: {"language", "source"},
    Task.T2TT: {"language", "source"},
    Task.TTS: {"source"},
}
_FORBIDDEN = {
    Task.AUDIO_AR: {"source"},
    Task.INTERLEAVED_AR: {"source"},
    Task.MASKED_AR: {"source"},
    Task.PARALLEL_AR: {"source"},
    Task.TEXT_AR: {"language", "source"},
}
_ORIGINALS = {
    Task.AUDIO_AR: "Continue the {language} speech.",
    Task.ASR: "Transcribe the {language} speech: {source}",
    Task.MT: "Translate the following text into {language}: {source}",
    Task.S2ST: "Translate the following speech into {language} speech: {source}",
    Task.S2TT: "Translate the following speech into {language} text: {source}",
    Task.TEXT_AR: "Continue the following text.",
    Task.T2ST: "Translate the following text into {language} speech: {source}",
    Task.T2TT: "Translate the following text into {language}: {source}",
    Task.TTS: "Synthesize speech from the following text: {source}",
}


class TaskTemplateTest(unittest.TestCase):
    def test_templates_cover_every_task_exactly_once(self):
        self.assertEqual(set(TEMPLATES), set(Task))
        self.assertEqual(set(CANONICAL_TEMPLATES), set(Task))
        for task in Task:
            self.assertEqual(len(task.templates), TEMPLATES_PER_TASK)
            self.assertEqual(len(set(task.templates)), TEMPLATES_PER_TASK)
            self.assertEqual(
                len(CANONICAL_TEMPLATES[task]),
                CANONICAL_TEMPLATES_PER_TASK,
            )
            if task in _ORIGINALS:
                self.assertNotIn(_ORIGINALS[task], task.templates)
                self.assertEqual(CANONICAL_TEMPLATES[task], (_ORIGINALS[task],))

    def test_template_placeholders_match_task_contract(self):
        for mapping in (TEMPLATES, CANONICAL_TEMPLATES):
            for task in Task:
                required = _REQUIRED[task]
                forbidden = _FORBIDDEN.get(task, set())
                for template in mapping[task]:
                    keys = set(_PLACEHOLDER_RE.findall(template))
                    self.assertTrue(
                        required.issubset(keys),
                        msg=(task, template),
                    )
                    self.assertFalse(
                        keys & forbidden,
                        msg=(task, template),
                    )
                    self.assertFalse(
                        keys - {"language", "source"},
                        msg=(task, template),
                    )
                    template.format(language="en", source="<source>")

    def test_sample_template_defaults_to_index_zero(self):
        task = Task.TTS
        with patch("speech_to_speech.templates.random.choice") as choice:
            template = task.sample_template()
        choice.assert_not_called()
        self.assertEqual(template, task.templates[0])

    def test_null_index_is_random(self):
        task = Task.S2ST
        with patch(
            "speech_to_speech.templates.random.choice",
            return_value=TEMPLATES[task][2],
        ) as choice:
            sampled = select_template(task, None)
        choice.assert_called_once_with(TEMPLATES[task])
        self.assertEqual(sampled, TEMPLATES[task][2])
        self.assertEqual(select_template(task, 1), TEMPLATES[task][1])
        with self.assertRaisesRegex(IndexError, "outside"):
            select_template(task, TEMPLATES_PER_TASK)

    def test_evaluation_template_index_pins_null_to_zero(self):
        self.assertEqual(evaluation_template_index(None), 0)
        self.assertEqual(evaluation_template_index(3), 3)

    def test_format_instruction_fills_required_placeholders(self):
        text = format_instruction(Task.TTS, source="hello", index=0)
        self.assertIn("hello", text)
        with self.assertRaisesRegex(ValueError, "fixed template index"):
            format_instruction(Task.TTS, source="hello", index=None)

    def test_speech_config_accepts_per_task_templates(self):
        from scripts._config_normalization import prepare

        raw = OmegaConf.create(
            {
                "datamodule": {
                    "codec": "longcat",
                    "dataloader": {"batch_size": 1, "num_workers": 0},
                    "tasks": {
                        "tts": {"template": None},
                        "asr": {"template": 2},
                    },
                }
            }
        )
        prepared = prepare(raw)
        speech = OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(SpeechConfig),
                prepared.datamodule,
            )
        )
        self.assertIsInstance(speech, SpeechConfig)
        self.assertEqual(set(speech.tasks or {}), {Task.TTS, Task.ASR})
        self.assertIsNone(speech.template_index(Task.TTS))
        self.assertEqual(speech.template_index(Task.ASR), 2)

        text = TextConfig(
            dataloader=DataLoaderConfig(batch_size=1, num_workers=0),
        )
        self.assertIsInstance(text, TextConfig)

    def test_missing_task_config_for_loader_fails(self):
        config = SpeechConfig(
            codec="longcat",
            dataloader=DataLoaderConfig(batch_size=1, num_workers=0),
            tasks={},
        )
        with self.assertRaisesRegex(KeyError, "missing"):
            config.template_index(Task.TTS)

    def test_prediction_modality_contract(self):
        self.assertIs(Task.TEXT_AR.prediction_modality, PredictionModality.TEXT)
        self.assertIs(Task.AUDIO_AR.prediction_modality, PredictionModality.AUDIO)
        self.assertIs(Task.PARALLEL_AR.prediction_modality, PredictionModality.PARALLEL)
        self.assertIs(
            Task.INTERLEAVED_AR.prediction_modality,
            PredictionModality.INTERLEAVED,
        )
        self.assertIs(Task.MASKED_AR.prediction_modality, PredictionModality.PARALLEL)
        self.assertEqual(
            Task.MASKED_AR.allowed_predictions,
            frozenset({PredictionModality.PARALLEL, PredictionModality.INTERLEAVED}),
        )
        self.assertIs(Task.MASKED_AR.source_layout, SourceLayout.TEXT_AUDIO)
        self.assertIsNone(Task.MASKED_AR.target_modality)
        self.assertIsNone(Task.PARALLEL_AR.target_modality)
        self.assertIsNone(Task.INTERLEAVED_AR.target_modality)


if __name__ == "__main__":
    unittest.main()
