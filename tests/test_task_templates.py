from __future__ import annotations

import random
import re
import unittest
from unittest.mock import patch

from speech_to_speech.prediction import PredictionModality
from speech_to_speech.source import SourceLayout
from speech_to_speech.task import Task
from speech_to_speech.templates import TEMPLATES, TEMPLATES_PER_TASK

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
        for task in Task:
            self.assertEqual(len(task.templates), TEMPLATES_PER_TASK)
            self.assertEqual(len(set(task.templates)), TEMPLATES_PER_TASK)
            if task in _ORIGINALS:
                self.assertNotIn(_ORIGINALS[task], task.templates)

    def test_template_placeholders_match_task_contract(self):
        for task in Task:
            required = _REQUIRED[task]
            forbidden = _FORBIDDEN.get(task, set())
            for template in task.templates:
                keys = set(_PLACEHOLDER_RE.findall(template))
                self.assertTrue(required.issubset(keys), msg=(task, template))
                self.assertFalse(keys & forbidden, msg=(task, template))
                self.assertFalse(keys - {"language", "source"}, msg=(task, template))
                template.format(language="en", source="<source>")

    def test_sample_template_uses_task_template_pool(self):
        task = Task.TTS
        with patch("speech_to_speech.task.random.choice", wraps=random.choice) as choice:
            template = task.sample_template()
        choice.assert_called_once_with(task.templates)
        self.assertIn(template, task.templates)

    def test_prediction_modality_contract(self):
        self.assertIs(Task.TEXT_AR.prediction_modality, PredictionModality.TEXT)
        self.assertIs(Task.AUDIO_AR.prediction_modality, PredictionModality.AUDIO)
        self.assertIs(Task.PARALLEL_AR.prediction_modality, PredictionModality.PARALLEL)
        self.assertIs(
            Task.INTERLEAVED_AR.prediction_modality,
            PredictionModality.INTERLEAVED,
        )
        self.assertIsNone(Task.PARALLEL_AR.target_modality)
        self.assertIsNone(Task.INTERLEAVED_AR.target_modality)
        self.assertEqual(
            Task.INTERLEAVED_AR.execution_signature,
            (SourceLayout.NONE, PredictionModality.INTERLEAVED),
        )
        self.assertEqual(
            Task.MASKED_AR.execution_signature,
            (SourceLayout.TEXT_AUDIO, PredictionModality.PARALLEL),
        )
        self.assertEqual(
            Task.T2ST.allowed_predictions,
            frozenset({PredictionModality.AUDIO, PredictionModality.PARALLEL}),
        )


if __name__ == "__main__":
    unittest.main()
