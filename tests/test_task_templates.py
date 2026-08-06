from __future__ import annotations

import re
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from speech_to_speech.datamodule.config import DataLoaderConfig, SpeechConfig
from speech_to_speech.datamodule.dataset.text import TextConfig
from speech_to_speech.task import (
    ControlToken,
    FULL_COT,
    PredictionModality,
    ResponseControl,
    SourceLayout,
    TARGET_COT,
    Task,
    response_control_tokens,
    resolve_response,
)
from speech_to_speech.task.templates import (
    TEMPLATES,
    TEMPLATES_PER_TASK,
    evaluation_template_index,
    format_instruction,
    format_response_instruction,
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
    Task.TTS_VOICE_CLONE: {"source"},
}
_FORBIDDEN = {
    Task.AUDIO_AR: {"source"},
    Task.INTERLEAVED_AR: {"source"},
    Task.MASKED_AR: {"source"},
    Task.PARALLEL_AR: {"source"},
    Task.TEXT_AR: {"language", "source"},
}


class TaskTemplateTest(unittest.TestCase):
    def test_templates_cover_every_task_exactly_once(self):
        self.assertEqual(set(TEMPLATES), set(Task))
        for task in Task:
            self.assertEqual(len(task.templates), TEMPLATES_PER_TASK)
            self.assertEqual(len(set(task.templates)), TEMPLATES_PER_TASK)

    def test_template_placeholders_match_task_contract(self):
        for task in Task:
            required = _REQUIRED[task]
            forbidden = _FORBIDDEN.get(task, set())
            for template in TEMPLATES[task]:
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
        with patch("speech_to_speech.task.templates.random.choice") as choice:
            template = task.sample_template()
        choice.assert_not_called()
        self.assertEqual(template, task.templates[0])

    def test_null_index_is_random(self):
        task = Task.S2ST
        with patch(
            "speech_to_speech.task.templates.random.choice",
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

    def test_full_cot_instruction_declares_the_exact_response_order(self):
        base = format_instruction(
            Task.S2ST,
            source="<audio>",
            language="English",
            index=0,
        )
        direct = format_response_instruction(
            base,
            resolve_response(Task.S2ST),
            language="English",
        )
        target = format_response_instruction(
            base,
            resolve_response(Task.S2ST, trace=TARGET_COT),
            language="English",
        )
        full = format_response_instruction(
            base,
            resolve_response(Task.S2ST, trace=FULL_COT),
            language="English",
        )

        self.assertEqual(direct, base)
        self.assertLess(
            target.index("1. produce the English translation as text"),
            target.index("2. generate the corresponding English speech"),
        )
        self.assertNotIn("transcribe the source speech as text", target)
        self.assertLess(
            full.index("1. transcribe the source speech as text"),
            full.index("2. produce the English translation as text"),
        )
        self.assertLess(
            full.index("2. produce the English translation as text"),
            full.index("3. generate the corresponding English speech"),
        )
        for token in ControlToken:
            self.assertNotIn(token.value, target)
            self.assertNotIn(token.value, full)

    def test_direct_text_tasks_describe_their_control_semantics(self):
        cases = (
            (
                Task.ASR,
                "English",
                "transcribe the speech as text",
            ),
            (
                Task.MT,
                "English",
                "produce the English translation as text",
            ),
            (
                Task.S2TT,
                "Chinese",
                "produce the Chinese translation as text",
            ),
            (
                Task.T2TT,
                "English",
                "produce the English translation as text",
            ),
        )
        for task, language, expected in cases:
            with self.subTest(task=task, language=language):
                base = format_instruction(
                    task,
                    source="<source>",
                    language=language,
                    index=0,
                )
                formatted = format_response_instruction(
                    base,
                    resolve_response(task),
                    language=language,
                )

                self.assertTrue(formatted.endswith(expected))

        asr = format_response_instruction(
            "Transcribe this speech.",
            resolve_response(Task.ASR),
            language="English",
        )
        self.assertNotIn("translation as text", asr)

    def test_control_tokens_remain_response_targets_not_prompt_literals(self):
        asr = response_control_tokens(ResponseControl.ASR)
        mt = response_control_tokens(ResponseControl.MT, target_language="English")

        self.assertIsNotNone(asr)
        self.assertIsNotNone(mt)
        assert asr is not None
        assert mt is not None
        self.assertEqual(asr.prefix, (ControlToken.ASR_BEGIN,))
        self.assertEqual(asr.end, ControlToken.ASR_END)
        self.assertEqual(
            mt.prefix,
            (ControlToken.MT_BEGIN, ControlToken.LANG_EN),
        )
        self.assertEqual(mt.end, ControlToken.MT_END)

        prompt = format_response_instruction(
            "Translate this speech.",
            resolve_response(Task.S2ST, trace=FULL_COT),
            language="English",
        )
        for token in ControlToken:
            self.assertNotIn(token.value, prompt)

    def test_speech_config_accepts_per_task_templates(self):
        from scripts._config.normalization import prepare

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
            {
                response.prediction
                for response in Task.MASKED_AR.program.responses
            },
            {PredictionModality.PARALLEL, PredictionModality.INTERLEAVED},
        )
        self.assertIs(Task.MASKED_AR.source_layout, SourceLayout.TEXT_AUDIO)
        self.assertIsNone(Task.MASKED_AR.target_modality)
        self.assertIsNone(Task.PARALLEL_AR.target_modality)
        self.assertIsNone(Task.INTERLEAVED_AR.target_modality)


if __name__ == "__main__":
    unittest.main()
