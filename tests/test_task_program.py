from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout

from speech_to_speech.datamodule.builder import build_speech_sample
from speech_to_speech.datamodule.sample import Language, Speech
from speech_to_speech.runtime import AudioSequenceLayout
from speech_to_speech.task import (
    ControlToken,
    DIRECT,
    FULL_COT,
    PROGRAMS,
    TARGET_COT,
    FieldRole,
    PredictionModality,
    ResponseControl,
    ResponseLayout,
    ResponseSpec,
    ResponseStep,
    SourceLayout,
    Task,
    TaskField,
    TaskObjective,
    resolve_response,
    uses_source_ctc,
    uses_target_ctc,
)


class TaskProgramTest(unittest.TestCase):
    def test_programs_cover_every_task(self) -> None:
        self.assertEqual(set(PROGRAMS), set(Task))

    def test_task_default_properties_are_derived_from_program(self) -> None:
        expected = {
            Task.AUDIO_AR: (SourceLayout.NONE, PredictionModality.AUDIO, False),
            Task.ASR: (SourceLayout.AUDIO, PredictionModality.TEXT, False),
            Task.INTERLEAVED_AR: (
                SourceLayout.NONE,
                PredictionModality.INTERLEAVED,
                False,
            ),
            Task.MASKED_AR: (
                SourceLayout.TEXT_AUDIO,
                PredictionModality.PARALLEL,
                False,
            ),
            Task.MT: (SourceLayout.TEXT, PredictionModality.TEXT, True),
            Task.PARALLEL_AR: (
                SourceLayout.NONE,
                PredictionModality.PARALLEL,
                False,
            ),
            Task.S2ST: (SourceLayout.AUDIO, PredictionModality.AUDIO, True),
            Task.S2TT: (SourceLayout.AUDIO, PredictionModality.TEXT, True),
            Task.TEXT_AR: (SourceLayout.NONE, PredictionModality.TEXT, False),
            Task.T2ST: (SourceLayout.TEXT, PredictionModality.AUDIO, True),
            Task.T2TT: (SourceLayout.TEXT, PredictionModality.TEXT, True),
            Task.TTS: (SourceLayout.TEXT, PredictionModality.AUDIO, False),
            Task.TTS_VOICE_CLONE: (
                SourceLayout.TEXT_AUDIO,
                PredictionModality.AUDIO,
                True,
            ),
        }
        for task, (source, prediction, source_role) in expected.items():
            with self.subTest(task=task):
                self.assertIs(task.source_layout, source)
                self.assertIs(task.prediction_modality, prediction)
                self.assertEqual(task.uses_source_role, source_role)

    def test_s2st_response_traces_are_explicit_program_variants(self) -> None:
        direct = resolve_response(Task.S2ST)
        target = resolve_response(Task.S2ST, trace=TARGET_COT)
        full = resolve_response(Task.S2ST, trace=FULL_COT)

        self.assertEqual(direct.name, DIRECT)
        self.assertIs(direct.prediction, PredictionModality.AUDIO)
        self.assertEqual(
            [(field.role, field.modality) for field in target.fields],
            [
                (FieldRole.TARGET, Modality.TEXT),
                (FieldRole.TARGET, Modality.AUDIO),
            ],
        )
        self.assertIs(target.layout, ResponseLayout.BLOCKWISE)
        self.assertEqual(
            [(field.role, field.modality) for field in full.fields],
            [
                (FieldRole.SOURCE, Modality.TEXT),
                (FieldRole.TARGET, Modality.TEXT),
                (FieldRole.TARGET, Modality.AUDIO),
            ],
        )
        self.assertIs(full.prediction, PredictionModality.PARALLEL)

    def test_response_step_controls_are_explicit_for_every_trace(self) -> None:
        expected = {
            (Task.AUDIO_AR, DIRECT): (ResponseControl.AUDIO,),
            (Task.ASR, DIRECT): (ResponseControl.ASR,),
            (Task.INTERLEAVED_AR, DIRECT): (
                ResponseControl.EOS,
                ResponseControl.AUDIO,
            ),
            (Task.MASKED_AR, DIRECT): (
                ResponseControl.EOS,
                ResponseControl.AUDIO,
            ),
            (Task.MASKED_AR, "interleaved"): (
                ResponseControl.EOS,
                ResponseControl.AUDIO,
            ),
            (Task.MT, DIRECT): (ResponseControl.MT,),
            (Task.PARALLEL_AR, DIRECT): (
                ResponseControl.EOS,
                ResponseControl.AUDIO,
            ),
            (Task.S2ST, DIRECT): (ResponseControl.AUDIO,),
            (Task.S2ST, TARGET_COT): (
                ResponseControl.MT,
                ResponseControl.AUDIO,
            ),
            (Task.S2ST, FULL_COT): (
                ResponseControl.ASR,
                ResponseControl.MT,
                ResponseControl.AUDIO,
            ),
            (Task.S2TT, DIRECT): (ResponseControl.MT,),
            (Task.S2TT, FULL_COT): (
                ResponseControl.ASR,
                ResponseControl.MT,
            ),
            (Task.TEXT_AR, DIRECT): (ResponseControl.EOS,),
            (Task.T2ST, DIRECT): (ResponseControl.AUDIO,),
            (Task.T2ST, TARGET_COT): (
                ResponseControl.MT,
                ResponseControl.AUDIO,
            ),
            (Task.T2TT, DIRECT): (ResponseControl.MT,),
            (Task.TTS, DIRECT): (ResponseControl.AUDIO,),
            (Task.TTS_VOICE_CLONE, DIRECT): (ResponseControl.AUDIO,),
        }

        actual = {
            (task, response.name): tuple(
                step.control for step in response.steps
            )
            for task, program in PROGRAMS.items()
            for response in program.responses
        }
        self.assertEqual(actual, expected)

    def test_response_spec_rejects_unsupported_audio_ordering(self) -> None:
        text = ResponseStep(
            TaskField(FieldRole.TARGET, Modality.TEXT),
            ResponseControl.EOS,
        )
        audio = ResponseStep(
            TaskField(FieldRole.TARGET, Modality.AUDIO),
            ResponseControl.AUDIO,
        )

        with self.assertRaisesRegex(ValueError, "audio response step must be the final"):
            ResponseSpec(
                name="audio_first",
                steps=(audio, text),
                prediction=PredictionModality.PARALLEL,
            )
        with self.assertRaisesRegex(ValueError, "at most one audio step"):
            ResponseSpec(
                name="two_audio_steps",
                steps=(audio, audio),
                prediction=PredictionModality.AUDIO,
            )

    def test_prediction_is_not_a_response_selector(self) -> None:
        with self.assertRaises(TypeError):
            resolve_response(
                Task.S2ST,
                prediction=PredictionModality.PARALLEL,  # type: ignore[call-arg]
            )

    def test_invalid_trace_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported response trace"):
            resolve_response(Task.TTS, trace=FULL_COT)

    def test_objective_and_ctc_are_derived_from_visibility(self) -> None:
        self.assertIs(Task.TEXT_AR.program.objective, TaskObjective.CAUSAL)
        self.assertIs(Task.MASKED_AR.program.objective, TaskObjective.RECONSTRUCTION)
        self.assertTrue(uses_source_ctc(Task.ASR))
        self.assertTrue(uses_source_ctc(Task.S2TT))
        self.assertTrue(uses_source_ctc(Task.S2ST))
        self.assertFalse(uses_source_ctc(Task.MASKED_AR))
        self.assertTrue(uses_target_ctc(Task.AUDIO_AR))
        self.assertTrue(uses_target_ctc(Task.S2ST))
        self.assertTrue(uses_target_ctc(Task.T2ST))
        self.assertFalse(uses_target_ctc(Task.TTS))
        self.assertFalse(uses_source_ctc(Task.TTS_VOICE_CLONE))
        self.assertFalse(uses_target_ctc(Task.TTS_VOICE_CLONE))
        self.assertFalse(uses_target_ctc(Task.S2ST, trace=TARGET_COT))
        self.assertFalse(uses_target_ctc(Task.S2ST, trace=FULL_COT))

    def test_s2st_full_cot_builder_serializes_the_declared_response(self) -> None:
        runtime = _runtime()
        source, target = _speech_pair()

        sample = build_speech_sample(
            source,
            target,
            Task.S2ST,
            runtime,
            prompt="prefix $$$PLACEHOLDER$$$ suffix",
            trace=FULL_COT,
        )

        expected_prompt = torch.tensor([2, 24, 27, 16, 17, 25, 3])
        expected_response = torch.tensor(
            [10, 4, 5, 11, 12, 14, 6, 7, 13, 24, 27, 18, 19, 25]
        )
        expected_labels = torch.cat(
            (torch.full_like(expected_prompt, -100), expected_response)
        )
        self.assertEqual(sample.trace, FULL_COT)
        self.assertIs(sample.prediction, PredictionModality.PARALLEL)
        self.assertTrue(torch.equal(sample.request["prompt_ids"], expected_prompt))
        self.assertTrue(torch.equal(sample.labels.response_ids, expected_response))
        self.assertTrue(torch.equal(sample.token_labels, expected_labels))
        self.assertEqual(sample.target_language, "en")
        self.assertEqual(int(sample.labels.response_ids[0]), runtime.control_token_id(ControlToken.ASR_BEGIN))
        self.assertEqual(int(sample.input_ids[16]), runtime.boa_token_id)
        self.assertEqual(int(sample.input_ids[17]), runtime.audio_schema_token_id)
        self.assertEqual(int(sample.token_labels[16]), runtime.boa_token_id)
        self.assertEqual(int(sample.token_labels[17]), runtime.audio_schema_token_id)

        expected_source_positions = torch.tensor([3, 4])
        source_positions = sample.audio_input_positions
        self.assertIsNotNone(source_positions)
        assert source_positions is not None
        self.assertTrue(torch.equal(source_positions, expected_source_positions))
        self.assertIsNotNone(sample.source_ctc)
        assert sample.source_ctc is not None
        self.assertTrue(
            torch.equal(
                sample.source_ctc["token_positions"],
                expected_source_positions,
            )
        )
        self.assertTrue(torch.equal(sample.source_ctc["text_token_ids"], source.text_token_ids))
        self.assertIsNone(sample.target_ctc)

        self.assertIsNotNone(sample.acoustic_target)
        assert sample.acoustic_target is not None
        self.assertTrue(
            torch.equal(
                sample.acoustic_target["token_positions"],
                torch.tensor([18, 19]),
            )
        )

    def test_s2st_direct_and_target_cot_builder_contracts(self) -> None:
        runtime = _runtime()
        source, target = _speech_pair()
        prompt = "prefix $$$PLACEHOLDER$$$ suffix"

        direct = build_speech_sample(
            source,
            target,
            Task.S2ST,
            runtime,
            prompt=prompt,
        )
        self.assertEqual(direct.trace, DIRECT)
        self.assertIs(direct.prediction, PredictionModality.AUDIO)
        self.assertNotIn("target_language", direct.request)
        self.assertIsNone(direct.target_language)
        self.assertTrue(
            torch.equal(
                direct.request["prompt_ids"],
                torch.tensor([2, 24, 27, 16, 17, 25, 3]),
            )
        )
        self.assertTrue(
            torch.equal(
                direct.labels.response_ids,
                torch.tensor([24, 27, 18, 19, 25]),
            )
        )
        self.assertTrue(
            torch.equal(
                direct.token_labels,
                torch.tensor(
                    [-100, -100, -100, -100, -100, -100, -100, 24, 27, 18, 19, 25]
                ),
            )
        )
        self.assertIsNotNone(direct.source_ctc)
        self.assertIsNotNone(direct.target_ctc)
        assert direct.target_ctc is not None
        self.assertTrue(
            torch.equal(
                direct.target_ctc["token_positions"],
                torch.tensor([9, 10]),
            )
        )
        self.assertTrue(torch.equal(direct.target_ctc["text_token_ids"], target.text_token_ids))
        self.assertIsNotNone(direct.acoustic_target)
        assert direct.acoustic_target is not None
        self.assertTrue(
            torch.equal(
                direct.acoustic_target["token_positions"],
                torch.tensor([9, 10]),
            )
        )

        short = build_speech_sample(
            source,
            target,
            Task.S2ST,
            runtime,
            prompt=prompt,
            trace=TARGET_COT,
        )
        self.assertEqual(short.trace, TARGET_COT)
        self.assertEqual(short.target_language, "en")
        self.assertTrue(
            torch.equal(
                short.request["prompt_ids"],
                torch.tensor([2, 24, 27, 16, 17, 25, 3]),
            )
        )
        self.assertTrue(
            torch.equal(
                short.labels.response_ids,
                torch.tensor([12, 14, 6, 7, 13, 24, 27, 18, 19, 25]),
            )
        )
        self.assertTrue(
            torch.equal(
                short.token_labels,
                torch.tensor(
                    [
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        -100,
                        12,
                        14,
                        6,
                        7,
                        13,
                        24,
                        27,
                        18,
                        19,
                        25,
                    ]
                ),
            )
        )
        self.assertIsNotNone(short.source_ctc)
        self.assertIsNone(short.target_ctc)
        self.assertIsNotNone(short.acoustic_target)
        assert short.acoustic_target is not None
        self.assertTrue(
            torch.equal(
                short.acoustic_target["token_positions"],
                torch.tensor([14, 15]),
            )
        )


class _PromptTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise AssertionError("builder prompt pieces must not add special tokens")
        values = {
            "prefix ": [2],
            " suffix": [3],
        }
        return values[text]


def _runtime() -> Any:
    lexical_text_vocab_size = 10
    control_token_ids = tuple(
        range(lexical_text_vocab_size, lexical_text_vocab_size + len(ControlToken))
    )
    audio_start = lexical_text_vocab_size + len(ControlToken)
    return SimpleNamespace(
        text_tokenizer=_PromptTokenizer(),
        audio_tokenizer=object(),
        layout=Layout(text=(0, audio_start), audio=(audio_start, audio_start + 12)),
        lexical_text_vocab_size=lexical_text_vocab_size,
        control_token_ids=control_token_ids,
        control_token_id=lambda token: control_token_ids[list(ControlToken).index(token)],
        pad_token_id=0,
        eos_token_id=9,
        boa_token_id=audio_start + 8,
        eoa_token_id=audio_start + 9,
        mask_token_id=audio_start + 10,
        audio_schema_token_id=audio_start + 11,
        input_audio_block_name="audio",
        input_boa_token_id=audio_start + 8,
        input_eoa_token_id=audio_start + 9,
        input_audio_schema_token_id=audio_start + 11,
        acoustic_generator_artifact=None,
        audio_sequence_layout=AudioSequenceLayout.SEMANTIC,
    )


def _speech_pair() -> tuple[Speech, Speech]:
    source = Speech(
        semantic_codes=torch.tensor([[0], [1]]),
        acoustic_codes=None,
        text_token_ids=torch.tensor([4, 5]),
        audio_token_ids=torch.tensor([0, 1]),
        audio_token_spans=torch.ones(2, dtype=torch.long),
        language=Language.ZH,
        duration_seconds=0.2,
    )
    target = Speech(
        semantic_codes=torch.tensor([[2], [3]]),
        acoustic_codes=torch.tensor([[4], [5]]),
        text_token_ids=torch.tensor([6, 7]),
        audio_token_ids=torch.tensor([2, 3]),
        audio_token_spans=torch.ones(2, dtype=torch.long),
        language=Language.EN,
        duration_seconds=0.3,
    )
    return source, target


if __name__ == "__main__":
    unittest.main()
