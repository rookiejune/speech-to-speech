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
    DIRECT,
    FULL_COT,
    PROGRAMS,
    TARGET_COT,
    FieldRole,
    PredictionModality,
    ResponseLayout,
    SourceLayout,
    Task,
    TaskObjective,
    resolve_response,
    uses_source_ctc,
    uses_target_ctc,
)


class TaskProgramTest(unittest.TestCase):
    def test_programs_cover_every_task(self) -> None:
        self.assertEqual(set(PROGRAMS), set(Task))

    def test_legacy_task_properties_are_derived_without_behavior_changes(self) -> None:
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

    def test_prediction_override_keeps_legacy_short_cot_default(self) -> None:
        response = resolve_response(
            Task.S2ST,
            prediction=PredictionModality.PARALLEL,
        )
        self.assertEqual(response.name, TARGET_COT)

    def test_invalid_trace_or_prediction_combination_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported response trace"):
            resolve_response(Task.TTS, trace=FULL_COT)
        with self.assertRaisesRegex(ValueError, "requires prediction=parallel"):
            resolve_response(
                Task.S2ST,
                trace=FULL_COT,
                prediction=PredictionModality.AUDIO,
            )

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
        self.assertFalse(uses_target_ctc(Task.S2ST, PredictionModality.PARALLEL))

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

        expected_prompt = torch.tensor([2, 18, 10, 11, 19, 3])
        expected_response = torch.tensor([4, 5, 9, 6, 7, 9, 18, 12, 13, 19])
        expected_labels = torch.tensor(
            [-100, -100, -100, -100, -100, -100, 4, 5, 9, 6, 7, 9, -100, 12, 13, 19]
        )
        self.assertEqual(sample.trace, FULL_COT)
        self.assertIs(sample.prediction, PredictionModality.PARALLEL)
        self.assertTrue(torch.equal(sample.request["prompt_ids"], expected_prompt))
        self.assertTrue(torch.equal(sample.labels.response_ids, expected_response))
        self.assertTrue(torch.equal(sample.token_labels, expected_labels))
        self.assertEqual(int(sample.input_ids[12]), runtime.boa_token_id)
        self.assertEqual(int(sample.token_labels[12]), -100)

        expected_source_positions = torch.tensor([2, 3])
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
                torch.tensor([13, 14]),
            )
        )

    def test_s2st_direct_and_short_cot_builder_contracts_do_not_regress(self) -> None:
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
        self.assertTrue(
            torch.equal(
                direct.request["prompt_ids"],
                torch.tensor([2, 18, 10, 11, 19, 3, 18]),
            )
        )
        self.assertTrue(torch.equal(direct.labels.response_ids, torch.tensor([12, 13, 19])))
        self.assertTrue(
            torch.equal(
                direct.token_labels,
                torch.tensor([-100, -100, -100, -100, -100, -100, -100, 12, 13, 19]),
            )
        )
        self.assertIsNotNone(direct.source_ctc)
        self.assertIsNotNone(direct.target_ctc)
        assert direct.target_ctc is not None
        self.assertTrue(
            torch.equal(
                direct.target_ctc["token_positions"],
                torch.tensor([7, 8]),
            )
        )
        self.assertTrue(torch.equal(direct.target_ctc["text_token_ids"], target.text_token_ids))
        self.assertIsNotNone(direct.acoustic_target)
        assert direct.acoustic_target is not None
        self.assertTrue(
            torch.equal(
                direct.acoustic_target["token_positions"],
                torch.tensor([7, 8]),
            )
        )

        short = build_speech_sample(
            source,
            target,
            Task.S2ST,
            runtime,
            prompt=prompt,
            prediction=PredictionModality.PARALLEL,
        )
        self.assertEqual(short.trace, TARGET_COT)
        self.assertTrue(
            torch.equal(short.request["prompt_ids"], torch.tensor([2, 18, 10, 11, 19, 3]))
        )
        self.assertTrue(
            torch.equal(
                short.labels.response_ids,
                torch.tensor([6, 7, 9, 18, 12, 13, 19]),
            )
        )
        self.assertTrue(
            torch.equal(
                short.token_labels,
                torch.tensor([-100, -100, -100, -100, -100, -100, 6, 7, 9, -100, 12, 13, 19]),
            )
        )
        self.assertIsNotNone(short.source_ctc)
        self.assertIsNone(short.target_ctc)
        self.assertIsNotNone(short.acoustic_target)
        assert short.acoustic_target is not None
        self.assertTrue(
            torch.equal(
                short.acoustic_target["token_positions"],
                torch.tensor([10, 11]),
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
    return SimpleNamespace(
        text_tokenizer=_PromptTokenizer(),
        audio_tokenizer=object(),
        layout=Layout(text=(0, 10), audio=(10, 21)),
        pad_token_id=0,
        eos_token_id=9,
        boa_token_id=18,
        eoa_token_id=19,
        input_audio_block_name="audio",
        input_boa_token_id=18,
        input_eoa_token_id=19,
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
