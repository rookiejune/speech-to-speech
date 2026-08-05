from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout
from torch import Tensor, nn

from speech_to_speech.generation.service import generate_responses
from speech_to_speech.generation.text import decode_response_text_steps
from speech_to_speech.model.generation import GenerationStepResult
from speech_to_speech.task import (
    FULL_COT,
    TARGET_COT,
    PredictionModality,
    Request,
    Task,
    resolve_response,
)


class _Tokenizer:
    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return ",".join(str(int(token_id)) for token_id in token_ids)


class _Runtime:
    layout = Layout(text=(0, 4), audio=(4, 8))
    text_tokenizer = _Tokenizer()
    pad_token_id = 0
    eos_token_id = 3
    boa_token_id = 6
    eoa_token_id = 7
    codec_audio_range = (4, 6)
    input_codec_audio_range = codec_audio_range
    audio_generation_allowed_ids = (4, 5, 7)
    acoustic_side_channel = False
    structured_full_sequence = False

    def generation_allowed_ids(self, modality: Modality) -> tuple[int, ...]:
        if modality is not Modality.TEXT:
            raise ValueError("test runtime only exposes text generation ids here.")
        return (1, 2)


class _ScriptModel:
    def __init__(self, script: list[int]) -> None:
        self.runtime = _Runtime()
        self.backbone = SimpleNamespace(
            get_input_embeddings=lambda: nn.Embedding(1, 1)
        )
        self.audio_token_frame_spans = torch.ones(2, dtype=torch.long)
        self._script = script
        self._step = 0

    def generation_step(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor,
        output_hidden_states: bool,
        token_ids: Tensor | None,
        token_kind: str | None = None,
        modality: Modality | None,
        past_key_values=None,
        use_cache: bool = False,
        audio_input_positions: Tensor | None = None,
        audio_head_past: object | None = None,
        input_modalities: frozenset[Modality] | None = None,
        validate_input: bool = True,
        validate_audio_input_positions: bool = True,
    ) -> GenerationStepResult:
        del (
            attention_mask,
            token_kind,
            modality,
            past_key_values,
            audio_input_positions,
            audio_head_past,
            input_modalities,
            validate_input,
            validate_audio_input_positions,
        )
        if self._step >= len(self._script):
            raise RuntimeError("generation script exhausted.")
        next_id = self._script[self._step]
        self._step += 1
        logits = torch.full(
            (*input_ids.shape, self.runtime.layout.vocab_size),
            float("-inf"),
        )
        logits[:, -1, next_id] = 0.0
        if token_ids is not None:
            logits = logits.index_select(-1, token_ids)
        return GenerationStepResult(
            logits=logits,
            past_key_values=SimpleNamespace() if use_cache else None,
            audio_head_past=None,
            hidden_states=(torch.zeros(*input_ids.shape, 1),)
            if output_hidden_states
            else None,
        )

    def select_audio_head_cache(self, past_key_values, indices):
        del indices
        return past_key_values

    def generate_tokens(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("multi-step responses must use generation_step().")


def _request(task: Task, trace: str) -> Request:
    response = resolve_response(task, trace=trace)
    return Request(
        prompt_ids=torch.tensor([1]),
        task=task,
        prediction=response.prediction,
        trace=response.name,
        audio_input_positions=None,
    )


def _results(rows, model):
    del model
    return [{"response_ids": row, "audio": None} for row in rows]


class TaskProgramGenerationTest(unittest.TestCase):
    def test_full_s2st_advances_through_two_text_steps_then_audio(self) -> None:
        expected = torch.tensor([1, 3, 2, 3, 6, 4, 7])
        for use_cache in (False, True):
            with self.subTest(use_cache=use_cache):
                model = _ScriptModel([1, 3, 2, 3, 1, 4, 7])
                with patch(
                    "speech_to_speech.generation.mixed.decode_token_audio_results",
                    side_effect=_results,
                ):
                    result = generate_responses(
                        [_request(Task.S2ST, FULL_COT)],
                        model,
                        max_new_tokens=10,
                        do_sample=False,
                        use_cache=use_cache,
                    )[0]

                torch.testing.assert_close(result["response_ids"], expected)

    def test_full_s2tt_does_not_stop_at_the_first_eos(self) -> None:
        model = _ScriptModel([1, 3, 2, 3])
        with patch(
            "speech_to_speech.generation.mixed.decode_token_audio_results",
            side_effect=_results,
        ):
            result = generate_responses(
                [_request(Task.S2TT, FULL_COT)],
                model,
                max_new_tokens=8,
                do_sample=False,
            )[0]

        torch.testing.assert_close(
            result["response_ids"],
            torch.tensor([1, 3, 2, 3]),
        )

    def test_same_prediction_with_different_traces_is_grouped_separately(self) -> None:
        model = _ScriptModel(
            [
                2,
                3,
                1,
                4,
                7,
                1,
                3,
                2,
                3,
                1,
                5,
                7,
            ]
        )
        requests = [
            _request(Task.S2ST, TARGET_COT),
            _request(Task.S2ST, FULL_COT),
        ]
        with patch(
            "speech_to_speech.generation.mixed.decode_token_audio_results",
            side_effect=_results,
        ):
            results = generate_responses(
                requests,
                model,
                max_new_tokens=10,
                do_sample=False,
            )

        torch.testing.assert_close(
            results[0]["response_ids"],
            torch.tensor([2, 3, 6, 4, 7]),
        )
        torch.testing.assert_close(
            results[1]["response_ids"],
            torch.tensor([1, 3, 2, 3, 6, 5, 7]),
        )

    def test_text_steps_decode_without_merging_source_and_target(self) -> None:
        response = resolve_response(
            Task.S2ST,
            prediction=PredictionModality.PARALLEL,
            trace=FULL_COT,
        )
        values = decode_response_text_steps(
            _Runtime(),
            torch.tensor([1, 3, 2, 3, 6, 4, 7]),
            response,
        )

        self.assertEqual(values, ["1", "2", None])


if __name__ == "__main__":
    unittest.main()
