from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout
from torch import Tensor

from speech_to_speech.callback.logging.outputs import OutputsLogger
from speech_to_speech.datamodule.types import FusedBatch, ModelBatch, ModelSample
from speech_to_speech.loss.ctc import CTCAlignmentLoss, CTCConfig
from speech_to_speech.loss.module import FlowObjective, RVQObjective, TokenObjective
from speech_to_speech.loss.types import LossItem, Outputs
from speech_to_speech.loss.validation import validation_metrics
from speech_to_speech.prediction import PredictionModality
from speech_to_speech.task import Task


class _Model:
    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.hidden_calls = 0
        self.text_logit_calls = 0

    def token_hidden_states(self, input_ids: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        self.hidden_calls += 1
        return torch.zeros((*input_ids.shape, 3), dtype=torch.float32)

    def token_logits(
        self,
        hidden_state: Tensor,
        modality: Modality | None = None,
        **kwargs: object,
    ) -> Tensor:
        del kwargs
        if modality is None:
            raise ValueError("test model requires an explicit token modality")
        start, end = self.layout.blocks[modality.value]
        return hidden_state.new_zeros((hidden_state.size(0), end - start))

    def text_logits(
        self,
        hidden_state: Tensor,
        local_ids: Tensor | None = None,
    ) -> Tensor:
        if local_ids is not None:
            raise AssertionError("CTC must request the complete frozen text head")
        self.text_logit_calls += 1
        return hidden_state

    def project_audio_hidden(
        self,
        hidden_state: Tensor,
        *,
        attention_mask: Tensor | None = None,
        selection_mask: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, object | None]:
        del attention_mask, past_key_values, use_cache
        if selection_mask is None:
            return hidden_state, None
        return hidden_state[selection_mask], None

    def audio_neighbor_targets(self, local_ids: Tensor) -> None:
        del local_ids
        raise AssertionError("neighbor targets are disabled in CTC tests")


class CTCLossChainTest(unittest.TestCase):
    def test_source_and_target_routes_select_noncausal_and_causal_states(self):
        hidden = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        calls: list[Tensor] = []

        def readout(states: Tensor) -> Tensor:
            calls.append(states.clone())
            return states

        item = CTCAlignmentLoss(
            0,
            CTCConfig(source_weight=1.0, target_weight=1.0),
        )(
            hidden,
            source={
                "token_positions": torch.tensor([[1, 2]]),
                "text_token_ids": torch.tensor([[1]]),
            },
            target={
                "token_positions": torch.tensor([[1, 3]]),
                "text_token_ids": torch.tensor([[2]]),
            },
            text_readout=readout,
        )

        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(calls[0], hidden[:, [1, 2]])
        torch.testing.assert_close(calls[1], hidden[:, [0, 2]])
        self.assertTrue(torch.isfinite(item.loss).all())
        details = _details(self, item)
        torch.testing.assert_close(details["sequences"], torch.ones(1))
        torch.testing.assert_close(details["source_tokens"], torch.ones(1))
        torch.testing.assert_close(details["target_tokens"], torch.ones(1))

    def test_all_token_objectives_reuse_hidden_states_for_ctc(self):
        layout = Layout(text=(0, 3), audio=(3, 5))
        factories = (
            (
                "token",
                lambda: TokenObjective(
                    layout,
                    ctc=CTCConfig(source_weight=1.0),
                    ctc_blank_token_id=0,
                ),
            ),
            (
                "flow",
                lambda: FlowObjective(
                    layout,
                    SimpleNamespace(),
                    ctc=CTCConfig(source_weight=1.0),
                    ctc_blank_token_id=0,
                ),
            ),
            (
                "rvq",
                lambda: RVQObjective(
                    layout,
                    ctc=CTCConfig(source_weight=1.0),
                    ctc_blank_token_id=0,
                ),
            ),
        )

        for name, factory in factories:
            with self.subTest(objective=name):
                model = _Model(layout)
                outputs = factory()(_source_batch(), model)

                self.assertIn("ctc", outputs)
                self.assertEqual(model.hidden_calls, 1)
                self.assertEqual(model.text_logit_calls, 1)
                expected = outputs["token"].weighted_mean(
                    _details(self, outputs["token"])["tokens"]
                ) + outputs["ctc"].weighted_mean(
                    _details(self, outputs["ctc"])["sequences"]
                )
                torch.testing.assert_close(outputs["loss"], expected)

    def test_ctc_total_uses_active_sequence_count(self):
        layout = Layout(text=(0, 3), audio=(3, 5))
        model = _Model(layout)
        batch = ModelBatch(
            input_ids=torch.tensor([[3, 4, 1], [3, 4, 1]]),
            token_labels=torch.tensor([[-100, -100, 1], [-100, -100, 1]]),
            acoustic_target=None,
            tasks=[Task.ASR, Task.ASR],
            predictions=[PredictionModality.TEXT, PredictionModality.TEXT],
            pad_token_id=99,
            source_ctc={
                "token_positions": torch.tensor([[0, 1], [-1, -1]]),
                "text_token_ids": torch.tensor([[1], [-1]]),
            },
        )
        objective = TokenObjective(
            layout,
            ctc=CTCConfig(source_weight=1.0),
            ctc_blank_token_id=0,
        )

        outputs = objective(batch, model)

        ctc = outputs["ctc"]
        details = _details(self, ctc)
        torch.testing.assert_close(details["sequences"], torch.tensor([1.0, 0.0]))
        token_mean = outputs["token"].weighted_mean(
            _details(self, outputs["token"])["tokens"]
        )
        torch.testing.assert_close(outputs["loss"] - token_mean, ctc.loss[0])

    def test_tts_is_excluded_but_t2st_uses_target_ctc(self):
        layout = Layout(text=(0, 3), audio=(3, 5))
        objective = TokenObjective(
            layout,
            ctc=CTCConfig(target_weight=1.0),
            ctc_blank_token_id=0,
        )
        tts_model = _Model(layout)
        tts = ModelBatch(
            input_ids=torch.tensor([[0, 3, 4]]),
            token_labels=torch.tensor([[-100, 3, 4]]),
            acoustic_target=None,
            tasks=[Task.TTS],
            predictions=[PredictionModality.AUDIO],
            pad_token_id=99,
        )

        tts_outputs = objective(tts, tts_model)

        self.assertNotIn("ctc", tts_outputs)
        self.assertEqual(tts_model.text_logit_calls, 0)

        t2st_model = _Model(layout)
        t2st = ModelBatch(
            input_ids=torch.tensor([[0, 3, 4]]),
            token_labels=torch.tensor([[-100, 3, 4]]),
            acoustic_target=None,
            tasks=[Task.T2ST],
            predictions=[PredictionModality.AUDIO],
            pad_token_id=99,
            target_ctc={
                "token_positions": torch.tensor([[1, 2]]),
                "text_token_ids": torch.tensor([[1]]),
            },
        )

        t2st_outputs = objective(t2st, t2st_model)

        self.assertIn("ctc", t2st_outputs)
        self.assertEqual(t2st_model.text_logit_calls, 1)
        details = _details(self, t2st_outputs["ctc"])
        torch.testing.assert_close(details["source_tokens"], torch.zeros(1))
        torch.testing.assert_close(details["target_tokens"], torch.ones(1))

    def test_all_padding_ctc_row_adds_zero_without_nan(self):
        layout = Layout(text=(0, 3), audio=(3, 5))
        batch = _mixed_target_batch().row(0)
        target = batch.target_ctc
        if target is None:
            self.fail("mixed-batch TTS row lost its aggregate CTC field")
        self.assertTrue(target["token_positions"].eq(-1).all())
        self.assertTrue(target["text_token_ids"].eq(-1).all())
        objective = TokenObjective(
            layout,
            ctc=CTCConfig(target_weight=1.0),
            ctc_blank_token_id=0,
        )

        outputs = objective(batch, _Model(layout))

        token_mean = outputs["token"].weighted_mean(
            _details(self, outputs["token"])["tokens"]
        )
        self.assertTrue(torch.isfinite(outputs["loss"]))
        torch.testing.assert_close(outputs["loss"], token_mean)
        ctc = outputs["ctc"]
        torch.testing.assert_close(_details(self, ctc)["sequences"], torch.zeros(1))
        torch.testing.assert_close(ctc.loss, torch.zeros(1))

    def test_ctc_validation_and_task_logging_use_alignment_namespace(self):
        item = LossItem(
            torch.tensor([2.0]),
            {
                "source_loss": torch.tensor([2.0]),
                "target_loss": torch.tensor([0.0]),
                "source_tokens": torch.tensor([1.0]),
                "target_tokens": torch.tensor([0.0]),
                "source_steps": torch.tensor([2.0]),
                "target_steps": torch.tensor([0.0]),
                "tokens": torch.tensor([1.0]),
                "sequences": torch.tensor([1.0]),
            },
        )
        metrics = validation_metrics({"loss": torch.tensor(2.0), "ctc": item})
        self.assertEqual(set(metrics), {"alignment/ctc/loss"})
        torch.testing.assert_close(
            metrics["alignment/ctc/loss"].weights,
            torch.ones(1),
        )

        callback = OutputsLogger()
        module = SimpleNamespace(log=Mock())
        callback.on_train_batch_end(
            SimpleNamespace(world_size=1),
            module,
            Outputs(loss=torch.tensor(2.0), ctc=item),
            FusedBatch((_source_batch(), _mt_batch())),
            0,
        )

        values = {call.args[0]: call.args[1] for call in module.log.call_args_list}
        self.assertIn("alignment/ctc/loss/asr", values)
        self.assertEqual(values["alignment/ctc/source_tokens/asr"], 1.0)
        self.assertEqual(values["alignment/ctc/source_steps/asr"], 2.0)
        self.assertEqual(values["alignment/ctc/sequences/asr"], 1.0)
        self.assertFalse(any(name.endswith("/mt") for name in values))

    def test_ctc_logging_masks_inactive_rows_and_unused_routes(self):
        item = LossItem(
            torch.tensor([0.0, 2.0]),
            {
                "source_loss": torch.tensor([0.0, 0.0]),
                "target_loss": torch.tensor([0.0, 2.0]),
                "source_tokens": torch.tensor([0.0, 0.0]),
                "target_tokens": torch.tensor([0.0, 1.0]),
                "source_steps": torch.tensor([0.0, 0.0]),
                "target_steps": torch.tensor([0.0, 2.0]),
                "tokens": torch.tensor([0.0, 1.0]),
                "sequences": torch.tensor([0.0, 1.0]),
            },
        )
        callback = OutputsLogger()
        module = SimpleNamespace(log=Mock())

        callback.on_train_batch_end(
            SimpleNamespace(world_size=1),
            module,
            Outputs(loss=torch.tensor(2.0), ctc=item),
            _mixed_target_batch(),
            0,
        )

        values = {call.args[0]: call.args[1] for call in module.log.call_args_list}
        self.assertEqual(values["alignment/ctc/loss/t2st"], 2.0)
        self.assertEqual(values["alignment/ctc/target_loss/t2st"], 2.0)
        self.assertEqual(values["alignment/ctc/target_tokens/t2st"], 1.0)
        self.assertFalse(any(name.endswith("/tts") for name in values))
        self.assertFalse(any("source_" in name for name in values))


def _source_batch() -> ModelBatch:
    return ModelBatch.from_samples(
        [
            ModelSample.from_sequence(
                torch.tensor([3, 4, 1]),
                torch.tensor([-100, -100, 1]),
                task=Task.ASR,
                prediction=PredictionModality.TEXT,
                source_ctc={
                    "token_positions": torch.tensor([0, 1]),
                    "text_token_ids": torch.tensor([1]),
                },
            )
        ],
        pad_token_id=99,
    )


def _mt_batch() -> ModelBatch:
    return ModelBatch.from_samples(
        [
            ModelSample.from_sequence(
                torch.tensor([0, 1]),
                torch.tensor([-100, 1]),
                task=Task.MT,
                prediction=PredictionModality.TEXT,
            )
        ],
        pad_token_id=99,
    )


def _mixed_target_batch() -> ModelBatch:
    return ModelBatch.from_samples(
        [
            ModelSample.from_sequence(
                torch.tensor([0, 3, 4]),
                torch.tensor([-100, 3, 4]),
                task=Task.TTS,
                prediction=PredictionModality.AUDIO,
            ),
            ModelSample.from_sequence(
                torch.tensor([0, 3, 4]),
                torch.tensor([-100, 3, 4]),
                task=Task.T2ST,
                prediction=PredictionModality.AUDIO,
                target_ctc={
                    "token_positions": torch.tensor([1, 2]),
                    "text_token_ids": torch.tensor([1]),
                },
            ),
        ],
        pad_token_id=99,
    )


def _details(test: unittest.TestCase, item: LossItem) -> dict[str, Tensor]:
    details = item.details
    if details is None:
        test.fail("loss details are unavailable")
    return details


if __name__ == "__main__":
    unittest.main()
