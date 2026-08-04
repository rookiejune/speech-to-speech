from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout
from lightning import pytorch as pl
from semantic_acoustic_codec.loss.flow import FlowRuntime
from torch import Tensor, nn

from speech_to_speech.callback.logging.outputs import OutputsLogger
from speech_to_speech.datamodule.types import FusedBatch, ModelBatch, ModelSample
from speech_to_speech.loss.ctc import CTCAlignmentLoss
from speech_to_speech.loss.module import FlowObjective, RVQObjective, TokenObjective
from speech_to_speech.loss.types import LossItem, Outputs
from speech_to_speech.loss.validation import validation_metrics
from speech_to_speech.model.ctc import (
    CTCConfig,
    CTCRoute,
    CTCRouteConfig,
    ObjectiveHiddenOutput,
)
from speech_to_speech.prediction import PredictionModality
from speech_to_speech.task import Task


class _Model(nn.Module):
    def __init__(self, layout: Layout) -> None:
        super().__init__()
        self.layout = layout
        self.hidden_calls = 0
        self.text_logit_calls = 0
        self.requested_ctc_routes: list[frozenset[CTCRoute]] = []
        self.ctc_logit_routes: list[CTCRoute] = []
        self.ctc_decoders = nn.ModuleDict(
            {
                CTCRoute.SOURCE.value: nn.Linear(3, 3, bias=False),
                CTCRoute.TARGET.value: nn.Linear(3, 3, bias=False),
            }
        )
        self.text_head = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            nn.init.eye_(self.ctc_decoder(CTCRoute.SOURCE).weight)
            nn.init.eye_(self.ctc_decoder(CTCRoute.TARGET).weight)
            nn.init.eye_(self.text_head.weight)
        self.text_head.requires_grad_(False)

    def ctc_decoder(self, route: CTCRoute) -> nn.Linear:
        decoder = self.ctc_decoders[route.value]
        if not isinstance(decoder, nn.Linear):
            raise AssertionError("test CTC decoder must be linear")
        return decoder

    @staticmethod
    def _hidden(input_ids: Tensor) -> Tensor:
        values = torch.arange(
            1,
            input_ids.numel() * 3 + 1,
            device=input_ids.device,
            dtype=torch.float32,
        )
        return values.reshape((*input_ids.shape, 3)) / 10

    def token_hidden_states(self, input_ids: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        self.hidden_calls += 1
        return self._hidden(input_ids)

    def objective_hidden_output(
        self,
        input_ids: Tensor,
        *,
        ctc_routes: frozenset[CTCRoute],
        **kwargs: object,
    ) -> ObjectiveHiddenOutput:
        del kwargs
        self.hidden_calls += 1
        self.requested_ctc_routes.append(ctc_routes)
        token = self._hidden(input_ids)
        return ObjectiveHiddenOutput(
            token=token,
            source_ctc=token + 0.1 if CTCRoute.SOURCE in ctc_routes else None,
            target_ctc=token + 0.2 if CTCRoute.TARGET in ctc_routes else None,
        )

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
        return self.text_head(hidden_state)

    def ctc_logits(
        self,
        route: CTCRoute,
        hidden_states: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.ctc_logit_routes.append(route)
        decoded = self.ctc_decoder(route)(hidden_states)
        return self.text_logits(decoded), mask

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
        calls: list[tuple[CTCRoute, Tensor]] = []

        def decode(
            route: CTCRoute,
            states: Tensor,
            mask: Tensor,
        ) -> tuple[Tensor, Tensor]:
            calls.append((route, states.clone()))
            return states, mask

        item = CTCAlignmentLoss(
            0,
            CTCConfig(
                source=CTCRouteConfig(weight=1.0),
                target=CTCRouteConfig(weight=1.0),
            ),
        )(
            hidden,
            source_hidden_states=hidden,
            target_hidden_states=hidden,
            source={
                "token_positions": torch.tensor([[1, 2]]),
                "text_token_ids": torch.tensor([[1]]),
            },
            target={
                "token_positions": torch.tensor([[1, 3]]),
                "text_token_ids": torch.tensor([[2]]),
            },
            decode=decode,
        )

        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][0], CTCRoute.SOURCE)
        self.assertIs(calls[1][0], CTCRoute.TARGET)
        torch.testing.assert_close(calls[0][1], hidden[:, [1, 2]])
        torch.testing.assert_close(calls[1][1], hidden[:, [0, 2]])
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
                    ctc=CTCConfig(source=CTCRouteConfig(weight=1.0)),
                    ctc_blank_token_id=0,
                ),
            ),
            (
                "flow",
                lambda: FlowObjective(
                    layout,
                    cast(FlowRuntime, SimpleNamespace()),
                    ctc=CTCConfig(source=CTCRouteConfig(weight=1.0)),
                    ctc_blank_token_id=0,
                ),
            ),
            (
                "rvq",
                lambda: RVQObjective(
                    layout,
                    ctc=CTCConfig(source=CTCRouteConfig(weight=1.0)),
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
                self.assertEqual(
                    model.requested_ctc_routes,
                    [frozenset({CTCRoute.SOURCE})],
                )
                self.assertEqual(model.ctc_logit_routes, [CTCRoute.SOURCE])
                expected = outputs["token"].weighted_mean(
                    _details(self, outputs["token"])["tokens"]
                ) + outputs["ctc"].weighted_mean(
                    _details(self, outputs["ctc"])["sequences"]
                )
                torch.testing.assert_close(outputs["loss"], expected)

    def test_both_routes_train_distinct_decoders_through_frozen_text_head(self):
        layout = Layout(text=(0, 3), audio=(3, 5))
        model = _Model(layout)
        objective = TokenObjective(
            layout,
            ctc=CTCConfig(
                source=CTCRouteConfig(weight=1.0),
                target=CTCRouteConfig(weight=1.0),
            ),
            ctc_blank_token_id=0,
        )

        outputs = objective(_s2st_batch(), model)
        outputs["loss"].backward()

        self.assertEqual(
            model.requested_ctc_routes,
            [frozenset({CTCRoute.SOURCE, CTCRoute.TARGET})],
        )
        self.assertEqual(
            model.ctc_logit_routes,
            [CTCRoute.SOURCE, CTCRoute.TARGET],
        )
        source_grad = model.ctc_decoder(CTCRoute.SOURCE).weight.grad
        target_grad = model.ctc_decoder(CTCRoute.TARGET).weight.grad
        self.assertIsNotNone(source_grad)
        self.assertIsNotNone(target_grad)
        assert source_grad is not None
        assert target_grad is not None
        self.assertGreater(float(source_grad.abs().sum()), 0.0)
        self.assertGreater(float(target_grad.abs().sum()), 0.0)
        self.assertIsNone(model.text_head.weight.grad)

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
            ctc=CTCConfig(source=CTCRouteConfig(weight=1.0)),
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
            ctc=CTCConfig(target=CTCRouteConfig(weight=1.0)),
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
        self.assertEqual(
            t2st_model.requested_ctc_routes,
            [frozenset({CTCRoute.TARGET})],
        )
        self.assertEqual(t2st_model.ctc_logit_routes, [CTCRoute.TARGET])
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
            ctc=CTCConfig(target=CTCRouteConfig(weight=1.0)),
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
            cast(pl.Trainer, SimpleNamespace(world_size=1)),
            cast(pl.LightningModule, module),
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
            cast(pl.Trainer, SimpleNamespace(world_size=1)),
            cast(pl.LightningModule, module),
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


def _s2st_batch() -> ModelBatch:
    return ModelBatch.from_samples(
        [
            ModelSample.from_sequence(
                torch.tensor([3, 4, 3, 4]),
                torch.tensor([-100, -100, 3, 4]),
                task=Task.S2ST,
                prediction=PredictionModality.AUDIO,
                source_ctc={
                    "token_positions": torch.tensor([0, 1]),
                    "text_token_ids": torch.tensor([1]),
                },
                target_ctc={
                    "token_positions": torch.tensor([2, 3]),
                    "text_token_ids": torch.tensor([2]),
                },
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
