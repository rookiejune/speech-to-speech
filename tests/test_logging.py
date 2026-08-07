from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from anytrain import observation
from anytrain.lightning import GradientComparison, GradientProbe, GradientTarget
from anytrain.module.idspace import Layout

from speech_to_speech.callback import TrainInterval
from speech_to_speech.callback.logging import GradLogger, LossSummary
from speech_to_speech.datamodule.batch import ModelBatch, ModelSample
from speech_to_speech.loss.supervised import TokenLoss
from speech_to_speech.pl_module import Config, SpeechToSpeechModule
from speech_to_speech.task import Task


class LoggingTest(unittest.TestCase):
    def test_total_loss_avoids_per_batch_distributed_synchronization(self):
        objective = Mock()
        outputs = {"loss": torch.tensor(2.0), "token": torch.tensor(2.0)}
        objective.forward.return_value = outputs
        module = SpeechToSpeechModule(Config(), model=Mock(), objective=objective)
        batch = _batch(Task.TTS)

        with patch.object(module, "log") as log:
            result = module.training_step(batch)

        self.assertIs(result, outputs)
        objective.forward.assert_called_once_with(batch, module.model)
        log.assert_called_once_with(
            "loss",
            outputs["loss"],
            prog_bar=True,
            on_step=True,
            sync_dist=False,
        )

    def test_grad_logger_runs_once_for_an_accumulated_global_step(self):
        callback = GradLogger(
            (
                GradientComparison(
                    GradientTarget("token"),
                    GradientTarget("rvq"),
                ),
            ),
            (GradientProbe("parameter", ("weight",)),),
            every_n_steps=2,
        )
        trainer = SimpleNamespace(global_step=2)
        module = SimpleNamespace()

        callback.on_train_batch_start(trainer, module, None, 0)
        first = callback.should_run(trainer, module)
        callback.on_train_batch_start(trainer, module, None, 1)
        second = callback.should_run(trainer, module)

        self.assertTrue(first)
        self.assertFalse(second)

    def test_step_interval_runs_once_per_global_step_during_accumulation(self):
        interval = TrainInterval(every_n_steps=2)
        observed = [interval.should_run(step) for step in (0, 0, 1, 1, 2, 2)]
        self.assertEqual(observed, [True, False, False, False, True, False])

        restored = TrainInterval(every_n_steps=2)
        restored.load_state_dict(interval.state_dict())
        self.assertFalse(restored.should_run(2))

    def test_loss_summary_records_only_the_total_scalar(self):
        callback = LossSummary()
        outputs = {
            "loss": torch.tensor(2.0),
            "token": torch.tensor(1.0),
            "flow_matching": torch.tensor(1.0),
        }

        callback.on_train_batch_end(
            SimpleNamespace(),
            SimpleNamespace(),
            outputs,
            None,
            0,
        )

        self.assertEqual(list(callback.values), ["loss"])

    def test_token_loss_declares_scalar_and_diagnostic_observations(self):
        loss = TokenLoss(Layout(text=(0, 4), audio=(4, 8)))
        specs = observation.registry.specs(loss)

        self.assertEqual([spec.name for spec in specs], ["loss", "diagnostics"])
        self.assertEqual(
            [spec.reduction for spec in specs],
            [observation.Reduction.Mean, observation.Reduction.Mean],
        )


def _batch(task: Task) -> ModelBatch:
    return ModelBatch.from_samples(
        [
            ModelSample.from_sequence(
                torch.tensor([1, 2]),
                torch.tensor([-100, 2]),
                task=task,
            )
        ],
        pad_token_id=0,
    )


if __name__ == "__main__":
    unittest.main()
