from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from speech_to_speech.callback.logging import FlowMatchingLogger, GradNormLogger
from speech_to_speech.loss import LossItem, Outputs, loss_items
from speech_to_speech.reporting import window_summary


class LoggingTest(unittest.TestCase):
    def test_grad_norm_logger_respects_its_interval(self):
        weight = torch.nn.Parameter(torch.zeros(2))
        bias = torch.nn.Parameter(torch.zeros(1))
        weight.grad = torch.tensor([3.0, 4.0])
        bias.grad = torch.tensor([12.0])
        module = SimpleNamespace(
            parameters=lambda: iter((weight, bias)),
            log=Mock(),
        )
        trainer = SimpleNamespace(global_step=1)
        callback = GradNormLogger(every_n_steps=2)

        callback.on_before_optimizer_step(trainer, module, Mock())
        module.log.assert_not_called()

        trainer.global_step = 2
        callback.on_before_optimizer_step(trainer, module, Mock())

        self.assertEqual(module.log.call_args.args[0], "train/grad_norm")
        torch.testing.assert_close(module.log.call_args.args[1], torch.tensor(13.0))

    def test_window_summary_reports_edges_and_window_means(self):
        summary = window_summary([1.0, 2.0, 4.0], window=2)

        self.assertEqual(summary["first"], 1.0)
        self.assertEqual(summary["last"], 4.0)
        self.assertEqual(summary["first_mean"], 1.5)
        self.assertEqual(summary["last_mean"], 3.0)
        self.assertEqual(summary["last_to_first"], 2.0)

    def test_window_summary_exposes_zero_baseline(self):
        summary = window_summary([0.0, 0.0])

        self.assertIsNone(summary["last_to_first"])

    def test_flow_logger_uses_injected_runtime(self):
        experiment = Mock()
        trainer = SimpleNamespace(
            logger=SimpleNamespace(experiment=experiment),
        )
        runtime = SimpleNamespace(
            time_sampler=SimpleNamespace(mean=0.0, std=1.0),
        )

        FlowMatchingLogger(runtime).on_fit_start(trainer, SimpleNamespace())

        text = experiment.add_text.call_args.args[1]
        self.assertIn("sampler=SimpleNamespace", text)
        self.assertIn("mean=0.0", text)

    def test_loss_items_use_stable_objective_order(self):
        item = LossItem(torch.ones(1), details=None)
        outputs = Outputs(
            loss=torch.ones(()),
            flow_matching=item,
            semantic=item,
        )

        self.assertEqual(
            [name for name, _ in loss_items(outputs)],
            ["semantic", "flow_matching"],
        )


if __name__ == "__main__":
    unittest.main()
