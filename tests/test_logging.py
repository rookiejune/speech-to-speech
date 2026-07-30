from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from speech_to_speech.callback.logging import (
    FlowMatchingLogger,
    GradLogger,
    GradNormLogger,
    LossSummary,
    OutputsLogger,
)
from speech_to_speech.callback import TrainInterval
from speech_to_speech.datamodule.types import ModelBatch, ModelSample
from speech_to_speech.loss import LossItem, Outputs, loss_items
from speech_to_speech.pl_module import Config, SpeechToSpeechModule
from speech_to_speech.task import Task


class LoggingTest(unittest.TestCase):
    def test_total_loss_is_synchronized_across_ranks(self):
        objective = Mock()
        outputs = Outputs(loss=torch.tensor(2.0))
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
            sync_dist=True,
        )

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

        callback.on_train_batch_start(trainer, module, None, 0)
        callback.on_before_optimizer_step(trainer, module, Mock())
        module.log.assert_not_called()

        trainer.global_step = 2
        callback.on_train_batch_start(trainer, module, None, 0)
        callback.on_before_optimizer_step(trainer, module, Mock())

        self.assertEqual(module.log.call_args.args[0], "grad_norm")
        torch.testing.assert_close(module.log.call_args.args[1], torch.tensor(13.0))

    def test_grad_logger_runs_once_for_an_accumulated_global_step(self):
        callback = GradLogger(("token", "rvq"), "weight", every_n_steps=2)
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

        observed = []
        for step in (0, 0, 1, 1, 2, 2):
            observed.append(interval.should_run(step))

        self.assertEqual(observed, [True, False, False, False, True, False])

        restored = TrainInterval(every_n_steps=2)
        restored.load_state_dict(interval.state_dict())
        self.assertFalse(restored.should_run(2))

    def test_loss_summary_uses_stable_objective_order(self):
        item = LossItem(torch.ones(1), details=None)
        callback = LossSummary()
        outputs = Outputs(
            loss=torch.tensor(2.0),
            flow_matching=item,
            token=item,
        )

        callback.on_train_batch_end(
            SimpleNamespace(),
            SimpleNamespace(),
            outputs,
            None,
            0,
        )

        self.assertEqual(list(callback.values), ["loss", "token", "flow_matching"])

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

    def test_flow_logger_records_bucketed_loss_by_time(self):
        experiment = Mock()
        strategy = Mock()
        strategy.reduce.side_effect = lambda value, *, reduce_op: value
        trainer = SimpleNamespace(
            global_step=2,
            is_global_zero=True,
            logger=SimpleNamespace(experiment=experiment),
            strategy=strategy,
        )
        runtime = SimpleNamespace(
            time_sampler=SimpleNamespace(t_min=0.0, t_max=1.0),
        )
        flow = LossItem(
            loss=torch.tensor([1.0, 3.0, 5.0, 7.0]),
            details={"t": torch.tensor([0.0, 0.2, 0.6, 1.0])},
        )

        callback = FlowMatchingLogger(
            runtime,
            every_n_steps=1,
            time_bucket_count=2,
        )

        callback.on_train_batch_end(
            trainer,
            SimpleNamespace(),
            {"flow_matching": flow},
            None,
            0,
        )

        scalar_calls = [
            (call.args[0], call.args[1], call.args[2])
            for call in experiment.add_scalar.call_args_list
        ]
        self.assertEqual(
            scalar_calls,
            [
                ("acoustic/flow_matching/loss_t/0.00_0.50", 2.0, 2),
                ("acoustic/flow_matching/loss_t/0.50_1.00", 6.0, 2),
            ],
        )
        self.assertEqual(strategy.reduce.call_count, 2)

    def test_outputs_logger_uses_homogeneous_microbatch_tasks(self):
        module = SimpleNamespace(log=Mock())
        trainer = SimpleNamespace(world_size=1)
        callback = OutputsLogger()
        batch = _batch(Task.MT)
        outputs = Outputs(
            loss=torch.tensor(2.0),
            token=LossItem(
                torch.tensor([3.0]),
                details={"tokens": torch.tensor([8.0])},
            ),
        )

        callback.on_train_batch_end(trainer, module, outputs, batch, 0)

        names = [call.args[0] for call in module.log.call_args_list]
        self.assertEqual(
            names,
            [
                "token/loss/mt",
                "token/tokens/mt",
            ],
        )
        token_call = module.log.call_args_list[1]
        self.assertEqual(token_call.args[1], 8.0)

    def test_outputs_logger_accumulates_token_counts(self):
        module = SimpleNamespace(log=Mock())
        trainer = SimpleNamespace(world_size=1)
        callback = OutputsLogger()
        batch = _batch(Task.TTS)
        outputs = Outputs(
            loss=torch.tensor(2.0),
            token=LossItem(
                torch.tensor([3.0, 4.0]),
                details={"tokens": torch.tensor([8.0, 2.0])},
            ),
        )
        # Homogeneous batch helper only has one row; build two-row batch.
        batch = ModelBatch.from_samples(
            [
                ModelSample(
                    input_ids=torch.tensor([1, 2]),
                    token_labels=torch.tensor([-100, 2]),
                    acoustic_target=None,
                    task=Task.TTS,
                    prediction=Task.TTS.prediction_modality,
                ),
                ModelSample(
                    input_ids=torch.tensor([1, 2, 3]),
                    token_labels=torch.tensor([-100, 2, 3]),
                    acoustic_target=None,
                    task=Task.TTS,
                    prediction=Task.TTS.prediction_modality,
                ),
            ],
            pad_token_id=0,
        )

        callback.on_train_batch_end(trainer, module, outputs, batch, 0)
        callback.on_train_batch_end(trainer, module, outputs, batch, 1)

        token_values = [
            call.args[1]
            for call in module.log.call_args_list
            if call.args[0] == "token/tokens/tts"
        ]
        self.assertEqual(token_values, [10.0, 20.0])

    def test_outputs_logger_uses_acoustic_tasks_for_acoustic_losses(self):
        module = SimpleNamespace(log=Mock())
        trainer = SimpleNamespace(world_size=1)
        callback = OutputsLogger()
        batch = _acoustic_batch(Task.TTS)
        outputs = Outputs(
            loss=torch.tensor(2.0),
            token=LossItem(
                torch.tensor([3.0]),
                details={"tokens": torch.tensor([8.0])},
            ),
            rvq=LossItem(
                torch.tensor([5.0]),
                details={"frames": torch.tensor([9.0])},
            ),
        )

        callback.on_train_batch_end(trainer, module, outputs, batch, 0)

        names = [call.args[0] for call in module.log.call_args_list]
        self.assertEqual(
            names,
            [
                "token/loss/tts",
                "token/tokens/tts",
                "acoustic/rvq/loss/tts",
                "acoustic/rvq/frames/tts",
            ],
        )

    def test_loss_items_use_stable_objective_order(self):
        item = LossItem(torch.ones(1), details=None)
        outputs = Outputs(
            loss=torch.ones(()),
            flow_matching=item,
            token=item,
        )

        self.assertEqual(
            [name for name, _ in loss_items(outputs)],
            ["token", "flow_matching"],
        )


if __name__ == "__main__":
    unittest.main()


def _batch(task: Task, audio_seconds: float = 0.0) -> ModelBatch:
    return ModelBatch.from_samples(
        [
            ModelSample(
                input_ids=torch.tensor([1, 2]),
                token_labels=torch.tensor([-100, 2]),
                acoustic_target=None,
                task=task,
                prediction=task.prediction_modality,
                audio_seconds=audio_seconds,
            )
        ],
        pad_token_id=0,
    )


def _acoustic_batch(task: Task) -> ModelBatch:
    return ModelBatch.from_samples(
        [
            ModelSample(
                input_ids=torch.tensor([1, 2]),
                token_labels=torch.tensor([-100, 2]),
                acoustic_target={
                    "semantic_codes": torch.tensor([[1]]),
                    "codes": torch.tensor([[2]]),
                    "token_positions": torch.tensor([1]),
                },
                task=task,
                prediction=task.prediction_modality,
            )
        ],
        pad_token_id=0,
    )
