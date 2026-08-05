from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from anytrain.lightning import GradientComparison, GradientProbe, GradientTarget

from speech_to_speech.callback.logging import (
    FlowMatchingLogger,
    GradLogger,
    LossSummary,
    OutputsLogger,
)
from speech_to_speech.callback import TrainInterval
from speech_to_speech.datamodule.batch import (
    FusedBatch,
    ModelBatch,
    ModelSample,
)
from speech_to_speech.loss.contract import LossItem, Outputs, loss_items
from speech_to_speech.pl_module import Config, SpeechToSpeechModule
from speech_to_speech.task import Task


class LoggingTest(unittest.TestCase):
    def test_total_loss_avoids_per_batch_distributed_synchronization(self):
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

    def test_outputs_logger_uses_fused_microbatch_tasks(self):
        module = SimpleNamespace(log=Mock())
        trainer = SimpleNamespace(world_size=1)
        callback = OutputsLogger()
        batch = FusedBatch((_batch(Task.ASR), _batch(Task.MT)))
        outputs = Outputs(
            loss=torch.tensor(2.0),
            token=LossItem(
                torch.tensor([1.0, 3.0]),
                details={"tokens": torch.tensor([1.0, 3.0])},
            ),
        )

        callback.on_train_batch_end(trainer, module, outputs, batch, 0)

        names = [call.args[0] for call in module.log.call_args_list]
        self.assertEqual(
            names,
            [
                "token/loss/asr",
                "token/tokens/asr",
                "token/loss/mt",
                "token/tokens/mt",
            ],
        )

    def test_outputs_logger_uses_distributed_weighted_means(self):
        module = SimpleNamespace(log=Mock())
        strategy = Mock()
        strategy.reduce.return_value = torch.tensor(
            [
                [4.0, 2.0],
                [40.0, 2.0],
                [5.0, 1.0],
                [7.0, 1.0],
            ]
        )
        trainer = SimpleNamespace(world_size=2, strategy=strategy)
        callback = OutputsLogger()
        batch = _batch(Task.ASR)
        outputs = Outputs(
            loss=torch.tensor(2.0),
            token=LossItem(
                torch.tensor([1.0]),
                details={"quality": torch.tensor([10.0])},
            ),
        )

        def gather(output: list[object], value: object) -> None:
            output[:] = [
                value,
                [
                    ("token/loss/mt", False),
                    ("token/quality/mt", False),
                ],
            ]

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_gather_object", side_effect=gather),
        ):
            callback.on_train_batch_end(trainer, module, outputs, batch, 0)

        values = {
            call.args[0]: float(call.args[1]) for call in module.log.call_args_list
        }
        self.assertEqual(
            values,
            {
                "token/loss/asr": 2.0,
                "token/quality/asr": 20.0,
                "token/loss/mt": 5.0,
                "token/quality/mt": 7.0,
            },
        )
        self.assertTrue(
            all(call.kwargs["sync_dist"] is False for call in module.log.call_args_list)
        )
        strategy.reduce.assert_called_once()

    def test_outputs_logger_reduces_once_at_trainer_log_cadence(self):
        module = SimpleNamespace(log=Mock())
        trainer = SimpleNamespace(
            world_size=1,
            global_step=1,
            log_every_n_steps=2,
        )
        callback = OutputsLogger()
        batch = _batch(Task.MT)

        callback.on_train_batch_end(
            trainer,
            module,
            Outputs(
                loss=torch.tensor(1.0),
                token=LossItem(
                    torch.tensor([1.0]),
                    details={"tokens": torch.tensor([2.0])},
                ),
            ),
            batch,
            0,
        )
        module.log.assert_not_called()

        trainer.global_step = 2
        callback.on_train_batch_end(
            trainer,
            module,
            Outputs(
                loss=torch.tensor(3.0),
                token=LossItem(
                    torch.tensor([3.0]),
                    details={"tokens": torch.tensor([4.0])},
                ),
            ),
            batch,
            1,
        )

        values = {call.args[0]: call.args[1] for call in module.log.call_args_list}
        self.assertEqual(values["token/loss/mt"], 2.0)
        self.assertEqual(values["token/tokens/mt"], 6.0)

    def test_outputs_logger_restores_pending_cadence_window(self):
        trainer = SimpleNamespace(
            world_size=1,
            global_step=1,
            log_every_n_steps=2,
        )
        batch = _batch(Task.MT)
        first = OutputsLogger()
        first.on_train_batch_end(
            trainer,
            SimpleNamespace(log=Mock()),
            Outputs(
                loss=torch.tensor(1.0),
                token=LossItem(
                    torch.tensor([1.0]),
                    details={"tokens": torch.tensor([2.0])},
                ),
            ),
            batch,
            0,
        )

        restored = OutputsLogger()
        restored.load_state_dict(first.state_dict())
        module = SimpleNamespace(log=Mock())
        trainer.global_step = 2
        restored.on_train_batch_end(
            trainer,
            module,
            Outputs(
                loss=torch.tensor(3.0),
                token=LossItem(
                    torch.tensor([3.0]),
                    details={"tokens": torch.tensor([4.0])},
                ),
            ),
            batch,
            1,
        )

        values = {call.args[0]: call.args[1] for call in module.log.call_args_list}
        self.assertEqual(values["token/loss/mt"], 2.0)
        self.assertEqual(values["token/tokens/mt"], 6.0)

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
                ModelSample.from_sequence(
                    torch.tensor([1, 2]),
                    torch.tensor([-100, 2]),
                    task=Task.TTS,
                ),
                ModelSample.from_sequence(
                    torch.tensor([1, 2, 3]),
                    torch.tensor([-100, 2, 3]),
                    task=Task.TTS,
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
            ModelSample.from_sequence(
                torch.tensor([1, 2]),
                torch.tensor([-100, 2]),
                task=task,
                audio_seconds=audio_seconds,
            )
        ],
        pad_token_id=0,
    )


def _acoustic_batch(task: Task) -> ModelBatch:
    return ModelBatch.from_samples(
        [
            ModelSample.from_sequence(
                torch.tensor([1, 2]),
                torch.tensor([-100, 2]),
                acoustic_target={
                    "semantic_codes": torch.tensor([[1]]),
                    "codes": torch.tensor([[2]]),
                    "token_positions": torch.tensor([1]),
                },
                task=task,
            )
        ],
        pad_token_id=0,
    )
