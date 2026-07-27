from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from speech_to_speech.callback.logging import (
    FlowMatchingLogger,
    GradNormLogger,
    LossSummary,
    OutputsLogger,
    ValidationSummary,
)
from speech_to_speech.callback import TrainInterval, processed_audio_seconds
from speech_to_speech.datamodule.types import ModelBatch, ModelSample
from speech_to_speech.loss import LossItem, Outputs, loss_items
from speech_to_speech.pl_module import Config, SpeechToSpeechModule
from speech_to_speech.task import Task


class LoggingTest(unittest.TestCase):
    def test_total_loss_is_synchronized_across_ranks(self):
        objective = Mock()
        outputs = Outputs(loss=torch.tensor(2.0))
        objective.forward.return_value = outputs
        objective.reduce.side_effect = lambda values: values[0]
        module = SpeechToSpeechModule(Config(), model=Mock(), objective=objective)
        batch = _batch(Task.TTS)

        with patch.object(module, "log") as log:
            result = module.training_step(batch)

        self.assertIs(result, outputs)
        objective.reduce.assert_called_once_with([outputs])
        log.assert_called_once_with(
            "train/loss",
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

        callback.on_before_optimizer_step(trainer, module, Mock())
        module.log.assert_not_called()

        trainer.global_step = 2
        callback.on_before_optimizer_step(trainer, module, Mock())

        self.assertEqual(module.log.call_args.args[0], "train/grad_norm")
        torch.testing.assert_close(module.log.call_args.args[1], torch.tensor(13.0))

    def test_audio_interval_counts_processed_batch_seconds(self):
        interval = TrainInterval(every_n_steps=100, every_audio_seconds=2.0)
        trainer = SimpleNamespace(global_step=0, world_size=1)
        module = SimpleNamespace(device=torch.device("cpu"))
        first = _batch(Task.TTS, audio_seconds=1.5)
        second = _batch(Task.TTS, audio_seconds=0.75)

        self.assertFalse(interval.should_run(trainer, module, first))
        self.assertTrue(interval.should_run(trainer, module, second))
        self.assertEqual(processed_audio_seconds((first, second)), 2.25)

    def test_audio_interval_sums_distributed_rank_seconds(self):
        reduce_ops = []

        class Strategy:
            def reduce(self, value, *, reduce_op):
                reduce_ops.append(reduce_op)
                return value + value.new_tensor(2.0)

        interval = TrainInterval(every_n_steps=100, every_audio_seconds=3.0)
        trainer = SimpleNamespace(global_step=0, world_size=2, strategy=Strategy())
        module = SimpleNamespace(device=torch.device("cpu"))

        self.assertTrue(interval.should_run(trainer, module, _batch(Task.TTS, 1.0)))
        self.assertEqual(reduce_ops, ["sum"])

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

    def test_validation_summary_separates_sanity_and_interval_metrics(self):
        callback = ValidationSummary()
        trainer = SimpleNamespace(
            callback_metrics={
                "train/loss": torch.tensor(5.0),
                "val/token_ce": torch.tensor(2.0),
                "val/rvq_ce": torch.tensor(3.0),
            },
            global_step=0,
            sanity_checking=True,
        )

        callback.on_validation_end(trainer, SimpleNamespace())
        trainer.callback_metrics = {
            "val/token_ce": torch.tensor(1.5),
            "val/rvq_ce": torch.tensor(2.5),
        }
        trainer.global_step = 1000
        trainer.sanity_checking = False
        callback.on_validation_end(trainer, SimpleNamespace())

        report = callback.report()
        self.assertEqual(
            report,
            {
                "sanity": [
                    {
                        "step": 0,
                        "metrics": {
                            "val/rvq_ce": 3.0,
                            "val/token_ce": 2.0,
                        },
                    }
                ],
                "interval": [
                    {
                        "step": 1000,
                        "metrics": {
                            "val/rvq_ce": 2.5,
                            "val/token_ce": 1.5,
                        },
                    }
                ],
            },
        )
        json.dumps(report, allow_nan=False)

    def test_validation_summary_rejects_non_scalar_metrics(self):
        callback = ValidationSummary()
        trainer = SimpleNamespace(
            callback_metrics={"val/token_ce": torch.tensor([1.0, 2.0])},
            global_step=0,
            sanity_checking=False,
        )

        with self.assertRaisesRegex(ValueError, "val/token_ce.*scalar"):
            callback.on_validation_end(trainer, SimpleNamespace())

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
                ("flow/loss_t/0.00_0.50", 2.0, 2),
                ("flow/loss_t/0.50_1.00", 6.0, 2),
            ],
        )
        self.assertEqual(strategy.reduce.call_count, 2)

    def test_outputs_logger_accepts_tuple_train_batches(self):
        module = SimpleNamespace(log=Mock())
        trainer = SimpleNamespace()
        callback = OutputsLogger()
        batch = (
            _batch(Task.ASR),
            _batch(Task.MT),
        )
        outputs = Outputs(
            loss=torch.tensor(2.0),
            token=LossItem(
                torch.tensor([1.0, 3.0]),
                details={"tokens": torch.tensor([4.0, 8.0])},
            ),
        )

        callback.on_train_batch_end(trainer, module, outputs, batch, 0)

        names = [call.args[0] for call in module.log.call_args_list]
        self.assertEqual(
            names,
            [
                "token_loss/asr",
                "token_tokens/asr",
                "token_loss/mt",
                "token_tokens/mt",
            ],
        )

    def test_outputs_logger_uses_acoustic_tasks_for_acoustic_losses(self):
        module = SimpleNamespace(log=Mock())
        trainer = SimpleNamespace()
        callback = OutputsLogger()
        batch = (
            _batch(Task.MT),
            _acoustic_batch(Task.TTS),
        )
        outputs = Outputs(
            loss=torch.tensor(2.0),
            token=LossItem(
                torch.tensor([1.0, 3.0]),
                details={"tokens": torch.tensor([4.0, 8.0])},
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
                "token_loss/mt",
                "token_tokens/mt",
                "token_loss/tts",
                "token_tokens/tts",
                "rvq_loss/tts",
                "rvq_frames/tts",
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
            )
        ],
        pad_token_id=0,
    )
