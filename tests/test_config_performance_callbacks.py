from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _config_helpers import *


@patch.dict(
    "os.environ",
    {
        "DYNAMIC_HOME": "/tmp/dynamic",
        "SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer",
    },
)
class ConfigPerformanceCallbackTest(ConfigTestCase):
    def test_overfit_performance_is_explicitly_opt_in(self):
        default = _overfit()
        enabled = _performance_overfit(
            "callbacks.performance.hardware_peak_flops=123.0"
        )

        self.assertFalse(default.callbacks.performance.enabled)
        self.assertTrue(enabled.callbacks.performance.enabled)
        self.assertEqual(enabled.callbacks.performance.hardware_peak_flops, 123.0)
        self.assertEqual(enabled.callbacks.performance.log_every_n_steps, 100)
        self.assertEqual(enabled.callbacks.performance.warmup_steps, 20)
        self.assertEqual(enabled.callbacks.performance.measure_window_steps, 100)
        self.assertTrue(enabled.callbacks.performance.sync_cuda)
        self.assertTrue(enabled.callbacks.performance.sync_distributed)

        with self.assertRaisesRegex(ValueError, "task_sample.enabled=false"):
            _overfit("callbacks.performance.enabled=true")

    @patch("speech_to_speech.training.composition.TrainingFlops")
    @patch("speech_to_speech.training.composition.PerformanceCallback")
    def test_overfit_performance_builds_the_dynamic_provider(
        self,
        performance,
        training_flops,
    ):
        disabled = _overfit()
        enabled = _performance_overfit(
            "callbacks.performance.hardware_peak_flops=123.0"
        )

        self.assertIsNone(build_performance(disabled.callbacks.performance))
        callback = build_performance(enabled.callbacks.performance)

        performance.assert_called_once_with(
            model_flops_per_batch=training_flops.return_value,
            hardware_peak_flops=123.0,
            log_every_n_steps=100,
            warmup_steps=20,
            measure_window_steps=100,
            sync_cuda=True,
            sync_distributed=True,
        )
        self.assertIs(callback, performance.return_value)

    @patch("speech_to_speech.training.composition.GradLogger")
    def test_overfit_performance_omits_extra_gradient_passes(self, grad_logger):
        default = _overfit()
        performance = _performance_overfit()
        flow_comparison = self._assert_gradient_logger(
            grad_logger,
            default,
            AcousticType.FLOW,
            "flow_matching",
            _default_gradient_probes(),
        )
        rvq_comparison = self._assert_gradient_logger(
            grad_logger,
            default,
            AcousticType.RVQ,
            "rvq",
            _default_gradient_probes(),
        )

        self.assertIsNone(
            _gradient_logger(performance, AcousticType.FLOW, flow_comparison)
        )
        grad_logger.assert_not_called()

        frozen = _overfit(
            "callback/parameter_policy@callbacks.parameter_policy=speech_interface"
        )
        self.assertIsNone(_gradient_logger(frozen, AcousticType.RVQ, rvq_comparison))
        grad_logger.assert_not_called()

        partial = _overfit(
            "callback/parameter_policy@callbacks.parameter_policy=speech_interface_top_third"
        )
        partial_gradient = _gradient_logger(partial, AcousticType.RVQ, rvq_comparison)

        self.assertIs(partial_gradient, grad_logger.return_value)
        grad_logger.assert_called_once_with(
            (rvq_comparison,),
            (
                GradientProbe(
                    "backbone_norm",
                    ("model.backbone.norm.weight",),
                ),
            ),
            every_n_steps=1,
        )

    def test_logging_builder_uses_the_configured_layout(self):
        tensorboard = _overfit().logging
        with patch(
            "speech_to_speech.training.composition.TensorBoardLogger"
        ) as logger:
            built = build_logger(tensorboard)

        self.assertIs(built, logger.return_value)
        logger.assert_called_once_with(
            save_dir=tensorboard.save_dir,
            name=tensorboard.run_name,
        )

        csv = _overfit("experiment=overfit/toy_smoke").logging
        with patch("speech_to_speech.training.composition.CSVLogger") as logger:
            built = build_logger(csv)

        self.assertIs(built, logger.return_value)
        logger.assert_called_once_with(save_dir=csv.save_dir, name=csv.run_name)

    def test_train_uses_async_checkpoint(self):
        config = _train()

        callbacks = train_script.training_callbacks(
            config,
            Path("/tmp/output"),
            Mock(),
        )

        self.assertIsInstance(callbacks[0], ParameterPolicyCallback)
        checkpoint = next(
            callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
        )
        self.assertTrue(checkpoint.async_save)
        self.assertFalse(checkpoint._enable_version_counter)

    def test_train_constructs_unit_schedule_callback(self):
        config = _train(
            "optim.schedule.log_every_n_units=25",
            "optim.schedule.measure_window_batches=3",
            "optim.schedule.sync_cuda=false",
            "optim.schedule.stop_at_end=true",
            (
                "optim.schedule.phases=["
                "{name:warmup,duration:100,lr:{type:linear,start:0.0,end:1.0}},"
                "{name:main,duration:900,lr:{type:constant,value:1.0}}"
                "]"
            ),
        )

        callbacks = train_script.training_callbacks(
            config,
            Path("/tmp/output"),
            Mock(),
        )

        callback = next(
            callback
            for callback in callbacks
            if isinstance(callback, UnitScheduleCallback)
        )
        self.assertEqual(callback.clock.unit, "tokens")
        self.assertEqual(callback.clock.log_every_n_units, 25.0)
        self.assertEqual(callback.clock.measure_window_batches, 3)
        self.assertFalse(callback.clock.sync_cuda)
        self.assertFalse(callback.clock.sync_distributed)
        self.assertEqual(
            callback.schedule.milestones(),
            (("warmup", 100.0), ("main", 1000.0)),
        )
        self.assertEqual(callback.schedule.stop_unit(), 1000.0)

    def test_batch_units_count_fused_tokens(self):
        batch = ModelBatch(
            input_ids=torch.tensor([[1, 2, 0], [3, 4, 5]]),
            token_labels=torch.tensor([[-100, 2, -100], [-100, -100, 5]]),
            acoustic_target=None,
            tasks=[Task.MT, Task.MT],
            predictions=[PredictionModality.TEXT, PredictionModality.TEXT],
            pad_token_id=0,
        )

        units = BatchUnits("tokens")(
            trainer=object(),
            pl_module=object(),
            outputs=None,
            batch=FusedBatch((batch, batch), ("left", "right")),
            batch_idx=0,
        )

        self.assertEqual(units.unit, "tokens")
        self.assertEqual(units.valid, 10.0)
        self.assertEqual(units.padded, 12.0)

    def test_batch_units_do_not_materialize_device_tensors(self):
        batch = ModelBatch(
            input_ids=torch.tensor([[1, 2, 0], [3, 4, 5]]),
            token_labels=torch.tensor([[-100, 2, -100], [-100, -100, 5]]),
            acoustic_target=None,
            tasks=[Task.MT, Task.MT],
            predictions=[PredictionModality.TEXT, PredictionModality.TEXT],
            pad_token_id=0,
        )

        with patch.object(
            torch.Tensor,
            "item",
            side_effect=AssertionError("unit callback must use batch metadata"),
        ):
            units = BatchUnits("tokens")(
                trainer=object(),
                pl_module=object(),
                outputs=None,
                batch=batch,
                batch_idx=0,
            )

        self.assertEqual(units.valid, 5.0)
        self.assertEqual(units.padded, 6.0)

    def test_module_configure_optimizers_uses_schedule_runtime(self):
        runtime = Mock()
        runtime.configure_optimizers.return_value = "configured"
        module = SpeechToSpeechModule(
            ModuleConfig(),
            model=_OptimizerModel(),
            objective=Mock(),
            schedule_runtime=runtime,
        )

        configured = module.configure_optimizers()

        self.assertEqual(configured, "configured")
        optimizer = runtime.configure_optimizers.call_args.args[0]
        self.assertEqual(optimizer.param_groups[0]["lr"], OptimConfig().learning_rate)

    def test_train_constructs_gradient_probe_callback(self):
        config = _train(
            "experiment=train/staged_joint/stage_1",
            "callbacks.gradient_probe.enabled=true",
        )
        built = Mock()

        with patch(
            "speech_to_speech.training.composition.GradLogger",
            return_value=built,
        ) as factory:
            callbacks = train_script.training_callbacks(
                config,
                Path("/tmp/output"),
                Mock(),
            )

        self.assertIn(built, callbacks)
        factory.assert_called_once_with(
            (
                GradientComparison(
                    GradientTarget("token", "asr"),
                    GradientTarget("token", "tts"),
                ),
            ),
            (
                GradientProbe(
                    "backbone_l0_attn_lora",
                    (
                        r"model\.backbone\.(?:layers|mimo_layers)\.0\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.lora_[AB]\..*\.weight$",
                    ),
                    match="regex",
                ),
                GradientProbe(
                    "backbone_l0_ffn_lora",
                    (
                        r"model\.backbone\.(?:layers|mimo_layers)\.0\.mlp\.(gate_proj|up_proj|down_proj)\.lora_[AB]\..*\.weight$",
                    ),
                    match="regex",
                ),
            ),
            every_n_steps=10_000,
        )

    def test_train_performance_omits_gradient_probe_callback(self):
        config = _train(
            "experiment=train/staged_joint/stage_1",
            "callbacks.gradient_probe.enabled=true",
        )
        performance = Mock()

        with (
            patch(
                "speech_to_speech.training.composition.build_performance",
                return_value=performance,
            ),
            patch("speech_to_speech.training.composition.GradLogger") as factory,
        ):
            callbacks = train_script.training_callbacks(
                config,
                Path("/tmp/output"),
                Mock(),
            )

        self.assertIsInstance(callbacks[0], ParameterPolicyCallback)
        self.assertIs(callbacks[1], performance)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
