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
class TrainConfigContractTest(ConfigTestCase):
    def test_train_root_defaults_to_loader_plan_and_lora_policy(self):
        default = _train()

        self.assertIsInstance(default, StagedTrainRVQConfig)
        self.assertIs(default.callbacks.parameter_policy.name, ParameterPolicyName.LORA)
        self.assertIsInstance(default.model.lora, LoraConfig)
        if default.model.lora is None:
            self.fail("default train config must enable PEFT LoRA")
        self.assertEqual(default.model.lora.init_lora_weights, "pissa")
        self.assertFalse(default.runtime.gradient_checkpointing)
        self.assertEqual(default.optim.name, "adamw")
        self.assertFalse(default.validation.enabled)
        self.assertEqual(default.validation.loader, "tts")
        self.assertEqual(default.validation.split_label, "dev")
        self.assertEqual(default.validation.every_n_steps, 1000)
        self.assertEqual(default.validation.sanity_steps, -1)
        self.assertFalse(default.callbacks.performance.enabled)
        self.assertTrue(default.datamodule.dataloader.costs.enabled)
        self.assertEqual(default.datamodule.dataloader.costs.max_batch_frames, 4800)
        self.assertEqual(default.datamodule.dataloader.costs.planning_window, 256)
        self.assertFalse(default.trainer.use_distributed_sampler)
        self.assertEqual(
            default.trainer.strategy,
            "ddp_find_unused_parameters_false",
        )
        self.assertIs(default.loader_plan.mode, LoaderStepMode.FUSED_JOINT)
        self.assertEqual(default.loader_plan.step_mode, "fused_joint")
        self.assertTrue(default.loader_plan.fuse_loaders_per_step)
        with self.assertRaises(AttributeError):
            getattr(default.datamodule, "sample_index")

    def test_bicodec_runtime_enables_gradient_checkpointing(self):
        config = _train(
            "runtime=bicodec",
            "audio_sequence_layout=flattened",
            "model/acoustic=none",
            "datamodule/dataset=qwen_tts_speaker",
            "+datamodule.shape=single",
        )

        self.assertIs(config.callbacks.parameter_policy.name, ParameterPolicyName.LORA)
        self.assertIsInstance(config.model.lora, LoraConfig)
        self.assertTrue(config.runtime.gradient_checkpointing)
        self.assertEqual(config.datamodule.dataloader.batch_size, 8)
        self.assertTrue(config.datamodule.dataloader.costs.enabled)
        self.assertEqual(config.datamodule.dataloader.costs.max_batch_frames, 4800)
        self.assertEqual(config.datamodule.dataloader.costs.planning_window, 256)

    def test_train_stage_2_uses_accumulation_safe_loader_plan(self):
        config = _train("experiment=train/staged_joint/stage_2")

        self.assertEqual(config.output_subdir, "staged-joint/stage_2/rvq-8l")
        self.assertEqual(set(config.loader_plan.loaders), {"asr", "tts", "mt"})
        self.assertEqual(config.loader_plan.accumulate_grad_batches, 10)
        self.assertIs(config.loader_plan.mode, LoaderStepMode.WEIGHTED_WINDOW)
        self.assertTrue(config.loader_plan.fuse_loaders_per_step)
        self.assertEqual(config.datamodule.codec, "longcat")
        self.assertEqual(config.datamodule.dataset.name, DatasetName.WMT19_TTS)
        self.assertEqual(config.text_datamodule.dataset.name.value, "wmt19")
        self.assertEqual(config.train.max_steps, 1000000)
        self.assertIsNone(config.train.ckpt_path)

    def test_train_resume_ckpt_and_static_ddp_guards_are_preserved(self):
        resumed = _train("train.ckpt_path=/tmp/last.ckpt")
        self.assertEqual(resumed.train.ckpt_path, "/tmp/last.ckpt")

        with self.assertRaisesRegex(ValueError, "unused-parameter"):
            _train(
                "experiment=train/staged_joint/stage_4",
                "loader_plan.step_mode=serial_joint",
                "loader_plan.accumulate_grad_batches=6",
                "trainer.strategy=ddp_find_unused_parameters_false",
            )

    def test_train_token_config_requires_semantic_codec_artifact(self):
        token = _train(
            "runtime=longcat_native",
            "model/acoustic=none",
            "audio_sequence_layout=flattened",
        )

        self.assertIsInstance(token, StagedTrainTokenConfig)
        self.assertEqual(token.model.acoustic.type, AcousticType.NONE.value)
        self.assertEqual(token.run_name, "token")
        with self.assertRaisesRegex(ValueError, "semantic_codec_artifact"):
            _train("model/acoustic=none")

    def test_staged_joint_experiments_bind_loader_plan_and_parameter_policy(self):
        for index, policy, expected_panels in _STAGED_JOINT_CASES:
            with self.subTest(stage=index):
                config = _train(f"experiment=train/staged_joint/stage_{index}")

                self.assertEqual(
                    config.output_subdir,
                    f"staged-joint/stage_{index}/{config.run_name}",
                )
                self.assertIs(config.callbacks.parameter_policy.name, policy)
                self.assertEqual(
                    config.trainer.strategy,
                    "ddp_find_unused_parameters_false",
                )
                self.assertGreater(config.loader_plan.accumulate_grad_batches, 1)
                expected_mode = (
                    LoaderStepMode.FUSED_JOINT
                    if index == 1
                    else LoaderStepMode.WEIGHTED_WINDOW
                )
                self.assertIs(config.loader_plan.mode, expected_mode)
                self.assertTrue(config.loader_plan.fuse_loaders_per_step)
                self.assertTrue(config.callbacks.task_sample.enabled)
                self.assertEqual(config.callbacks.task_sample.every_n_steps, 10_000)
                self.assertEqual(
                    tuple(
                        (panel.loader, panel.task)
                        for panel in config.callbacks.task_sample.panels
                    ),
                    expected_panels,
                )
                self.assertTrue(
                    all(
                        panel.split == "train" and panel.indices == [0, 1, 2]
                        for panel in config.callbacks.task_sample.panels
                    )
                )

    def test_static_ddp_rejects_multi_loader_dynamic_branches(self):
        with self.assertRaisesRegex(ValueError, "serial_joint.*unused-parameter"):
            _train(
                "experiment=train/staged_joint/stage_4",
                "loader_plan.step_mode=serial_joint",
                "loader_plan.accumulate_grad_batches=6",
                "trainer.strategy=ddp_find_unused_parameters_false",
            )

    def test_fused_joint_allows_find_unused_ddp(self):
        config = _train(
            "experiment=train/staged_joint/stage_2",
            "loader_plan.step_mode=fused_joint",
            "loader_plan.accumulate_grad_batches=3",
            "trainer.strategy=ddp_find_unused_parameters_true",
        )

        self.assertIs(config.loader_plan.mode, LoaderStepMode.FUSED_JOINT)
        self.assertEqual(
            config.trainer.strategy,
            "ddp_find_unused_parameters_true",
        )

    def test_serial_joint_uses_loader_count_accumulation_and_find_unused_ddp(self):
        config = _train(
            "experiment=train/staged_joint/stage_2",
            "trainer=staged_ddp",
            "loader_plan.step_mode=serial_joint",
            "loader_plan.accumulate_grad_batches=3",
        )

        self.assertIs(config.loader_plan.mode, LoaderStepMode.SERIAL_JOINT)
        self.assertFalse(config.loader_plan.fuse_loaders_per_step)
        self.assertEqual(config.loader_plan.accumulate_grad_batches, 3)
        self.assertEqual(
            config.trainer.strategy,
            "ddp_find_unused_parameters_true",
        )
        self.assertEqual(
            config.loader_plan.loader_weights(),
            {"asr": 0.45, "tts": 0.45, "mt": 0.1},
        )

        with self.assertRaisesRegex(ValueError, "serial_joint.*positive loaders"):
            _train(
                "experiment=train/staged_joint/stage_2",
                "trainer=staged_ddp",
                "loader_plan.step_mode=serial_joint",
                "loader_plan.loaders.asr.weight=1.0",
                "loader_plan.loaders.tts.weight=1.0",
                "loader_plan.loaders.mt.weight=1.0",
                "loader_plan.accumulate_grad_batches=10",
            )

    def test_fused_joint_accepts_non_equal_task_loss_weights(self):
        config = _train(
            "experiment=train/staged_joint/stage_2",
            "loader_plan.step_mode=fused_joint",
            "loader_plan.accumulate_grad_batches=3",
        )

        self.assertIs(config.loader_plan.mode, LoaderStepMode.FUSED_JOINT)
        self.assertEqual(
            config.loader_plan.loader_weights(),
            {"asr": 0.45, "tts": 0.45, "mt": 0.1},
        )

    def test_joint_step_modes_require_single_task_loaders(self):
        with self.assertRaisesRegex(ValueError, "exactly one positive task"):
            _train(
                "experiment=train/staged_joint/stage_2",
                "loader_plan.step_mode=fused_joint",
                "++loader_plan.loaders.asr.task_weights.s2tt=1.0",
            )

    def test_fused_multi_loader_requires_a_full_window(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            _train(
                "experiment=train/staged_joint/stage_4",
                "loader_plan.accumulate_grad_batches=1",
                "trainer.strategy=ddp_find_unused_parameters_false",
            )

    def test_stable_codec_stage1_long_run_enables_fixed_samples_for_asr_and_tts(self):
        config = _train(
            "experiment=train/stable_codec/stage_1",
            "datamodule.dataset.split_manifest=/tmp/splits.json",
        )

        self.assertEqual(config.train.max_steps, 1_000_000)
        self.assertEqual(config.output_subdir, "stable-codec/stage_1/token")
        self.assertEqual(config.runtime.codec, "stable_codec")
        self.assertIs(config.audio_sequence_layout, AudioSequenceLayout.FLATTENED)
        self.assertIsNone(config.runtime.audio_tokenizer)
        self.assertTrue(config.callbacks.task_sample.enabled)
        self.assertTrue(config.validation.enabled)
        self.assertEqual(config.validation.every_n_steps, 10_000)
        self.assertEqual(config.validation.sanity_steps, 0)
        self.assertEqual(
            [
                (panel.split, panel.loader, panel.task, panel.indices)
                for panel in config.callbacks.task_sample.panels
            ],
            [
                ("train", "asr", "asr", [0, 1, 2]),
                ("train", "tts", "tts", [0, 1, 2]),
            ],
        )
        self.assertEqual(config.callbacks.checkpoint.every_n_train_steps, 10_000)

    def test_validation_sample_panel_requires_validation_dataset(self):
        with self.assertRaisesRegex(ValueError, "split_manifest"):
            _train("experiment=train/stable_codec/stage_1")

    def test_mt_sample_panel_requires_train_split(self):
        with self.assertRaisesRegex(ValueError, "validation.*speech loaders"):
            _train(
                "experiment=train/staged_joint/stage_2",
                "callbacks.task_sample.enabled=true",
                "callbacks.task_sample.panels=[{split:validation,loader:mt,task:mt,indices:[0]}]",
                "validation.enabled=true",
                "datamodule.dataset.split_manifest=/tmp/splits.json",
            )

    def test_train_datamodule_routes_mt_to_text_loader(self):
        config = _train("experiment=train/staged_joint/stage_2")

        datamodule = build_train_datamodule(config, object())

        self.assertIsInstance(datamodule, DataModule)
        self.assertEqual(datamodule.loader_names, ("asr", "tts", "mt"))
        self.assertEqual(
            datamodule.loader_specs["asr"].kind,
            LoaderKind.SPEECH,
        )
        self.assertEqual(
            datamodule.loader_specs["tts"].kind,
            LoaderKind.SPEECH,
        )
        self.assertEqual(
            datamodule.loader_specs["mt"].kind,
            LoaderKind.TEXT,
        )
        self.assertEqual(datamodule.schedule.weights, config.loader_plan.loader_weights())
        self.assertEqual(datamodule.schedule.accumulate_grad_batches, 10)
        self.assertEqual(datamodule.schedule.step_mode, "weighted_window")
        self.assertTrue(datamodule.schedule.fuse_loaders_per_step)

        with self.assertRaisesRegex(ValueError, "cannot mix pure text and speech"):
            LoaderConfig(
                weight=1.0,
                task_weights={"mt": 1.0, "tts": 1.0},
            )

    def test_train_datamodule_clones_the_selected_loader_for_validation(self):
        config = _train(
            "experiment=train/staged_joint/stage_2",
            "validation.enabled=true",
            "validation.loader=tts",
            "validation.split_label=dev",
            "datamodule.dataset.split_manifest=/tmp/splits.json",
        )

        datamodule = build_train_datamodule(config, object())

        training = datamodule.loader_specs["tts"].speech_config
        validation = datamodule.validation_spec
        if training is None or validation is None:
            self.fail("validation speech loader was not configured")
        validation_config = validation.speech_config
        if validation_config is None:
            self.fail("validation must reuse a speech loader")
        self.assertIsNot(training, validation_config)
        self.assertIsNot(training.dataset, validation_config.dataset)
        self.assertEqual(training.dataset.split_label, "train")
        self.assertEqual(validation_config.dataset.split_label, "dev")
        self.assertEqual(validation.task_weights, {Task.TTS: 1.0})

    def test_train_datamodule_builds_limited_wmt19_mt_validation(self):
        config = _train(
            "experiment=train/staged_joint/stage_2",
            "validation.enabled=true",
            "validation.loader=mt",
        )

        datamodule = build_train_datamodule(config, object())

        validation = datamodule.validation_spec
        if validation is None or validation.text_config is None:
            self.fail("MT validation text loader was not configured")
        self.assertIs(validation.kind, LoaderKind.TEXT)
        self.assertEqual(validation.task_weights, {Task.MT: 1.0})
        self.assertEqual(validation.text_config.dataset.split, "validation")
        self.assertEqual(validation.max_samples, 1000)

    def test_enabled_validation_requires_a_distinct_manifest_split(self):
        with self.assertRaisesRegex(ValueError, "split_manifest"):
            _train("validation.enabled=true")
        with self.assertRaisesRegex(ValueError, "unknown validation loader"):
            _train(
                "validation.enabled=true",
                "validation.loader=missing",
                "datamodule.dataset.split_manifest=/tmp/splits.json",
            )
        mt = _train(
            "experiment=train/staged_joint/stage_2",
            "validation.enabled=true",
            "validation.loader=mt",
        )
        self.assertEqual(mt.validation.text_split, "validation")
        self.assertEqual(mt.validation.max_samples, 1000)
        with self.assertRaisesRegex(ValueError, "must differ"):
            _train(
                "validation.enabled=true",
                "validation.split_label=train",
                "datamodule.dataset.split_manifest=/tmp/splits.json",
            )

    def test_train_trainer_forwards_step_validation_options(self):
        config = _train(
            "validation.enabled=true",
            "validation.every_n_steps=25",
            "validation.sanity_steps=2",
            "datamodule.dataset.split_manifest=/tmp/splits.json",
        )
        callbacks = []

        with (
            patch("scripts.train.entry_trainer") as entry,
            patch("scripts.train.build_logger") as logger,
        ):
            trainer = train_script.build_trainer(config, Path("/tmp/output"), callbacks)

        self.assertIs(trainer, entry.return_value)
        self.assertIs(entry.call_args.kwargs["logger"], logger.return_value)
        self.assertEqual(entry.call_args.kwargs["accumulate_grad_batches"], 1)
        self.assertEqual(entry.call_args.kwargs["val_check_interval"], 25)
        self.assertEqual(entry.call_args.kwargs["num_sanity_val_steps"], 2)

    def test_serial_joint_trainer_forwards_accumulation_window(self):
        config = _train(
            "experiment=train/staged_joint/stage_2",
            "trainer=staged_ddp",
            "loader_plan.step_mode=serial_joint",
            "loader_plan.loaders.asr.weight=1.0",
            "loader_plan.loaders.tts.weight=1.0",
            "loader_plan.loaders.mt.weight=1.0",
            "loader_plan.accumulate_grad_batches=3",
            "validation.enabled=true",
            "validation.loader=mt",
            "validation.every_n_steps=25",
        )

        with (
            patch("scripts.train.entry_trainer") as entry,
            patch("scripts.train.build_logger"),
        ):
            train_script.build_trainer(config, Path("/tmp/output"), [])

        self.assertEqual(entry.call_args.kwargs["accumulate_grad_batches"], 3)
        self.assertEqual(entry.call_args.kwargs["val_check_interval"], 75)

    @patch("scripts.train.build_trainer")
    @patch("scripts.train.training_callbacks", return_value=[])
    @patch("scripts.train.build_datamodule")
    @patch("scripts.train.build")
    @patch("scripts.train.runtime_for_sequence_layout")
    @patch("scripts.train.pl.seed_everything")
    def test_train_run_passes_ckpt_path_to_trainer_fit(
        self,
        seed,
        runtime_for_sequence_layout,
        build,
        datamodule,
        callbacks,
        trainer_factory,
    ):
        del seed, callbacks
        config = _train(
            "train.ckpt_path=/tmp/resume.ckpt",
            "trainer.enable_checkpointing=false",
        )
        module = Mock()
        model = Mock()
        build.return_value = (AcousticType.RVQ, module, model)
        runtime_for_sequence_layout.return_value = Mock(acoustic_side_channel=False)
        trainer = Mock(is_global_zero=False)
        trainer_factory.return_value = trainer

        train_script.run(config)

        trainer.fit.assert_called_once_with(
            module,
            datamodule=datamodule.return_value,
            ckpt_path="/tmp/resume.ckpt",
        )


if __name__ == "__main__":
    unittest.main()
