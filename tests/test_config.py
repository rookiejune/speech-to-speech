from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from anytrain.lightning import ModelCheckpoint
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig
from omegaconf.errors import (
    ConfigAttributeError,
    ConfigKeyError,
    InterpolationResolutionError,
)

from scripts._config import (
    OverfitFlowConfig,
    OverfitRVQConfig,
    OverfitTokenConfig,
    StagedTrainRVQConfig,
    StagedTrainTokenConfig,
    overfit,
    train as parse_train,
)
from scripts._logging import build as build_logger
from scripts import train as train_script
from scripts.overfit import (
    _composition,
    _gradient_logger,
    _performance,
    _prepare_generation_module,
    runtime_config,
)
from scripts.train import (
    _is_text_loader,
    build_datamodule as build_train_datamodule,
)
from speech_to_speech.datamodule import DataModule
from speech_to_speech.datamodule.module import LoaderKind
from speech_to_speech.datamodule.dataset import DatasetName
from speech_to_speech.datamodule.types import DataShape
from speech_to_speech.model import (
    AdapterType,
    Config as ModelConfig,
    ToyConfig,
)
from speech_to_speech.model.acoustic import AcousticType, DecoderConfig
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.runtime import BackboneInitialization, Config as RuntimeConfig
from speech_to_speech.stage import (
    ParameterGroup,
    ParameterPolicyName,
    StageLoaderConfig,
    StageName,
)
from speech_to_speech.task import Task


class _DeviceRestoreModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(()))
        self.moves: list[torch.device] = []

    def to(self, device: torch.device):  # type: ignore[override]
        self.moves.append(device)
        return self


@patch.dict(
    "os.environ",
    {
        "DYNAMIC_HOME": "/tmp/dynamic",
        "SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer",
    },
)
class ConfigTest(unittest.TestCase):
    def test_roots_parse_to_src_aligned_configs(self):
        flow = overfit(_compose("overfit"))
        rvq = overfit(_compose("overfit", "model/acoustic=rvq"))
        token = overfit(_compose("overfit", "runtime=unicodec", "model/acoustic=none"))

        self.assertIsInstance(flow, OverfitFlowConfig)
        self.assertIsInstance(rvq, OverfitRVQConfig)
        self.assertIsInstance(token, OverfitTokenConfig)
        self.assertEqual(token.acoustic.type, AcousticType.NONE.value)
        self.assertIsInstance(flow.runtime, RuntimeConfig)
        self.assertIsInstance(flow.model, ModelConfig)
        self.assertIsInstance(flow.pl_module, ModuleConfig)
        self.assertIsInstance(flow.acoustic.decoder, DecoderConfig)
        self.assertEqual(flow.runtime.codec, "longcat")
        self.assertEqual(token.runtime.codec, "unicodec")
        self.assertIs(flow.model.semantic_audio_adapter, AdapterType.LINEAR)
        self.assertFalse(flow.callbacks.performance.enabled)

    def test_acoustic_generator_initialization_is_explicit(self):
        flow = overfit(
            _compose("overfit", "acoustic.init_artifact=/tmp/flow-artifact")
        )
        rvq = overfit(
            _compose(
                "overfit",
                "model/acoustic=rvq",
                "acoustic.init_artifact=/tmp/rvq-artifact",
            )
        )

        self.assertEqual(flow.acoustic.init_artifact, "/tmp/flow-artifact")
        self.assertEqual(rvq.acoustic.init_artifact, "/tmp/rvq-artifact")

    def test_toy_smoke_selects_model_and_dataset_without_a_toy_runtime(self):
        config = overfit(_compose("overfit", "experiment=toy_smoke"))

        self.assertIsInstance(config, OverfitFlowConfig)
        self.assertIsInstance(config.runtime, RuntimeConfig)
        self.assertEqual(config.runtime.codec, "longcat")
        self.assertEqual(config.runtime.backbone, "Qwen/Qwen3-0.6B")
        self.assertEqual(config.runtime.device, "cpu")
        self.assertIsInstance(config.model.toy, ToyConfig)
        self.assertEqual(config.model.toy.hidden_size, 32)
        self.assertIs(config.data.name, DatasetName.TOY)
        self.assertEqual(config.data.toy_samples, 8)
        self.assertEqual(config.data.toy_frames, 4)
        self.assertEqual(config.train.max_steps, 2)
        self.assertFalse(config.callbacks.task_sample.enabled)
        self.assertFalse(config.callbacks.evaluation.enabled)

        production = overfit(_compose("overfit"))
        self.assertIsNone(production.model.toy)
        self.assertIs(production.data.name, DatasetName.WMT19_TTS)

        selected = overfit(_compose("overfit", "model=toy", "data=toy"))
        self.assertIsInstance(selected.model.toy, ToyConfig)
        self.assertIs(selected.data.name, DatasetName.TOY)

    def test_random_backbone_requires_unambiguous_full_training(self):
        random = overfit(
            _compose("overfit", "runtime.backbone_initialization=random")
        )

        self.assertIs(
            random.runtime.backbone_initialization,
            BackboneInitialization.RANDOM,
        )
        with self.assertRaisesRegex(ValueError, "cannot be combined with model.toy"):
            overfit(
                _compose(
                    "overfit",
                    "model=toy",
                    "data=toy",
                    "runtime.backbone_initialization=random",
                )
            )
        with self.assertRaisesRegex(ValueError, "fully trainable backbone"):
            parse_train(
                _compose("train", "runtime.backbone_initialization=random")
            )

        train = parse_train(
            _compose(
                "train",
                "runtime.backbone_initialization=random",
                "parameter_policy=full",
            )
        )
        self.assertIs(
            train.runtime.backbone_initialization,
            BackboneInitialization.RANDOM,
        )

    def test_full_codec_sequence_smoke_is_token_only_comparison(self):
        config = overfit(_compose("overfit", "experiment=longcat_full_sequence_smoke"))

        self.assertIsInstance(config, OverfitTokenConfig)
        self.assertEqual(config.runtime.codec, "longcat")
        self.assertEqual(config.runtime.audio_representation.value, "full_codec_sequence")
        self.assertEqual(config.acoustic.type, AcousticType.NONE.value)
        self.assertIsInstance(config.model.toy, ToyConfig)
        self.assertIs(config.data.name, DatasetName.TOY)
        self.assertEqual(config.run_name, "longcat-full-sequence-token")
        self.assertEqual(config.train.max_steps, 2)
        self.assertFalse(config.callbacks.task_sample.enabled)
        self.assertFalse(config.callbacks.evaluation.enabled)

    def test_decoupled_semantic_smoke_loads_artifact_config(self):
        with patch.dict(
            "os.environ",
            {"SPEECH_TO_SPEECH_SEMANTIC_CODEC_ARTIFACT": "/tmp/semantic-codec"},
        ):
            config = overfit(
                _compose(
                    "overfit",
                    "experiment=longcat_decoupled_semantic_only_smoke",
                )
            )

        self.assertIsInstance(config, OverfitTokenConfig)
        self.assertEqual(config.runtime.codec, "longcat")
        self.assertEqual(
            config.runtime.semantic_codec_artifact,
            "/tmp/semantic-codec",
        )
        self.assertEqual(config.runtime.device, "cpu")
        self.assertEqual(config.acoustic.type, AcousticType.NONE.value)
        self.assertIsInstance(config.model.toy, ToyConfig)
        self.assertIs(config.data.name, DatasetName.TOY)

    def test_bicodec_smokes_use_qwen_single_speaker_cells(self):
        with patch.dict(
            "os.environ",
            {"SPEECH_TO_SPEECH_BICODEC_SEMANTIC_ARTIFACT": "/tmp/bicodec-semantic"},
        ):
            semantic = overfit(
                _compose("overfit", "experiment=bicodec_semantic_only_smoke")
            )
        full = overfit(
            _compose("overfit", "experiment=bicodec_full_sequence_smoke")
        )

        for config in (semantic, full):
            self.assertIsInstance(config, OverfitTokenConfig)
            self.assertEqual(config.runtime.codec, "bicodec")
            self.assertIs(config.data.name, DatasetName.QWEN_TTS_SPEAKER)
            self.assertIs(config.data.shape, DataShape.SINGLE)
            self.assertIsNone(config.data.speaker)

    def test_composition_must_match_codec_capabilities(self):
        flow = overfit(_compose("overfit"))
        token = overfit(_compose("overfit", "runtime=unicodec", "model/acoustic=none"))
        token_with_semantic_support = overfit(
            _compose(
                "overfit",
                "runtime=longcat_native",
                "model/acoustic=none",
                "runtime.semantic_codec_artifact=/tmp/semantic-codec",
            )
        )

        self.assertIs(
            _composition(token, uses_acoustic_side_channel=False),
            AcousticType.NONE,
        )
        self.assertIs(
            _composition(
                token_with_semantic_support,
                uses_acoustic_side_channel=False,
            ),
            AcousticType.NONE,
        )
        with self.assertRaisesRegex(ValueError, "model/acoustic=none"):
            _composition(flow, uses_acoustic_side_channel=False)

    def test_root_schema_rejects_unknown_and_foreign_fields(self):
        cases = [
            (overfit, _compose("overfit", "+unknown=1"), "unknown"),
            (
                overfit,
                _compose("overfit", "+acoustic.normalize_features=true"),
                "acoustic.normalize_features",
            ),
            (
                overfit,
                _compose(
                    "overfit",
                    "model/acoustic=rvq",
                    "+acoustic.repa.weight=0.1",
                ),
                "acoustic.repa",
            ),
            (
                overfit,
                _compose("overfit", "+train.ckpt_path=/tmp/resume.ckpt"),
                "train.ckpt_path",
            ),
        ]

        for parser, raw, key in cases:
            with self.subTest(key=key):
                with self.assertRaises(
                    (ConfigKeyError, ConfigAttributeError)
                ) as raised:
                    parser(raw)
                self.assertIn(key, str(raised.exception))

    def test_overfit_performance_is_explicitly_opt_in(self):
        default = overfit(_compose("overfit"))
        enabled = overfit(
            _compose(
                "overfit",
                "callbacks.performance.enabled=true",
                "callbacks.task_sample.enabled=false",
                "callbacks.performance.hardware_peak_flops=123.0",
            )
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
            overfit(
                _compose(
                    "overfit",
                    "callbacks.performance.enabled=true",
                )
            )

    @patch("scripts.overfit.TrainingFlops")
    @patch("scripts.overfit.PerformanceCallback")
    def test_overfit_performance_builds_the_dynamic_provider(
        self,
        performance,
        training_flops,
    ):
        disabled = overfit(_compose("overfit"))
        enabled = overfit(
            _compose(
                "overfit",
                "callbacks.task_sample.enabled=false",
                "callbacks.performance.enabled=true",
                "callbacks.performance.hardware_peak_flops=123.0",
            )
        )

        self.assertIsNone(_performance(disabled))
        callback = _performance(enabled)

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

    @patch("scripts.overfit.GradLogger")
    def test_overfit_performance_omits_extra_gradient_passes(self, grad_logger):
        default = overfit(_compose("overfit"))
        performance = overfit(
            _compose(
                "overfit",
                "callbacks.task_sample.enabled=false",
                "callbacks.performance.enabled=true",
            )
        )
        loss_pair = ("token", "flow_matching")

        gradient = _gradient_logger(default, AcousticType.FLOW, loss_pair)

        self.assertIs(gradient, grad_logger.return_value)
        grad_logger.assert_called_once_with(
            loss_pair,
            "model.backbone.model.layers.0.self_attn.q_proj.weight",
            every_n_steps=1,
        )
        grad_logger.reset_mock()

        rvq_pair = ("token", "rvq")
        rvq_gradient = _gradient_logger(default, AcousticType.RVQ, rvq_pair)

        self.assertIs(rvq_gradient, grad_logger.return_value)
        grad_logger.assert_called_once_with(
            rvq_pair,
            "model.backbone.model.layers.0.self_attn.q_proj.weight",
            every_n_steps=1,
        )
        grad_logger.reset_mock()

        self.assertIsNone(_gradient_logger(performance, AcousticType.FLOW, loss_pair))
        grad_logger.assert_not_called()

        frozen = overfit(_compose("overfit", "parameter_policy=speech_interface"))
        self.assertIsNone(_gradient_logger(frozen, AcousticType.RVQ, rvq_pair))
        grad_logger.assert_not_called()

        partial = overfit(
            _compose("overfit", "parameter_policy=speech_interface_top_third")
        )
        partial_gradient = _gradient_logger(partial, AcousticType.RVQ, rvq_pair)

        self.assertIs(partial_gradient, grad_logger.return_value)
        grad_logger.assert_called_once_with(
            rvq_pair,
            "model.backbone.model.norm.weight",
            every_n_steps=1,
        )

    def test_training_outputs_use_one_tensorboard_root(self):
        configs = (
            overfit(_compose("overfit")),
            overfit(_compose("overfit", "experiment=unicodec_overfit")),
        )

        for config in configs:
            with self.subTest(output_subdir=config.output_subdir):
                root = Path(config.repo_output_root)
                self.assertEqual(
                    Path(config.output_dir),
                    root / config.output_subdir,
                )
                self.assertEqual(
                    Path(config.logging.save_dir),
                    root / "tensorboard",
                )
                self.assertEqual(config.logging.run_name, config.output_subdir)

        csv = overfit(_compose("overfit", "experiment=toy_smoke"))
        self.assertEqual(csv.logging.save_dir, csv.output_dir)
        self.assertEqual(csv.logging.run_name, "csv")

    def test_repo_output_root_prefers_the_project_training_root(self):
        with patch.dict(
            "os.environ",
            {"SPEECH_TO_SPEECH_TRAIN_ROOT": "/tmp/speech-train"},
        ):
            overfit_config = overfit(_compose("overfit"))

        self.assertEqual(overfit_config.repo_output_root, "/tmp/speech-train")

    def test_repo_output_root_falls_back_to_the_dynamic_train_root(self):
        with patch.dict(
            "os.environ",
            {
                "DYNAMIC_HOME": "/tmp/dynamic",
                "SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer",
            },
            clear=True,
        ):
            config = overfit(_compose("overfit"))

        self.assertEqual(config.repo_output_root, "/tmp/dynamic/train/speech-to-speech")

    def test_missing_training_root_fails_without_dynamic_home(self):
        with (
            patch.dict(
                "os.environ",
                {"SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer"},
                clear=True,
            ),
            self.assertRaisesRegex(InterpolationResolutionError, "DYNAMIC_HOME"),
        ):
            overfit(_compose("overfit"))

    def test_logging_builder_uses_the_configured_layout(self):
        tensorboard = overfit(_compose("overfit")).logging
        with patch("scripts._logging.TensorBoardLogger") as logger:
            built = build_logger(tensorboard)

        self.assertIs(built, logger.return_value)
        logger.assert_called_once_with(
            save_dir=tensorboard.save_dir,
            name=tensorboard.run_name,
        )

        csv = overfit(_compose("overfit", "experiment=toy_smoke")).logging
        with patch("scripts._logging.CSVLogger") as logger:
            built = build_logger(csv)

        self.assertIs(built, logger.return_value)
        logger.assert_called_once_with(save_dir=csv.save_dir, name=csv.run_name)

    def test_output_subdir_cannot_escape_the_repo_output_root(self):
        for override in ("output_subdir=/tmp/run", "output_subdir=../run"):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "output_subdir"):
                    overfit(_compose("overfit", override))

        with self.assertRaisesRegex(ValueError, "output_dir must equal"):
            overfit(_compose("overfit", "output_dir=/tmp/other"))

    def test_unicodec_experiments_close_the_token_training_chain(self):
        cases = [
            (
                "unicodec_overfit",
                100,
                "auto",
                "auto",
                False,
                True,
            ),
            (
                "unicodec_ddp_smoke",
                2,
                "auto",
                "ddp_find_unused_parameters_true",
                True,
                False,
            ),
        ]

        for experiment, max_steps, devices, strategy, checkpointing, sampler in cases:
            with self.subTest(experiment=experiment):
                config = overfit(_compose("overfit", f"experiment={experiment}"))

                self.assertIsInstance(config, OverfitTokenConfig)
                self.assertEqual(config.runtime.codec, "unicodec")
                self.assertIsNone(config.runtime.audio_tokenizer)
                self.assertEqual(config.train.max_steps, max_steps)
                self.assertEqual(config.trainer.devices, devices)
                self.assertEqual(config.trainer.strategy, strategy)
                self.assertEqual(config.trainer.precision, "bf16-mixed")
                self.assertEqual(config.trainer.max_epochs, -1)
                self.assertEqual(config.trainer.log_every_n_steps, 1)
                self.assertIs(config.trainer.enable_checkpointing, checkpointing)
                self.assertIs(config.trainer.use_distributed_sampler, sampler)
                self.assertTrue(config.callbacks.task_sample.enabled)
                self.assertEqual(config.callbacks.task_sample.every_n_steps, 1)

    def test_rvq_native_stage_smoke_configs_cover_all_parameter_stages(self):
        policy_names = [
            ParameterPolicyName.FULL,
            ParameterPolicyName.SPEECH_INTERFACE,
            ParameterPolicyName.SPEECH_INTERFACE,
            ParameterPolicyName.SPEECH_INTERFACE_TOP_THIRD,
            ParameterPolicyName.FULL,
        ]
        for index, stage in enumerate(StageName):
            with self.subTest(stage=stage):
                config = overfit(
                    _compose(
                        "overfit",
                        f"experiment=011_rvq_native_stage_{index}_smoke",
                    )
                )

                self.assertIsInstance(config, OverfitRVQConfig)
                self.assertIs(config.stage.name, stage)
                self.assertIs(config.parameter_policy.name, policy_names[index])
                self.assertEqual(config.task, "s2st")
                self.assertEqual(config.runtime.codec, "longcat")
                self.assertIsNone(config.runtime.audio_tokenizer)
                self.assertEqual(config.train.max_steps, 2)
                self.assertFalse(config.callbacks.task_sample.enabled)
                self.assertTrue(config.callbacks.evaluation.enabled)

        stage_1 = overfit(
            _compose("overfit", "stage=stage_1", "parameter_policy=speech_interface")
        )
        self.assertIn(ParameterGroup.BACKBONE, stage_1.parameter_policy.frozen_groups)
        self.assertEqual(set(stage_1.stage.loaders), {"asr", "tts"})
        self.assertEqual(stage_1.stage.batches_per_step, 2)

        stage_2 = overfit(_compose("overfit", "stage=stage_2"))
        self.assertEqual(set(stage_2.stage.loaders), {"asr", "tts", "mt"})
        self.assertEqual(stage_2.stage.loaders["mt"].task_weights, {"mt": 1.0})
        self.assertEqual(stage_2.stage.batches_per_step, 10)

        stage_4 = overfit(_compose("overfit", "stage=stage_4"))
        self.assertEqual(
            set(stage_4.stage.loaders),
            {"asr_s2tt", "tts_t2st", "s2st", "mt"},
        )
        self.assertEqual(stage_4.stage.loaders["s2st"].weight, 0.70)

    def test_train_root_uses_stage_config_and_static_ddp(self):
        default = parse_train(_compose("train"))

        self.assertIsInstance(default, StagedTrainRVQConfig)
        self.assertEqual(default.stage.name, StageName.STAGE_1)
        self.assertFalse(default.validation.enabled)
        self.assertEqual(default.validation.loader, "tts")
        self.assertEqual(default.validation.split_label, "dev")
        self.assertEqual(default.validation.every_n_steps, 1000)
        self.assertEqual(default.validation.sanity_steps, -1)
        self.assertFalse(default.callbacks.performance.enabled)
        self.assertFalse(default.trainer.use_distributed_sampler)
        with self.assertRaises(AttributeError):
            getattr(default.data, "sample_index")

        config = parse_train(_compose("train", "stage=stage_2"))
        self.assertEqual(config.stage.name, StageName.STAGE_2)
        self.assertEqual(set(config.stage.loaders), {"asr", "tts", "mt"})
        self.assertEqual(config.stage.batches_per_step, 10)
        self.assertEqual(config.data.codec, "longcat")
        self.assertEqual(config.data.dataset.name, DatasetName.WMT19_TTS)
        self.assertEqual(config.text_data.dataset.name.value, "wmt19")
        self.assertEqual(config.train.max_steps, 1000000)
        self.assertIsNone(config.train.ckpt_path)

        resumed = parse_train(_compose("train", "train.ckpt_path=/tmp/last.ckpt"))
        self.assertEqual(resumed.train.ckpt_path, "/tmp/last.ckpt")

        ddp = parse_train(_compose("train", "trainer=static_ddp", "stage=stage_4"))

        self.assertEqual(ddp.trainer.strategy, "ddp_find_unused_parameters_false")
        self.assertFalse(ddp.trainer.use_distributed_sampler)
        self.assertEqual(set(ddp.stage.loaders), {"asr_s2tt", "tts_t2st", "s2st", "mt"})

        token = parse_train(
            _compose("train", "runtime=longcat_full_sequence", "model/acoustic=none")
        )

        self.assertIsInstance(token, StagedTrainTokenConfig)
        self.assertEqual(token.acoustic.type, AcousticType.NONE.value)
        self.assertEqual(token.run_name, "stage_1-token")
        with self.assertRaisesRegex(ValueError, "semantic_codec_artifact"):
            parse_train(_compose("train", "model/acoustic=none"))

    def test_parameter_policy_smoke_composes_each_supported_policy(self):
        policies = (
            ParameterPolicyName.FULL,
            ParameterPolicyName.SPEECH_INTERFACE,
            ParameterPolicyName.SPEECH_INTERFACE_TOP_THIRD,
            ParameterPolicyName.SEMANTIC_ONLY,
            ParameterPolicyName.ACOUSTIC_ONLY,
        )

        for policy in policies:
            with self.subTest(policy=policy.value):
                config = parse_train(
                    _compose(
                        "train",
                        "experiment=train/parameter_policy_smoke",
                        f"parameter_policy={policy.value}",
                    )
                )

                self.assertIs(config.parameter_policy.name, policy)
                self.assertEqual(config.train.max_steps, 2)
                self.assertEqual(config.trainer.accelerator, "cpu")
                self.assertIn(policy.value, config.output_subdir)

    def test_staged_joint_experiments_bind_stage_and_parameter_policy(self):
        policies = (
            ParameterPolicyName.SPEECH_INTERFACE,
            ParameterPolicyName.SPEECH_INTERFACE,
            ParameterPolicyName.SPEECH_INTERFACE_TOP_THIRD,
            ParameterPolicyName.FULL,
        )

        for index, policy in enumerate(policies, start=1):
            with self.subTest(stage=index):
                config = parse_train(
                    _compose(
                        "train",
                        f"experiment=train/staged_joint_stage_{index}",
                    )
                )

                self.assertIs(config.stage.name, StageName(f"stage_{index}"))
                self.assertIs(config.parameter_policy.name, policy)

    def test_stable_codec_stage1_long_run_enables_fixed_samples_for_asr_and_tts(self):
        config = parse_train(
            _compose("train", "experiment=train/stable_codec_stage1_train")
        )

        self.assertEqual(config.train.max_steps, 1_000_000)
        self.assertEqual(config.stage.name, StageName.STAGE_1)
        self.assertEqual(config.runtime.codec, "stable_codec")
        self.assertEqual(config.runtime.audio_representation.value, "full_codec_sequence")
        self.assertIsNone(config.runtime.audio_tokenizer)
        self.assertTrue(config.callbacks.task_sample.enabled)
        self.assertEqual(
            config.callbacks.task_sample.indices_by_loader,
            {"asr": [0], "tts": [0]},
        )
        self.assertEqual(config.callbacks.checkpoint.every_n_train_steps, 10_000)

    def test_fdu_stage_smoke_configs_cover_formal_train_stages(self):
        stage_0 = overfit(_compose("overfit", "experiment=fdu_stage_0_smoke"))
        self.assertIsInstance(stage_0, OverfitRVQConfig)
        self.assertIs(stage_0.stage.name, StageName.STAGE_0)
        self.assertEqual(stage_0.task, "s2st")
        self.assertEqual(stage_0.train.max_steps, 2)
        self.assertFalse(stage_0.callbacks.task_sample.enabled)
        self.assertTrue(stage_0.callbacks.evaluation.enabled)
        self.assertIn("013-fdu-stage-smoke", stage_0.output_dir)

        for index in range(1, 5):
            with self.subTest(stage=index):
                config = parse_train(
                    _compose("train", f"experiment=train/fdu_stage_{index}_smoke")
                )

                self.assertIsInstance(config, StagedTrainRVQConfig)
                self.assertIs(config.stage.name, StageName(f"stage_{index}"))
                self.assertEqual(config.train.max_steps, 2)
                self.assertEqual(
                    config.trainer.strategy,
                    "ddp_find_unused_parameters_false",
                )
                self.assertFalse(config.trainer.use_distributed_sampler)
                self.assertFalse(config.trainer.enable_checkpointing)
                self.assertEqual(config.data.dataloader.batch_size, 4)
                self.assertEqual(config.data.dataloader.num_workers, 4)
                self.assertEqual(config.text_data.dataloader.batch_size, 4)
                self.assertIn("013-fdu-stage-smoke", config.output_dir)

    def test_fdu_acoustic_none_smoke_configs_cover_all_stages(self):
        stage_0 = overfit(
            _compose("overfit", "experiment=fdu_stage_0_acoustic_none_smoke")
        )
        self.assertIsInstance(stage_0, OverfitTokenConfig)
        self.assertIs(stage_0.stage.name, StageName.STAGE_0)
        self.assertEqual(stage_0.task, "s2st")
        self.assertEqual(stage_0.acoustic.type, AcousticType.NONE.value)
        self.assertEqual(stage_0.runtime.audio_representation.value, "full_codec_sequence")
        self.assertEqual(stage_0.run_name, "stage_0-token")
        self.assertFalse(stage_0.callbacks.evaluation.enabled)
        self.assertEqual(stage_0.train.max_steps, 2)

        for index in range(1, 5):
            with self.subTest(stage=index):
                config = parse_train(
                    _compose(
                        "train",
                        f"experiment=train/fdu_stage_{index}_acoustic_none_smoke",
                    )
                )

                self.assertIsInstance(config, StagedTrainTokenConfig)
                self.assertIs(config.stage.name, StageName(f"stage_{index}"))
                self.assertEqual(config.acoustic.type, AcousticType.NONE.value)
                self.assertEqual(config.runtime.audio_representation.value, "full_codec_sequence")
                self.assertEqual(config.run_name, f"stage_{index}-token")
                self.assertEqual(config.train.max_steps, 2)
                self.assertEqual(
                    config.trainer.strategy,
                    "ddp_find_unused_parameters_false",
                )
                self.assertFalse(config.trainer.use_distributed_sampler)
                self.assertFalse(config.trainer.enable_checkpointing)
                self.assertEqual(config.data.dataloader.batch_size, 4)
                self.assertEqual(config.text_data.dataloader.batch_size, 4)
                self.assertIn("013-fdu-stage-smoke", config.output_dir)

    def test_train_datamodule_routes_mt_to_text_loader(self):
        config = parse_train(_compose("train", "stage=stage_2"))

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
        self.assertEqual(datamodule.schedule.weights, config.stage.loader_weights())
        self.assertEqual(datamodule.schedule.batches_per_step, 10)

        config.stage.loaders["bad"] = StageLoaderConfig(
            weight=1.0,
            task_weights={"mt": 1.0, "tts": 1.0},
        )
        with self.assertRaisesRegex(ValueError, "cannot mix pure text and speech"):
            build_train_datamodule(config, object())

    def test_train_datamodule_clones_the_selected_loader_for_validation(self):
        config = parse_train(
            _compose(
                "train",
                "stage=stage_2",
                "validation.enabled=true",
                "validation.loader=tts",
                "validation.split_label=dev",
                "data.dataset.split_manifest=/tmp/splits.json",
            )
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

    def test_stage1_pilot_validation_smoke_owns_the_full_validation_budget(self):
        config = parse_train(
            _compose(
                "train",
                "experiment=train/014_stage1_pilot_validation_smoke",
                "data.dataset.root=/tmp/pilot",
                "data.dataset.split_manifest=/tmp/pilot/splits.json",
            )
        )

        self.assertIs(config.stage.name, StageName.STAGE_1)
        self.assertEqual(config.train.max_steps, 1)
        self.assertTrue(config.validation.enabled)
        self.assertEqual(config.validation.loader, "tts")
        self.assertEqual(config.validation.split_label, "dev")
        self.assertEqual(config.validation.every_n_steps, 1)
        self.assertEqual(config.validation.sanity_steps, -1)
        self.assertEqual(config.data.dataloader.batch_size, 1)
        self.assertEqual(config.data.dataloader.num_workers, 0)
        self.assertFalse(config.trainer.enable_checkpointing)

        canary = parse_train(
            _compose(
                "train",
                "experiment=train/014_stage1_pilot_canary",
                "data.dataset.root=/tmp/pilot",
                "data.dataset.split_manifest=/tmp/pilot/splits.json",
            )
        )
        self.assertEqual(canary.train.max_steps, 100)
        self.assertEqual(canary.validation.every_n_steps, 50)
        self.assertEqual(canary.validation.sanity_steps, -1)
        self.assertTrue(canary.trainer.enable_checkpointing)
        self.assertEqual(canary.callbacks.checkpoint.every_n_train_steps, 50)
        self.assertEqual(canary.callbacks.checkpoint.save_top_k, -1)

        resumed = parse_train(
            _compose(
                "train",
                "experiment=train/014_stage1_pilot_resume_500",
                "data.dataset.root=/tmp/pilot",
                "data.dataset.split_manifest=/tmp/pilot/splits.json",
                "train.ckpt_path=/tmp/pilot/step-100.ckpt",
            )
        )
        self.assertEqual(resumed.train.max_steps, 500)
        self.assertEqual(resumed.train.ckpt_path, "/tmp/pilot/step-100.ckpt")
        self.assertEqual(resumed.validation.every_n_steps, 100)
        self.assertTrue(resumed.trainer.enable_checkpointing)
        self.assertEqual(resumed.callbacks.checkpoint.every_n_train_steps, 100)

        continued = parse_train(
            _compose(
                "train",
                "experiment=train/014_stage1_pilot_resume_2000",
                "data.dataset.root=/tmp/pilot",
                "data.dataset.split_manifest=/tmp/pilot/splits.json",
                "train.ckpt_path=/tmp/pilot/step-500.ckpt",
            )
        )
        self.assertEqual(continued.train.max_steps, 2000)
        self.assertEqual(continued.train.ckpt_path, "/tmp/pilot/step-500.ckpt")
        self.assertEqual(continued.validation.every_n_steps, 250)
        self.assertEqual(continued.callbacks.checkpoint.every_n_train_steps, 250)

        fp32 = parse_train(
            _compose(
                "train",
                "experiment=train/014_stage1_pilot_fp32_500",
                "data.dataset.root=/tmp/pilot",
                "data.dataset.split_manifest=/tmp/pilot/splits.json",
            )
        )
        self.assertEqual(fp32.train.seed, 0)
        self.assertEqual(fp32.train.max_steps, 500)
        self.assertIsNone(fp32.train.ckpt_path)
        self.assertEqual(fp32.validation.every_n_steps, 100)
        self.assertEqual(fp32.callbacks.checkpoint.every_n_train_steps, 100)

        resumed_fp32 = parse_train(
            _compose(
                "train",
                "experiment=train/014_stage1_pilot_fp32_resume_1000",
                "data.dataset.root=/tmp/pilot",
                "data.dataset.split_manifest=/tmp/pilot/splits.json",
                "train.ckpt_path=/tmp/pilot/fp32-step-500.ckpt",
            )
        )
        self.assertEqual(resumed_fp32.train.seed, 0)
        self.assertEqual(resumed_fp32.train.max_steps, 1000)
        self.assertEqual(
            resumed_fp32.train.ckpt_path,
            "/tmp/pilot/fp32-step-500.ckpt",
        )
        self.assertEqual(resumed_fp32.validation.every_n_steps, 100)
        self.assertEqual(resumed_fp32.validation.sanity_steps, 0)
        self.assertEqual(
            resumed_fp32.callbacks.checkpoint.every_n_train_steps,
            100,
        )

        with self.assertRaises(InterpolationResolutionError):
            parse_train(
                _compose(
                    "train",
                    "experiment=train/014_stage1_pilot_resume_500",
                    "data.dataset.root=/tmp/pilot",
                    "data.dataset.split_manifest=/tmp/pilot/splits.json",
                )
            )

    def test_enabled_validation_requires_a_distinct_manifest_split(self):
        with self.assertRaisesRegex(ValueError, "split_manifest"):
            parse_train(_compose("train", "validation.enabled=true"))
        with self.assertRaisesRegex(ValueError, "unknown validation loader"):
            parse_train(
                _compose(
                    "train",
                    "validation.enabled=true",
                    "validation.loader=missing",
                    "data.dataset.split_manifest=/tmp/splits.json",
                )
            )
        with self.assertRaisesRegex(ValueError, "must differ"):
            parse_train(
                _compose(
                    "train",
                    "validation.enabled=true",
                    "validation.split_label=train",
                    "data.dataset.split_manifest=/tmp/splits.json",
                )
            )

    def test_train_trainer_forwards_step_validation_options(self):
        config = parse_train(
            _compose(
                "train",
                "validation.enabled=true",
                "validation.every_n_steps=25",
                "validation.sanity_steps=2",
                "data.dataset.split_manifest=/tmp/splits.json",
            )
        )
        callbacks = []

        with (
            patch("scripts.train.entry_trainer") as entry,
            patch("scripts.train.build_logger") as logger,
        ):
            trainer = train_script.build_trainer(config, Path("/tmp/output"), callbacks)

        self.assertIs(trainer, entry.return_value)
        self.assertIs(entry.call_args.kwargs["logger"], logger.return_value)
        self.assertEqual(entry.call_args.kwargs["val_check_interval"], 25)
        self.assertEqual(entry.call_args.kwargs["num_sanity_val_steps"], 2)

    def test_train_uses_async_checkpoint(self):
        config = parse_train(_compose("train"))

        callbacks = train_script.training_callbacks(
            config,
            Path("/tmp/output"),
            Mock(),
        )

        checkpoint = next(
            callback
            for callback in callbacks
            if isinstance(callback, ModelCheckpoint)
        )
        self.assertTrue(checkpoint.async_save)
        self.assertFalse(checkpoint._enable_version_counter)

    @patch("scripts.train.build_trainer")
    @patch("scripts.train.training_callbacks", return_value=[])
    @patch("scripts.train.build_datamodule")
    @patch("scripts.train.apply_parameter_policy")
    @patch("scripts.train.rvq")
    @patch("scripts.train._composition", return_value=AcousticType.RVQ)
    @patch("scripts.train.Runtime")
    @patch("scripts.train.pl.seed_everything")
    def test_train_run_passes_ckpt_path_to_trainer_fit(
        self,
        seed,
        runtime,
        composition,
        rvq,
        policy,
        datamodule,
        callbacks,
        trainer_factory,
    ):
        del seed, composition, policy, callbacks
        config = parse_train(
            _compose(
                "train",
                "train.ckpt_path=/tmp/resume.ckpt",
                "trainer.enable_checkpointing=false",
            )
        )
        module = Mock()
        model = Mock()
        rvq.return_value = (module, model)
        runtime.return_value = Mock(acoustic_side_channel=False)
        trainer = Mock(is_global_zero=False)
        trainer_factory.return_value = trainer

        train_script.run(config)

        trainer.fit.assert_called_once_with(
            module,
            datamodule=datamodule.return_value,
            ckpt_path="/tmp/resume.ckpt",
        )

    def test_text_ar_uses_the_text_loader(self):
        self.assertTrue(_is_text_loader({Task.TEXT_AR: 1.0}))

    def test_removed_parallel_groups_are_not_composable(self):
        cases = [
            ("overfit", "codec=unicodec"),
            ("overfit", "sampler=smoke"),
            ("overfit", "optimizer=sft"),
            ("overfit", "init=random"),
            ("overfit", "trainer=overfit"),
        ]

        for config_name, override in cases:
            with self.subTest(config_name=config_name, override=override):
                with self.assertRaises(ConfigCompositionException):
                    _compose(config_name, override)

    def test_public_model_config_parses_domain_enums(self):
        config = overfit(
            _compose(
                "overfit",
                "model.semantic_audio_adapter=mlp",
                "model.semantic_audio_output_adapter=null",
            )
        )

        self.assertIs(config.model.semantic_audio_adapter, AdapterType.MLP)
        self.assertIsNone(config.model.semantic_audio_output_adapter)

        with self.assertRaises(ValueError):
            overfit(_compose("overfit", "model.semantic_audio_adapter=invalid"))

    def test_runtime_owns_codec_and_flow_sampling(self):
        config = overfit(
            _compose(
                "overfit",
                "runtime.flow_method=euler",
                "runtime.flow_nfe=4",
                "runtime.flow_num_steps=2",
            )
        )

        with patch.dict("os.environ", {"LOCAL_RANK": "1"}):
            runtime = runtime_config(config)

        self.assertEqual(runtime.codec, "longcat")
        self.assertEqual(runtime.device, "cuda:1")
        self.assertEqual(runtime.flow_method, "euler")
        self.assertEqual(runtime.flow_nfe, 4)
        self.assertEqual(runtime.flow_num_steps, 2)

    @patch("scripts.overfit.torch.cuda.set_device")
    def test_generation_module_restores_runtime_device_after_fit(
        self,
        set_device,
    ):
        module = _DeviceRestoreModule()
        device = torch.device("cuda:0")

        _prepare_generation_module(module, device)

        set_device.assert_called_once_with(device)
        self.assertEqual(module.moves, [device])

    def test_generation_module_without_runtime_device_stays_on_current_device(self):
        module = _DeviceRestoreModule()

        device = _prepare_generation_module(module, None)

        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(module.moves, [])

    def test_runtime_rejects_invalid_flow_settings_for_every_composition(self):
        for override in (
            "runtime.flow_method=invalid",
            "runtime.flow_nfe=0",
            "runtime.flow_num_steps=1",
        ):
            with self.subTest(override=override):
                raw = _compose(
                    "overfit",
                    "runtime=unicodec",
                    "model/acoustic=none",
                    override,
                )
                with self.assertRaises(ValueError):
                    overfit(raw)

    def test_full_codec_sequence_requires_token_acoustic_config(self):
        token = overfit(
            _compose(
                "overfit",
                "runtime=longcat_full_sequence",
                "model/acoustic=none",
            )
        )

        self.assertIsInstance(token, OverfitTokenConfig)
        self.assertEqual(token.runtime.audio_representation.value, "full_codec_sequence")
        with self.assertRaisesRegex(ValueError, "model/acoustic=none"):
            overfit(_compose("overfit", "runtime=longcat_full_sequence"))
        with self.assertRaisesRegex(ValueError, "model/acoustic=none"):
            overfit(
                _compose(
                    "overfit",
                    "runtime=longcat_full_sequence",
                    "model/acoustic=rvq",
                )
            )

    def test_semantic_codec_artifact_requires_token_only_longcat(self):
        token = overfit(
            _compose(
                "overfit",
                "runtime=longcat_native",
                "model/acoustic=none",
                "runtime.semantic_codec_artifact=/tmp/semantic-codec",
            )
        )

        self.assertEqual(
            token.runtime.semantic_codec_artifact,
            "/tmp/semantic-codec",
        )
        with self.assertRaisesRegex(ValueError, "semantic_codec_artifact"):
            overfit(
                _compose(
                    "overfit",
                    "runtime=longcat_native",
                    "model/acoustic=none",
                )
            )
        with self.assertRaisesRegex(ValueError, "model/acoustic=none"):
            overfit(
                _compose(
                    "overfit",
                    "runtime=longcat_native",
                    "runtime.semantic_codec_artifact=/tmp/semantic-codec",
                )
            )
        with self.assertRaisesRegex(ValueError, "decoupled"):
            overfit(
                _compose(
                    "overfit",
                    "runtime=unicodec",
                    "model/acoustic=none",
                    "runtime.semantic_codec_artifact=/tmp/semantic-codec",
                )
            )
        with self.assertRaisesRegex(ValueError, "decoupled"):
            overfit(
                _compose(
                    "overfit",
                    "runtime=longcat_full_sequence",
                    "model/acoustic=none",
                    "runtime.semantic_codec_artifact=/tmp/semantic-codec",
                )
            )

    def test_overfit_run_name_preserves_composition_and_decoder_depth(self):
        cases = [
            ((), "flow-8l"),
            (("model/acoustic=rvq",), "rvq-8l"),
            (("acoustic.decoder.layers=3",), "flow-3l"),
            (("runtime=unicodec", "model/acoustic=none"), "token"),
        ]

        for overrides, expected in cases:
            with self.subTest(expected=expected):
                config = overfit(_compose("overfit", *overrides))
                self.assertEqual(config.run_name, expected)
                self.assertEqual(Path(config.output_dir).name, expected)

    def test_overfit_jobs_use_the_token_safe_run_name(self):
        root = Path(__file__).parents[1]
        jobs = {"01_tts.sh": "tts", "02_s2st.sh": "s2st"}

        for filename, task in jobs.items():
            with self.subTest(job=filename):
                source = (root / "jobs" / "002" / filename).read_text()
                match = re.search(r'output_subdir="([^"]+)"', source)
                self.assertIsNotNone(match)
                subdir = match.group(1).replace(r"\${", "${")
                config = overfit(
                    _compose(
                        "overfit",
                        "runtime=unicodec",
                        "model/acoustic=none",
                        f"task={task}",
                        "repo_output_root=/tmp/train",
                        f"output_subdir={subdir}",
                    )
                )
                self.assertEqual(
                    config.output_dir,
                    f"/tmp/train/002-single-batch-overfit/{task}/token",
                )
                self.assertEqual(
                    config.logging.save_dir,
                    "/tmp/train/tensorboard",
                )
                self.assertEqual(
                    config.logging.run_name,
                    f"002-single-batch-overfit/{task}/token",
                )

    def test_training_jobs_override_root_and_relative_subdir(self):
        root = Path(__file__).parents[1]
        jobs = [*sorted((root / "jobs" / "002").glob("*.sh"))]
        jobs.extend(sorted((root / "jobs" / "005").glob("*.sh")))

        for path in jobs:
            with self.subTest(job=path.name):
                source = path.read_text()
                self.assertIn(
                    'repo_output_root="${SPEECH_TO_SPEECH_TRAIN_ROOT}"',
                    source,
                )
                match = re.search(r'output_subdir="([^"]+)"', source)
                self.assertIsNotNone(match)
                self.assertFalse(Path(match.group(1)).is_absolute())
                self.assertNotRegex(source, r"\boutput_dir=")

    def test_staged_joint_job_uses_train_entry(self):
        root = Path(__file__).parents[1]
        source = (root / "jobs" / "011" / "03_staged_joint_train.sh").read_text()

        self.assertIn("scripts/train.py", source)
        self.assertNotIn("scripts/overfit.py", source)
        self.assertIn('"trainer=static_ddp"', source)
        self.assertIn("fdu_stage_data_args data.dataset.root", source)
        self.assertIn("SPEECH_TO_SPEECH_STAGE:-stage_1", source)
        self.assertIn('experiment="train/staged_joint_${stage}"', source)
        self.assertIn('"experiment=${experiment}"', source)

    def test_fdu_project_jobs_source_workspace_environment(self):
        root = Path(__file__).parents[1]
        jobs = [
            *sorted((root / "jobs" / "011").glob("*.sh")),
            *sorted((root / "jobs" / "013").glob("*.sh")),
            *sorted((root / "jobs" / "014").glob("*.sh")),
        ]

        self.assertFalse((root / "jobs" / "013" / "fdu_env.sh").exists())
        for path in jobs:
            with self.subTest(job=str(path.relative_to(root))):
                source = path.read_text()

                self.assertIn("workspace/jobs/env.sh", source)
                self.assertNotRegex(
                    source,
                    r"/(?:home|mnt|Users)/|hf-mirror|Qwen3-0\.6B|HF_HOME|ANYTRAIN_HOME",
                )

    def test_stage1_resume_job_requires_and_forwards_checkpoint(self):
        root = Path(__file__).parents[1]
        jobs = {
            "03_stage1_pilot_resume_500.sh": "train/014_stage1_pilot_resume_500",
            "04_stage1_pilot_resume_2000.sh": "train/014_stage1_pilot_resume_2000",
            "06_stage1_pilot_fp32_resume_1000.sh": (
                "train/014_stage1_pilot_fp32_resume_1000"
            ),
        }

        for name, experiment in jobs.items():
            with self.subTest(job=name):
                source = (root / "jobs" / "014" / name).read_text()

                self.assertIn("SPEECH_TO_SPEECH_STAGE_CKPT_PATH:?", source)
                self.assertIn(f'"experiment={experiment}"', source)
                self.assertIn('"train.ckpt_path=${checkpoint}"', source)
                self.assertIn('"$@"', source)

    def test_stage1_fp32_job_starts_without_a_checkpoint(self):
        root = Path(__file__).parents[1]
        source = (
            root / "jobs" / "014" / "05_stage1_pilot_fp32_500.sh"
        ).read_text()

        self.assertIn('"experiment=train/014_stage1_pilot_fp32_500"', source)
        self.assertNotIn("SPEECH_TO_SPEECH_STAGE_CKPT_PATH", source)
        self.assertNotIn("train.ckpt_path=", source)
        self.assertIn('"$@"', source)

    def test_fdu_smoke_jobs_select_explicit_configs(self):
        root = Path(__file__).parents[1]
        jobs = [
            (
                "10_stage_0_smoke.sh",
                "fdu_stage_0_smoke",
                "scripts/overfit.py",
                "fdu_stage_data_args data.root",
            ),
            (
                "11_stage_1_smoke.sh",
                "train/fdu_stage_1_smoke",
                "scripts/train.py",
                "fdu_stage_data_args data.dataset.root",
            ),
            (
                "12_stage_2_smoke.sh",
                "train/fdu_stage_2_smoke",
                "scripts/train.py",
                "fdu_stage_data_args data.dataset.root",
            ),
            (
                "13_stage_3_smoke.sh",
                "train/fdu_stage_3_smoke",
                "scripts/train.py",
                "fdu_stage_data_args data.dataset.root",
            ),
            (
                "14_stage_4_smoke.sh",
                "train/fdu_stage_4_smoke",
                "scripts/train.py",
                "fdu_stage_data_args data.dataset.root",
            ),
            (
                "20_stage_0_acoustic_none_smoke.sh",
                "fdu_stage_0_acoustic_none_smoke",
                "scripts/overfit.py",
                "fdu_stage_data_args data.root",
            ),
            (
                "21_stage_1_acoustic_none_smoke.sh",
                "train/fdu_stage_1_acoustic_none_smoke",
                "scripts/train.py",
                "fdu_stage_data_args data.dataset.root",
            ),
            (
                "22_stage_2_acoustic_none_smoke.sh",
                "train/fdu_stage_2_acoustic_none_smoke",
                "scripts/train.py",
                "fdu_stage_data_args data.dataset.root",
            ),
            (
                "23_stage_3_acoustic_none_smoke.sh",
                "train/fdu_stage_3_acoustic_none_smoke",
                "scripts/train.py",
                "fdu_stage_data_args data.dataset.root",
            ),
            (
                "24_stage_4_acoustic_none_smoke.sh",
                "train/fdu_stage_4_acoustic_none_smoke",
                "scripts/train.py",
                "fdu_stage_data_args data.dataset.root",
            ),
        ]

        for filename, experiment, entry, data_call in jobs:
            with self.subTest(job=filename):
                source = (root / "jobs" / "013" / filename).read_text()

                self.assertIn("workspace/jobs/env.sh", source)
                self.assertIn(entry, source)
                self.assertEqual(
                    re.findall(r"\bexperiment=([a-z0-9_/]+)", source),
                    [experiment],
                )
                self.assertIn(data_call, source)
                self.assertIn(
                    '"repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}"',
                    source,
                )
                self.assertIn('"$@"', source)

    def test_jobs_default_the_training_root_to_dynamic_home_train(self):
        root = Path(__file__).parents[1]
        source = (root / "../workspace" / "jobs" / "env.sh").resolve().read_text()

        self.assertIn(
            'SPEECH_TO_SPEECH_TRAIN_ROOT="${SPEECH_TO_SPEECH_TRAIN_ROOT:-${DYNAMIC_HOME}/train/speech-to-speech}"',
            source,
        )

    def test_unicodec_jobs_require_a_compatible_python(self):
        root = Path(__file__).parents[1]
        env = (root / "../workspace" / "jobs" / "env.sh").resolve().read_text()
        jobs = {
            "02_unicodec.sh": ("unicodec_overfit", "overfit"),
            "05_unicodec_ddp.sh": ("unicodec_ddp_smoke", "ddp-smoke"),
        }

        self.assertNotIn(
            "SPEECH_TO_SPEECH_UNICODEC_PYTHON=",
            env,
        )
        for filename, (experiment, output_name) in jobs.items():
            with self.subTest(job=filename):
                source = (root / "jobs" / "005" / filename).read_text()
                config = overfit(_compose("overfit", f"experiment={experiment}"))
                self.assertIn(
                    "SPEECH_TO_SPEECH_UNICODEC_PYTHON:?Set ",
                    source,
                )
                self.assertTrue(config.output_subdir.endswith(f"/{output_name}"))
                self.assertIn(
                    f'output_subdir="{config.output_subdir}"',
                    source,
                )

    def test_unicodec_smoke_jobs_select_complete_experiments(self):
        root = Path(__file__).parents[1]
        jobs = {
            "02_unicodec.sh": "unicodec_overfit",
            "05_unicodec_ddp.sh": "unicodec_ddp_smoke",
        }

        for filename, expected in jobs.items():
            with self.subTest(job=filename):
                source = (root / "jobs" / "005" / filename).read_text()
                self.assertEqual(
                    re.findall(r"\bexperiment=([a-z0-9_]+)", source),
                    [expected],
                )

def _compose(config_name: str, *overrides: str) -> DictConfig:
    root = Path(__file__).parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        return compose(config_name=config_name, overrides=list(overrides))


if __name__ == "__main__":
    unittest.main()
