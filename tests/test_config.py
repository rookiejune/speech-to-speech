from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from anytrain.lightning import (
    GradientComparison,
    GradientProbe,
    GradientTarget,
    ModelCheckpoint,
)
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig
from omegaconf.errors import (
    ConfigAttributeError,
    ConfigKeyError,
    InterpolationResolutionError,
)
from peft import LoraConfig

from scripts._overfit_config import (
    OverfitFlowConfig,
    OverfitRVQConfig,
    OverfitTokenConfig,
    overfit,
)
from scripts._train_config import (
    StagedTrainRVQConfig,
    StagedTrainTokenConfig,
    train as parse_train,
)
from scripts._entry import (
    performance as build_performance,
    runtime_config,
)
from scripts._logging import build as build_logger
from scripts import train as train_script
from scripts.overfit import (
    _gradient_logger,
    _prepare_generation_module,
)
from scripts.train import (
    build_datamodule as build_train_datamodule,
)
from speech_to_speech.audio_route import (
    BICODEC_GENERATE_GLOBAL,
    BICODEC_REUSE_PROMPT_GLOBAL,
)
from speech_to_speech.datamodule import DataModule
from speech_to_speech.datamodule.module import LoaderKind
from speech_to_speech.datamodule.dataset.speech import DatasetName
from speech_to_speech.datamodule.types import DataShape
from speech_to_speech.model import (
    AdapterType,
    AudioInputAdapterType,
    AudioOutputAdapterType,
    Config as ModelConfig,
    ToyConfig,
)
from speech_to_speech.model.acoustic import AcousticType, DecoderConfig
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.runtime import (
    BackboneInitialization,
    BackboneType,
    Config as RuntimeConfig,
)
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
        token = overfit(
            _compose(
                "overfit",
                "runtime=unicodec",
                "model/acoustic=none",
                "audio_route=full_output",
            )
        )

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
        self.assertIs(config.data.dataset.name, DatasetName.TOY)
        self.assertEqual(config.data.dataset.toy_samples, 8)
        self.assertEqual(config.data.dataset.toy_frames, 4)
        self.assertEqual(config.train.max_steps, 2)
        self.assertFalse(config.callbacks.task_sample.enabled)
        self.assertFalse(config.callbacks.evaluation.enabled)

        production = overfit(_compose("overfit"))
        self.assertIsNone(production.model.toy)
        self.assertIs(production.data.dataset.name, DatasetName.WMT19_TTS)

        selected = overfit(
            _compose("overfit", "model=toy", "data@data.dataset=toy")
        )
        self.assertIsInstance(selected.model.toy, ToyConfig)
        self.assertIs(selected.data.dataset.name, DatasetName.TOY)

    def test_qwen2_5_omni_text_runtime_uses_thinker_adapter(self):
        config = overfit(_compose("overfit", "runtime=qwen2_5_omni_text"))

        self.assertIs(config.runtime.backbone_type, BackboneType.QWEN2_5_OMNI_THINKER)
        self.assertEqual(config.runtime.backbone, "Qwen/Qwen2.5-Omni-7B")
        self.assertEqual(config.runtime.backbone_body, "model")

    def test_kimi_audio_runtime_uses_tuple_readout(self):
        config = overfit(_compose("overfit", "runtime=kimi_audio"))

        self.assertEqual(config.runtime.backbone, "moonshotai/Kimi-Audio-7B-Instruct")
        self.assertTrue(config.runtime.backbone_trust_remote_code)
        self.assertEqual(config.runtime.backbone_readout, "last_hidden_state[1]")
        self.assertFalse(config.runtime.backbone_supports_cache_position)

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
                    "data@data.dataset=toy",
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
                "model.lora=null",
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
        self.assertIs(config.data.dataset.name, DatasetName.TOY)
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
        self.assertIs(config.data.dataset.name, DatasetName.TOY)

    def test_bicodec_smokes_use_qwen_single_speaker_cells(self):
        semantic = overfit(
            _compose("overfit", "experiment=bicodec_semantic_only_smoke")
        )
        full = overfit(
            _compose("overfit", "experiment=bicodec_full_sequence_smoke")
        )

        for config in (semantic, full):
            self.assertIsInstance(config, OverfitTokenConfig)
            self.assertEqual(config.runtime.codec, "bicodec")
            self.assertIs(
                config.data.dataset.name,
                DatasetName.QWEN_TTS_SPEAKER,
            )
            self.assertIs(config.data.shape, DataShape.SINGLE)
            self.assertIsNone(config.data.dataset.speaker)

        self.assertEqual(semantic.audio_route, BICODEC_REUSE_PROMPT_GLOBAL)
        self.assertEqual(full.audio_route, BICODEC_GENERATE_GLOBAL)
        self.assertEqual(semantic.run_name, "bicodec-reuse-prompt-global")
        self.assertEqual(full.run_name, "bicodec-generate-global")
        self.assertIn("bicodec-reuse-prompt-global-smoke", semantic.output_dir)
        self.assertIn("bicodec-generate-global-smoke", full.output_dir)
        self.assertIsNone(semantic.runtime.semantic_codec_artifact)
        self.assertIsNone(full.runtime.semantic_codec_artifact)

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

    @patch("scripts._entry.TrainingFlops")
    @patch("scripts._entry.PerformanceCallback")
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
        flow_comparison = GradientComparison(
            GradientTarget("token"),
            GradientTarget("flow_matching"),
        )

        gradient = _gradient_logger(default, AcousticType.FLOW, flow_comparison)

        self.assertIs(gradient, grad_logger.return_value)
        grad_logger.assert_called_once_with(
            (
                flow_comparison,
            ),
            _default_gradient_probes(),
            every_n_steps=1,
        )
        grad_logger.reset_mock()

        rvq_comparison = GradientComparison(
            GradientTarget("token"),
            GradientTarget("rvq"),
        )
        rvq_gradient = _gradient_logger(default, AcousticType.RVQ, rvq_comparison)

        self.assertIs(rvq_gradient, grad_logger.return_value)
        grad_logger.assert_called_once_with(
            (
                rvq_comparison,
            ),
            _default_gradient_probes(),
            every_n_steps=1,
        )
        grad_logger.reset_mock()

        self.assertIsNone(
            _gradient_logger(performance, AcousticType.FLOW, flow_comparison)
        )
        grad_logger.assert_not_called()

        frozen = overfit(_compose("overfit", "parameter_policy=speech_interface"))
        self.assertIsNone(_gradient_logger(frozen, AcousticType.RVQ, rvq_comparison))
        grad_logger.assert_not_called()

        partial = overfit(
            _compose("overfit", "parameter_policy=speech_interface_top_third")
        )
        partial_gradient = _gradient_logger(partial, AcousticType.RVQ, rvq_comparison)

        self.assertIs(partial_gradient, grad_logger.return_value)
        grad_logger.assert_called_once_with(
            (
                rvq_comparison,
            ),
            (GradientProbe("backbone_norm", ("model.backbone.model.norm.weight",)),),
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

    def test_train_root_uses_stage_config_and_accumulation_safe_ddp(self):
        default = parse_train(_compose("train"))

        self.assertIsInstance(default, StagedTrainRVQConfig)
        self.assertEqual(default.stage.name, StageName.STAGE_1)
        self.assertIs(default.parameter_policy.name, ParameterPolicyName.LORA)
        self.assertIsInstance(default.model.lora, LoraConfig)
        if default.model.lora is None:
            self.fail("default train config must enable PEFT LoRA")
        self.assertEqual(default.model.lora.init_lora_weights, "pissa")
        self.assertEqual(default.pl_module.optimizer, "adamw")
        self.assertFalse(default.validation.enabled)
        self.assertEqual(default.validation.loader, "tts")
        self.assertEqual(default.validation.split_label, "dev")
        self.assertEqual(default.validation.every_n_steps, 1000)
        self.assertEqual(default.validation.sanity_steps, -1)
        self.assertFalse(default.callbacks.performance.enabled)
        self.assertFalse(default.trainer.use_distributed_sampler)
        self.assertEqual(
            default.trainer.strategy,
            "ddp_find_unused_parameters_false",
        )
        self.assertTrue(default.stage.fuse_loaders_per_step)
        with self.assertRaises(AttributeError):
            getattr(default.data, "sample_index")

        config = parse_train(_compose("train", "stage=stage_2"))
        self.assertEqual(config.stage.name, StageName.STAGE_2)
        self.assertEqual(set(config.stage.loaders), {"asr", "tts", "mt"})
        self.assertEqual(config.stage.accumulate_grad_batches, 10)
        self.assertTrue(config.stage.fuse_loaders_per_step)
        self.assertEqual(config.data.codec, "longcat")
        self.assertEqual(config.data.dataset.name, DatasetName.WMT19_TTS)
        self.assertEqual(config.text_data.dataset.name.value, "wmt19")
        self.assertEqual(config.train.max_steps, 1000000)
        self.assertIsNone(config.train.ckpt_path)

        resumed = parse_train(_compose("train", "train.ckpt_path=/tmp/last.ckpt"))
        self.assertEqual(resumed.train.ckpt_path, "/tmp/last.ckpt")

        with self.assertRaisesRegex(ValueError, "unused-parameter"):
            parse_train(
                _compose(
                    "train",
                    "stage=stage_4",
                    "stage.fuse_loaders_per_step=false",
                    "trainer.strategy=ddp_find_unused_parameters_false",
                )
            )

        token = parse_train(
            _compose(
                "train",
                "runtime=longcat_full_sequence",
                "model/acoustic=none",
                "audio_route=full_output",
            )
        )

        self.assertIsInstance(token, StagedTrainTokenConfig)
        self.assertEqual(token.acoustic.type, AcousticType.NONE.value)
        self.assertEqual(token.run_name, "stage_1-token")
        with self.assertRaisesRegex(ValueError, "semantic_codec_artifact"):
            parse_train(_compose("train", "model/acoustic=none"))

    def test_parameter_policy_smoke_composes_each_supported_policy(self):
        policies = (
            ParameterPolicyName.FULL,
            ParameterPolicyName.LORA,
            ParameterPolicyName.SPEECH_INTERFACE,
            ParameterPolicyName.SPEECH_INTERFACE_TOP_THIRD,
            ParameterPolicyName.SEMANTIC_ONLY,
            ParameterPolicyName.ACOUSTIC_ONLY,
        )

        for policy in policies:
            with self.subTest(policy=policy.value):
                overrides = [
                    "experiment=train/parameter_policy_smoke",
                    f"parameter_policy={policy.value}",
                ]
                if policy is ParameterPolicyName.LORA:
                    overrides.append("model/lora=qwen")
                else:
                    overrides.append("model.lora=null")
                config = parse_train(
                    _compose("train", *overrides)
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
        panels = (
            (("asr", "asr"), ("tts", "tts")),
            (("asr", "asr"), ("tts", "tts"), ("mt", "mt")),
            (
                ("asr_s2tt", "asr"),
                ("asr_s2tt", "s2tt"),
                ("tts_t2st", "tts"),
                ("tts_t2st", "t2st"),
                ("mt", "mt"),
            ),
            (
                ("asr_s2tt", "asr"),
                ("asr_s2tt", "s2tt"),
                ("tts_t2st", "tts"),
                ("tts_t2st", "t2st"),
                ("s2st", "s2st"),
                ("mt", "mt"),
            ),
        )

        for index, (policy, expected_panels) in enumerate(
            zip(policies, panels),
            start=1,
        ):
            with self.subTest(stage=index):
                config = parse_train(
                    _compose(
                        "train",
                        f"experiment=train/staged_joint_stage_{index}",
                    )
                )

                self.assertIs(config.stage.name, StageName(f"stage_{index}"))
                self.assertIs(config.parameter_policy.name, policy)
                self.assertEqual(
                    config.trainer.strategy,
                    "ddp_find_unused_parameters_false",
                )
                self.assertGreater(config.stage.accumulate_grad_batches, 1)
                self.assertTrue(config.stage.fuse_loaders_per_step)
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
        with self.assertRaisesRegex(ValueError, "one loader branch per microbatch"):
            parse_train(
                _compose(
                    "train",
                    "stage=stage_4",
                    "stage.fuse_loaders_per_step=false",
                    "trainer.strategy=ddp_find_unused_parameters_false",
                )
            )

    def test_fused_multi_loader_requires_a_full_window(self):
        with self.assertRaisesRegex(ValueError, "too small"):
            parse_train(
                _compose(
                    "train",
                    "stage=stage_4",
                    "stage.accumulate_grad_batches=1",
                    "trainer.strategy=ddp_find_unused_parameters_false",
                )
            )

    def test_stable_codec_stage1_long_run_enables_fixed_samples_for_asr_and_tts(self):
        config = parse_train(
            _compose(
                "train",
                "experiment=train/stable_codec_stage1_train",
                "data.dataset.split_manifest=/tmp/splits.json",
            )
        )

        self.assertEqual(config.train.max_steps, 1_000_000)
        self.assertEqual(config.stage.name, StageName.STAGE_1)
        self.assertEqual(config.runtime.codec, "stable_codec")
        self.assertEqual(config.runtime.audio_representation.value, "full_codec_sequence")
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
            parse_train(
                _compose(
                    "train",
                    "experiment=train/stable_codec_stage1_train",
                )
            )

    def test_mt_sample_panel_requires_train_split(self):
        with self.assertRaisesRegex(ValueError, "validation.*speech loaders"):
            parse_train(
                _compose(
                    "train",
                    "stage=stage_2",
                    "callbacks.task_sample.enabled=true",
                    "callbacks.task_sample.panels=[{split:validation,loader:mt,task:mt,indices:[0]}]",
                    "validation.enabled=true",
                    "data.dataset.split_manifest=/tmp/splits.json",
                )
            )

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
        self.assertEqual(datamodule.schedule.accumulate_grad_batches, 10)
        self.assertTrue(datamodule.schedule.fuse_loaders_per_step)

        with self.assertRaisesRegex(ValueError, "cannot mix pure text and speech"):
            StageLoaderConfig(
                weight=1.0,
                task_weights={"mt": 1.0, "tts": 1.0},
            )

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

    def test_train_datamodule_builds_limited_wmt19_mt_validation(self):
        config = parse_train(
            _compose(
                "train",
                "stage=stage_2",
                "validation.enabled=true",
                "validation.loader=mt",
            )
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
        mt = parse_train(
            _compose(
                "train",
                "stage=stage_2",
                "validation.enabled=true",
                "validation.loader=mt",
            )
        )
        self.assertEqual(mt.validation.text_split, "validation")
        self.assertEqual(mt.validation.max_samples, 1000)
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
        self.assertEqual(entry.call_args.kwargs["accumulate_grad_batches"], 1)
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

    def test_train_constructs_gradient_probe_callback(self):
        config = parse_train(
            _compose(
                "train",
                "callbacks.gradient_probe.enabled=true",
            )
        )
        built = Mock()

        with patch("scripts.train.GradLogger", return_value=built) as factory:
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
                        r"model\.backbone\.model\.layers\.0\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.lora_[AB]\..*\.weight$",
                    ),
                    match="regex",
                ),
                GradientProbe(
                    "backbone_l0_ffn_lora",
                    (
                        r"model\.backbone\.model\.layers\.0\.mlp\.(gate_proj|up_proj|down_proj)\.lora_[AB]\..*\.weight$",
                    ),
                    match="regex",
                ),
            ),
            every_n_steps=10_000,
        )

    def test_train_performance_omits_gradient_probe_callback(self):
        config = parse_train(
            _compose(
                "train",
                "callbacks.gradient_probe.enabled=true",
            )
        )
        performance = Mock()

        with (
            patch("scripts.train.performance", return_value=performance),
            patch("scripts.train.GradLogger") as factory,
        ):
            callbacks = train_script.training_callbacks(
                config,
                Path("/tmp/output"),
                Mock(),
            )

        self.assertIs(callbacks[0], performance)
        factory.assert_not_called()

    @patch("scripts.train.build_trainer")
    @patch("scripts.train.training_callbacks", return_value=[])
    @patch("scripts.train.build_datamodule")
    @patch("scripts.train.apply_parameter_policy")
    @patch("scripts.train.build")
    @patch("scripts.train.Runtime")
    @patch("scripts.train.pl.seed_everything")
    def test_train_run_passes_ckpt_path_to_trainer_fit(
        self,
        seed,
        runtime,
        build,
        policy,
        datamodule,
        callbacks,
        trainer_factory,
    ):
        del seed, policy, callbacks
        config = parse_train(
            _compose(
                "train",
                "train.ckpt_path=/tmp/resume.ckpt",
                "trainer.enable_checkpointing=false",
            )
        )
        module = Mock()
        model = Mock()
        build.return_value = (AcousticType.RVQ, module, model)
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
        loader = StageLoaderConfig(
            weight=1.0,
            task_weights={"text_ar": 1.0},
        )

        self.assertTrue(loader.is_text)
        self.assertEqual(loader.tasks, {Task.TEXT_AR: 1.0})

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
                "model.audio_output_adapter.type=none",
            )
        )

        self.assertIs(config.model.semantic_audio_adapter, AdapterType.MLP)
        self.assertIs(
            config.model.audio_output_adapter.type,
            AudioOutputAdapterType.NONE,
        )

        with self.assertRaises(ValueError):
            overfit(_compose("overfit", "model.semantic_audio_adapter=invalid"))

    def test_audio_input_adapter_is_structured_and_disabled_by_default(self):
        default = overfit(_compose("overfit"))
        self.assertIs(
            default.model.audio_input_adapter.type,
            AudioInputAdapterType.NONE,
        )
        configured = overfit(
            _compose("overfit", "model.audio_input_adapter.type=transformer")
        )
        self.assertIs(
            configured.model.audio_input_adapter.type,
            AudioInputAdapterType.TRANSFORMER,
        )

    def test_audio_output_adapter_is_structured_and_linear_by_default(self):
        default = overfit(_compose("overfit"))
        self.assertIs(
            default.model.audio_output_adapter.type,
            AudioOutputAdapterType.LINEAR,
        )
        configured = overfit(
            _compose("overfit", "model.audio_output_adapter.type=mlp")
        )
        self.assertIs(
            configured.model.audio_output_adapter.type,
            AudioOutputAdapterType.MLP,
        )

    def test_lora_model_and_parameter_policy_must_be_selected_together(self):
        config = overfit(
            _compose(
                "overfit",
                "model/lora=qwen",
                "parameter_policy=lora",
            )
        )

        self.assertIsInstance(config.model.lora, LoraConfig)
        if config.model.lora is None:
            self.fail("PEFT LoRA config was not composed")
        self.assertEqual(config.model.lora.r, 16)
        self.assertEqual(config.model.lora.lora_alpha, 32)
        self.assertEqual(config.model.lora.init_lora_weights, "pissa")
        self.assertIs(config.parameter_policy.name, ParameterPolicyName.LORA)
        self.assertEqual(
            config.parameter_policy.trainable_groups,
            [
                ParameterGroup.BACKBONE_ADAPTER,
                ParameterGroup.SEMANTIC_AUDIO_EMBEDDING,
                ParameterGroup.SEMANTIC_AUDIO_ADAPTER,
                ParameterGroup.AUDIO_INPUT_ADAPTER,
                ParameterGroup.AUDIO_OUTPUT,
                ParameterGroup.ACOUSTIC_DECODER,
            ],
        )

        for overrides in (
            ("model/lora=qwen",),
            ("parameter_policy=lora",),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, "must be selected together"),
            ):
                overfit(_compose("overfit", *overrides))

    def test_lora_muon_requires_pissa_initialization(self):
        config = overfit(
            _compose(
                "overfit",
                "model/lora=qwen",
                "parameter_policy=lora",
                "pl_module.optimizer=muon",
            )
        )
        self.assertEqual(config.pl_module.optimizer, "muon")
        self.assertEqual(config.model.lora.init_lora_weights, "pissa")

        with self.assertRaisesRegex(ValueError, "pissa initialization"):
            overfit(
                _compose(
                    "overfit",
                    "model/lora=qwen",
                    "model.lora.init_lora_weights=gaussian",
                    "parameter_policy=lora",
                    "pl_module.optimizer=muon",
                )
            )

    def test_pl_module_optimizer_is_selectable(self):
        default = parse_train(_compose("train"))
        muon = parse_train(_compose("train", "pl_module.optimizer=muon"))

        self.assertEqual(default.pl_module.optimizer, "adamw")
        self.assertEqual(muon.pl_module.optimizer, "muon")
        self.assertEqual(muon.model.lora.init_lora_weights, "pissa")

    def test_lora_rejects_unsupported_performance_provider(self):
        with self.assertRaisesRegex(ValueError, "LoRA training FLOPs"):
            overfit(
                _compose(
                    "overfit",
                    "model/lora=qwen",
                    "parameter_policy=lora",
                    "callbacks.performance.enabled=true",
                )
            )

    def test_lora_rejects_peft_inference_mode_for_training(self):
        with self.assertRaisesRegex(ValueError, "inference_mode=false"):
            overfit(
                _compose(
                    "overfit",
                    "model/lora=qwen",
                    "+model.lora.inference_mode=true",
                    "parameter_policy=lora",
                )
            )

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
            runtime = runtime_config(config.runtime)

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
                    "audio_route=full_output",
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
                "audio_route=full_output",
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
            (
                (
                    "runtime=unicodec",
                    "model/acoustic=none",
                    "audio_route=full_output",
                ),
                "token",
            ),
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
                        "audio_route=full_output",
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
        self.assertIn('"trainer=staged_static_ddp"', source)
        self.assertIn("fdu_stage_data_args data.dataset.root", source)
        self.assertIn("SPEECH_TO_SPEECH_STAGE:-stage_1", source)
        self.assertIn('experiment="train/staged_joint_${stage}"', source)
        self.assertIn('"experiment=${experiment}"', source)
        self.assertIn('job_reject_overrides experiment task stage -- "$@"', source)

    def test_job_wrappers_source_existing_project_environment(self):
        root = Path(__file__).parents[1]
        env = root / "jobs" / "env.sh"
        jobs = sorted(path for path in (root / "jobs").rglob("*.sh") if path != env)

        self.assertTrue(env.is_file())
        self.assertTrue(env.stat().st_mode & 0o111)
        self.assertTrue(jobs)
        self.assertFalse((root / "jobs" / "013" / "fdu_env.sh").exists())
        for path in jobs:
            with self.subTest(job=str(path.relative_to(root))):
                source = path.read_text()
                jobs_dir = next(parent for parent in path.parents if parent.name == "jobs")

                self.assertEqual(jobs_dir / "env.sh", env)
                self.assertIn(
                    'JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
                    source,
                )
                self.assertIn('source "${JOB_DIR%/jobs/*}/jobs/env.sh"', source)
                self.assertNotIn("workspace/jobs/env.sh", source)
                self.assertTrue(path.stat().st_mode & 0o111)
                self.assertNotRegex(
                    source,
                    r"/(?:home|mnt|Users)/|hf-mirror|Qwen3-0\.6B|HF_HOME|ANYTRAIN_HOME",
                )

    def test_project_jobs_env_owns_speech_settings(self):
        root = Path(__file__).parents[1]
        project_env = (root / "jobs" / "env.sh").read_text()
        workspace_env = (root / "../workspace" / "jobs" / "env.sh").resolve().read_text()

        self.assertIn('source "${REPOS_ROOT}/workspace/jobs/env.sh"', project_env)
        for name in (
            "SPEECH_TO_SPEECH_ROOT",
            "SPEECH_TO_SPEECH_PYTHON",
            "SPEECH_TO_SPEECH_TRAIN_ROOT",
            "SPEECH_TO_SPEECH_AUDIO_TOKENIZER",
            "job_reject_overrides",
            "fdu_stage_data_args",
            "fdu_qwen_root",
        ):
            with self.subTest(name=name):
                self.assertIn(name, project_env)
                self.assertNotIn(name, workspace_env)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", workspace_env)

    def test_jobs_default_the_training_root_to_dynamic_home_train(self):
        root = Path(__file__).parents[1]
        source = (root / "jobs" / "env.sh").read_text()

        self.assertIn(
            'SPEECH_TO_SPEECH_TRAIN_ROOT="${SPEECH_TO_SPEECH_TRAIN_ROOT:-${DYNAMIC_HOME}/train/speech-to-speech}"',
            source,
        )

    def test_unicodec_jobs_require_a_compatible_python(self):
        root = Path(__file__).parents[1]
        env = (root / "jobs" / "env.sh").read_text()
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


def _default_gradient_probes() -> tuple[GradientProbe, ...]:
    return (
        GradientProbe(
            "backbone_l0_attn",
            (
                "model.backbone.model.layers.0.self_attn.q_proj.weight",
                "model.backbone.model.layers.0.self_attn.k_proj.weight",
                "model.backbone.model.layers.0.self_attn.v_proj.weight",
                "model.backbone.model.layers.0.self_attn.o_proj.weight",
            ),
        ),
        GradientProbe(
            "backbone_l0_ffn",
            (
                "model.backbone.model.layers.0.mlp.gate_proj.weight",
                "model.backbone.model.layers.0.mlp.up_proj.weight",
                "model.backbone.model.layers.0.mlp.down_proj.weight",
            ),
        ),
    )


def _compose(config_name: str, *overrides: str) -> DictConfig:
    root = Path(__file__).parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        return compose(config_name=config_name, overrides=list(overrides))


if __name__ == "__main__":
    unittest.main()
