from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig
from omegaconf.errors import ConfigAttributeError, ConfigKeyError

from scripts._config import (
    CodecOracleConfig,
    OverfitFlowConfig,
    OverfitRVQConfig,
    OverfitTokenConfig,
    codec_oracle,
    overfit,
)
from scripts.codec_oracle import build_runtime as build_oracle_runtime
from scripts.overfit import _composition, runtime_config
from speech_to_speech.codec_oracle import Config as OracleConfig
from speech_to_speech.codec_oracle import DataConfig, Initialization
from speech_to_speech.datamodule import DatasetName
from speech_to_speech.model import (
    AdapterType,
    Config as ModelConfig,
    DecoderConfig,
    ToyConfig,
)
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.runtime import Config as RuntimeConfig


@patch.dict(
    "os.environ",
    {"SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer"},
)
class ConfigTest(unittest.TestCase):
    def test_roots_parse_to_src_aligned_configs(self):
        flow = overfit(_compose("overfit"))
        rvq = overfit(_compose("overfit", "model/acoustic=rvq"))
        token = overfit(
            _compose("overfit", "runtime=unicodec", "~model/acoustic")
        )
        oracle = codec_oracle(_compose("codec_oracle"))

        self.assertIsInstance(flow, OverfitFlowConfig)
        self.assertIsInstance(rvq, OverfitRVQConfig)
        self.assertIsInstance(token, OverfitTokenConfig)
        self.assertIsInstance(oracle, CodecOracleConfig)
        self.assertIsInstance(flow.runtime, RuntimeConfig)
        self.assertIsInstance(flow.model, ModelConfig)
        self.assertIsInstance(flow.pl_module, ModuleConfig)
        self.assertIsInstance(flow.acoustic.decoder, DecoderConfig)
        self.assertIsInstance(oracle.codec_oracle, OracleConfig)
        self.assertIsInstance(oracle.codec_oracle.data, DataConfig)
        self.assertEqual(flow.runtime.codec, "longcat")
        self.assertEqual(token.runtime.codec, "unicodec")
        self.assertIs(flow.model.semantic_audio_adapter, AdapterType.LINEAR)
        self.assertIs(
            oracle.codec_oracle.initialization,
            Initialization.CODEC,
        )
        self.assertFalse(hasattr(oracle, "acoustic"))

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
        self.assertFalse(config.callbacks.sample.enabled)
        self.assertFalse(config.callbacks.evaluation.enabled)

        production = overfit(_compose("overfit"))
        self.assertIsNone(production.model.toy)
        self.assertIs(production.data.name, DatasetName.WMT19_TTS)

        selected = overfit(_compose("overfit", "model=toy", "data=toy"))
        self.assertIsInstance(selected.model.toy, ToyConfig)
        self.assertIs(selected.data.name, DatasetName.TOY)

    def test_composition_must_match_codec_capabilities(self):
        flow = overfit(_compose("overfit"))
        token = overfit(
            _compose("overfit", "runtime=unicodec", "~model/acoustic")
        )

        self.assertIsNone(_composition(token, uses_acoustic_decoder=False))
        with self.assertRaisesRegex(ValueError, "model/acoustic=flow"):
            _composition(token, uses_acoustic_decoder=True)
        with self.assertRaisesRegex(ValueError, "remove the acoustic"):
            _composition(flow, uses_acoustic_decoder=False)

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
                codec_oracle,
                _compose("codec_oracle", "+acoustic.type=flow"),
                "acoustic",
            ),
        ]

        for parser, raw, key in cases:
            with self.subTest(key=key):
                with self.assertRaises((ConfigKeyError, ConfigAttributeError)) as raised:
                    parser(raw)
                self.assertIn(key, str(raised.exception))

    def test_oracle_rejects_audio_tokenizer_instead_of_ignoring_it(self):
        raw = _compose(
            "codec_oracle",
            "runtime.audio_tokenizer=/tmp/tokenizer",
        )

        with self.assertRaisesRegex(ValueError, "audio_tokenizer must be null"):
            codec_oracle(raw)

    def test_codec_oracle_root_is_the_production_training_default(self):
        config = codec_oracle(_compose("codec_oracle"))
        data = config.codec_oracle.data

        self.assertIsNone(data.sample_limit)
        self.assertEqual(data.batch_size, 8)
        self.assertEqual(data.num_workers, 0)
        self.assertFalse(data.pin_memory)
        self.assertFalse(data.persistent_workers)
        self.assertTrue(data.lba.enabled)
        self.assertEqual(data.lba.max_batch_seconds, 8.0)
        self.assertEqual(config.train.max_steps, 1_000_000)
        self.assertEqual(config.trainer.precision, "bf16-mixed")
        self.assertEqual(config.trainer.max_epochs, -1)
        self.assertEqual(config.trainer.log_every_n_steps, 10)
        self.assertEqual(config.callbacks.oracle.sample_every_n_steps, 10_000)
        self.assertEqual(config.callbacks.checkpoint.every_n_train_steps, 10_000)
        self.assertEqual(config.callbacks.checkpoint.save_top_k, -1)

    def test_codec_oracle_smoke_experiments_own_the_test_budgets(self):
        cases = [
            ("acoustic_oracle_smoke", 1, False, 1, "auto", True),
            (
                "acoustic_oracle_ddp_lba_smoke",
                32,
                True,
                "auto",
                "ddp",
                True,
            ),
        ]

        for experiment, sample_limit, lba, devices, strategy, sampler in cases:
            with self.subTest(experiment=experiment):
                config = codec_oracle(
                    _compose("codec_oracle", f"experiment={experiment}")
                )

                self.assertEqual(config.codec_oracle.data.sample_limit, sample_limit)
                self.assertIs(config.codec_oracle.data.lba.enabled, lba)
                self.assertEqual(config.train.max_steps, 2)
                self.assertEqual(config.runtime.flow_nfe, 4)
                self.assertEqual(config.runtime.flow_num_steps, 2)
                self.assertEqual(config.trainer.devices, devices)
                self.assertEqual(config.trainer.strategy, strategy)
                self.assertIs(config.trainer.use_distributed_sampler, sampler)
                self.assertEqual(
                    (
                        config.callbacks.oracle.sample_every_n_steps,
                        config.callbacks.oracle.histogram_every_n_steps,
                        config.callbacks.grad_norm.every_n_steps,
                        config.callbacks.checkpoint.every_n_train_steps,
                    ),
                    (1, 1, 1, 1),
                )

    def test_unicodec_experiments_close_the_token_training_chain(self):
        cases = [
            (
                "unicodec_overfit",
                100,
                1,
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
                self.assertTrue(config.callbacks.sample.enabled)
                self.assertEqual(config.callbacks.sample.every_n_steps, 1)

    def test_removed_parallel_groups_are_not_composable(self):
        cases = [
            ("overfit", "codec=unicodec"),
            ("overfit", "sampler=smoke"),
            ("overfit", "optimizer=sft"),
            ("overfit", "init=random"),
            ("overfit", "oracle=default"),
            ("overfit", "data/oracle@data=wmt19_tts"),
            ("overfit", "trainer=overfit"),
            ("codec_oracle", "codec_oracle=lba"),
            ("codec_oracle", "experiment=acoustic_oracle"),
            ("codec_oracle", "experiment=acoustic_oracle_ddp_lba"),
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
                "model.acoustic_prompt_adapter=MLP",
            )
        )

        self.assertIs(config.model.semantic_audio_adapter, AdapterType.MLP)
        self.assertIsNone(config.model.semantic_audio_output_adapter)
        self.assertIs(config.model.acoustic_prompt_adapter, AdapterType.MLP)

        with self.assertRaises(ValueError):
            overfit(_compose("overfit", "model.acoustic_prompt_adapter=invalid"))

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

        oracle = codec_oracle(
            _compose(
                "codec_oracle",
                "runtime.flow_nfe=4",
                "runtime.flow_num_steps=2",
            )
        )
        built = build_oracle_runtime(oracle, torch.device("cuda:0"))
        self.assertEqual(built.config.flow_nfe, 4)
        self.assertEqual(built.config.flow_num_steps, 2)

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
                    "~model/acoustic",
                    override,
                )
                with self.assertRaises(ValueError):
                    overfit(raw)

    def test_overfit_run_name_preserves_composition_and_decoder_depth(self):
        cases = [
            ((), "flow-8l"),
            (("model/acoustic=rvq",), "rvq-8l"),
            (("acoustic.decoder.layers=3",), "flow-3l"),
            (("runtime=unicodec", "~model/acoustic"), "token"),
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
                match = re.search(r'output_dir="([^"]+)"', source)
                self.assertIsNotNone(match)
                override = match.group(1).replace(
                    "${SPEECH_TO_SPEECH_TRAIN_ROOT}",
                    "/tmp/train",
                ).replace(r"\${", "${")
                config = overfit(
                    _compose(
                        "overfit",
                        "runtime=unicodec",
                        "~model/acoustic",
                        f"task={task}",
                        f"output_dir={override}",
                    )
                )
                self.assertEqual(
                    config.output_dir,
                    f"/tmp/train/002-single-batch-overfit/{task}/token",
                )

    def test_codec_screening_jobs_select_complete_experiments(self):
        root = Path(__file__).parents[1]
        jobs = {
            "01_longcat.sh": "acoustic_oracle_smoke",
            "02_unicodec.sh": "unicodec_overfit",
            "04_longcat_ddp_lba.sh": "acoustic_oracle_ddp_lba_smoke",
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
