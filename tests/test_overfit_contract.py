from __future__ import annotations

import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from scripts._overfit_config import overfit
from scripts.overfit import (
    _gradient_logger,
    build_datamodule,
    training_callbacks,
)
from speech_to_speech.datamodule.config import SpeechConfig
from speech_to_speech.datamodule.dataset import DatasetName
from speech_to_speech.datamodule.types import DataShape
from speech_to_speech.model.acoustic import AcousticType
from speech_to_speech.task import Task


@patch.dict(
    "os.environ",
    {
        "DYNAMIC_HOME": "/tmp/dynamic",
        "SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer",
    },
)
class OverfitContractTest(unittest.TestCase):
    def test_overfit_uses_the_shared_speech_data_contract(self) -> None:
        config = overfit(_compose("overfit", "experiment=toy_smoke"))

        self.assertIsInstance(config.data, SpeechConfig)
        self.assertEqual(config.data.codec, config.runtime.codec)
        self.assertEqual(config.data.dataloader.batch_size, 1)
        self.assertEqual(config.data.dataloader.num_workers, 0)
        self.assertFalse(config.data.dataloader.pin_memory)
        self.assertFalse(config.data.dataloader.persistent_workers)
        self.assertIs(config.data.shape, DataShape.PAIR)
        self.assertFalse(config.data.encode_missing_codes)
        self.assertIs(config.data.dataset.name, DatasetName.TOY)
        self.assertEqual(config.sample_index, 0)
        with self.assertRaises(AttributeError):
            getattr(config.data, "sample_index")

    def test_flat_overfit_experiments_package_dataset_and_sample_separately(
        self,
    ) -> None:
        root = Path(__file__).parents[1]
        experiments = sorted((root / "configs" / "experiment").glob("*.yaml"))

        self.assertEqual(
            {path.stem for path in experiments},
            {
                "bicodec_full_sequence_smoke",
                "bicodec_semantic_only_smoke",
                "longcat_decoupled_semantic_only_smoke",
                "longcat_full_sequence_smoke",
                "overfit",
                "toy_smoke",
                "unicodec_ddp_smoke",
                "unicodec_overfit",
            },
        )
        for path in experiments:
            with self.subTest(experiment=path.name):
                source = path.read_text()
                self.assertIn("- /data@data.dataset:", source)
                self.assertRegex(source, r"(?m)^sample_index: 0$")
                self.assertNotRegex(source, r"(?m)^  sample_index:")

    def test_datamodule_receives_the_parsed_speech_config_directly(self) -> None:
        config = overfit(_compose("overfit"))
        runtime = Mock()
        loader = Mock()
        datamodule = Mock()

        with (
            patch("scripts.overfit.LoaderSpec.speech", return_value=loader) as speech,
            patch("scripts.overfit.DataModule", return_value=datamodule) as factory,
        ):
            built = build_datamodule(config, runtime, Task.TTS)

        self.assertIs(built, datamodule)
        speech.assert_called_once_with(
            config.data,
            {Task.TTS: 1.0},
            sample_index=config.sample_index,
        )
        factory.assert_called_once_with(runtime, {"train": loader})

    def test_optional_diagnostics_use_configured_values_and_stable_order(self) -> None:
        config = overfit(
            _compose(
                "overfit",
                "callbacks.task_sample.every_n_steps=2",
                "callbacks.task_sample.every_audio_seconds=2.5",
                "callbacks.text_retention.every_n_steps=3",
                "callbacks.text_retention.every_audio_seconds=3.5",
                "callbacks.text_retention.max_new_tokens=9",
                "callbacks.grad_norm.every_n_steps=4",
                "callbacks.grad_norm.every_audio_seconds=4.5",
                "callbacks.gradient_pair.every_n_steps=5",
                "callbacks.gradient_pair.every_audio_seconds=5.5",
                "callbacks.gradient_pair.full_parameter=full.weight",
                "callbacks.flow_matching.every_n_steps=6",
                "callbacks.flow_matching.every_audio_seconds=6.5",
            )
        )
        runtime = Mock()
        summary = Mock()
        evaluation = Mock()

        with ExitStack() as stack:
            factories = {
                name: stack.enter_context(patch(f"scripts.overfit.{name}"))
                for name in (
                    "OOMDiagnostics",
                    "OutputsLogger",
                    "FlowMatchingLogger",
                    "GradLogger",
                    "GradNormLogger",
                    "TaskSampleLogger",
                    "TextRetentionLogger",
                )
            }
            performance = stack.enter_context(
                patch("scripts.overfit.performance", return_value=None)
            )
            callbacks = training_callbacks(
                config,
                runtime,
                acoustic_type=AcousticType.FLOW,
                loss_pair=("token", "flow_matching"),
                task=Task.TTS,
                summary=summary,
                evaluation=evaluation,
            )

        self.assertEqual(
            callbacks,
            [
                factories["OOMDiagnostics"].return_value,
                factories["OutputsLogger"].return_value,
                factories["FlowMatchingLogger"].return_value,
                factories["GradLogger"].return_value,
                factories["GradNormLogger"].return_value,
                factories["TaskSampleLogger"].return_value,
                factories["TextRetentionLogger"].return_value,
                summary,
                evaluation,
            ],
        )
        performance.assert_called_once_with(config.callbacks.performance)
        factories["FlowMatchingLogger"].assert_called_once_with(
            runtime.flow_matching,
            every_n_steps=6,
            every_audio_seconds=6.5,
        )
        factories["GradLogger"].assert_called_once_with(
            ("token", "flow_matching"),
            "full.weight",
            every_n_steps=5,
            every_audio_seconds=5.5,
        )
        factories["GradNormLogger"].assert_called_once_with(
            every_n_steps=4,
            every_audio_seconds=4.5,
        )
        factories["TaskSampleLogger"].assert_called_once_with(
            [config.sample_index],
            every_n_steps=2,
            loader_name="train",
            task=Task.TTS,
            every_audio_seconds=2.5,
        )
        factories["TextRetentionLogger"].assert_called_once_with(
            {
                name: {
                    "instruction": probe.instruction,
                    "reference": probe.reference,
                }
                for name, probe in config.callbacks.text_retention.probes.items()
            },
            every_n_steps=3,
            every_audio_seconds=3.5,
            max_new_tokens=9,
        )

    def test_structural_callbacks_remain_when_diagnostics_are_disabled(self) -> None:
        config = overfit(
            _compose(
                "overfit",
                "callbacks.task_sample.enabled=false",
                "callbacks.evaluation.enabled=false",
                "callbacks.text_retention.enabled=false",
                "callbacks.grad_norm.enabled=false",
                "callbacks.gradient_pair.enabled=false",
                "callbacks.flow_matching.enabled=false",
            )
        )
        runtime = Mock()
        summary = Mock()
        oom = Mock()
        outputs = Mock()

        with (
            patch("scripts.overfit.performance", return_value=None),
            patch("scripts.overfit.OOMDiagnostics", return_value=oom),
            patch("scripts.overfit.OutputsLogger", return_value=outputs),
        ):
            callbacks = training_callbacks(
                config,
                runtime,
                acoustic_type=AcousticType.FLOW,
                loss_pair=("token", "flow_matching"),
                task=Task.TTS,
                summary=summary,
                evaluation=None,
            )

        self.assertEqual(callbacks, [oom, outputs, summary])

    def test_oom_follows_performance_before_other_callbacks(self) -> None:
        config = overfit(
            _compose(
                "overfit",
                "callbacks.performance.enabled=true",
                "callbacks.task_sample.enabled=false",
            )
        )
        performance = Mock()
        oom = Mock()

        with (
            patch("scripts.overfit.performance", return_value=performance),
            patch("scripts.overfit.OOMDiagnostics", return_value=oom),
        ):
            callbacks = training_callbacks(
                config,
                Mock(),
                acoustic_type=AcousticType.NONE,
                loss_pair=None,
                task=Task.TTS,
                summary=Mock(),
                evaluation=None,
            )

        self.assertIs(callbacks[0], performance)
        self.assertIs(callbacks[1], oom)

    def test_partial_policy_uses_the_configured_partial_parameter(self) -> None:
        config = overfit(
            _compose(
                "overfit",
                "parameter_policy=speech_interface_top_third",
                "model/acoustic=rvq",
                "callbacks.gradient_pair.partial_parameter=partial.weight",
            )
        )

        with patch("scripts.overfit.GradLogger") as logger:
            built = _gradient_logger(
                config,
                AcousticType.RVQ,
                ("token", "rvq"),
            )

        self.assertIs(built, logger.return_value)
        logger.assert_called_once_with(
            ("token", "rvq"),
            "partial.weight",
            every_n_steps=1,
        )

    def test_exposed_overfit_values_are_validated_during_parse(self) -> None:
        cases = (
            ("sample_index=-1", ValueError, "sample_index"),
            ("data.dataloader.batch_size=0", ValueError, "batch_size"),
            ("data.dataloader.num_workers=-1", ValueError, "num_workers"),
            (
                "callbacks.text_retention.every_n_steps=0",
                ValueError,
                "every_n_steps",
            ),
            (
                "callbacks.text_retention.max_new_tokens=0",
                ValueError,
                "max_new_tokens",
            ),
            (
                "callbacks.grad_norm.every_n_steps=0",
                ValueError,
                "callbacks.grad_norm.every_n_steps",
            ),
            (
                "callbacks.gradient_pair.every_n_steps=0",
                ValueError,
                "callbacks.gradient_pair.every_n_steps",
            ),
            (
                "callbacks.gradient_pair.every_audio_seconds=0",
                ValueError,
                "callbacks.gradient_pair.every_audio_seconds",
            ),
            (
                "callbacks.gradient_pair.full_parameter=''",
                TypeError,
                "callbacks.gradient_pair.full_parameter",
            ),
            (
                "callbacks.flow_matching.every_n_steps=0",
                ValueError,
                "callbacks.flow_matching.every_n_steps",
            ),
            (
                "callbacks.flow_matching.every_audio_seconds=0",
                ValueError,
                "callbacks.flow_matching.every_audio_seconds",
            ),
        )

        for override, error, message in cases:
            with (
                self.subTest(override=override),
                self.assertRaisesRegex(error, message),
            ):
                overfit(_compose("overfit", override))

        with self.assertRaisesRegex(ValueError, "must differ"):
            overfit(
                _compose(
                    "overfit",
                    "callbacks.gradient_pair.partial_parameter="
                    "model.backbone.model.layers.0.self_attn.q_proj.weight",
                )
            )


def _compose(config_name: str, *overrides: str) -> DictConfig:
    root = Path(__file__).parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        return compose(config_name=config_name, overrides=list(overrides))


if __name__ == "__main__":
    unittest.main()
