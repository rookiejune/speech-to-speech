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
class RuntimeConfigContractTest(ConfigTestCase):
    def test_runtime_owns_codec_and_flow_sampling(self):
        config = _overfit(
            "runtime.flow_method=euler",
            "runtime.flow_nfe=4",
            "runtime.flow_num_steps=2",
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
                    "audio_sequence_layout=flattened",
                    override,
                )
                with self.assertRaises(ValueError):
                    overfit(raw)

    def test_full_codec_sequence_requires_token_acoustic_config(self):
        token = overfit(
            _compose(
                "overfit",
                "runtime=longcat_native",
                "model/acoustic=none",
                "audio_sequence_layout=flattened",
            )
        )

        self.assertIsInstance(token, OverfitTokenConfig)
        self.assertIs(token.audio_sequence_layout, AudioSequenceLayout.FLATTENED)
        with self.assertRaisesRegex(ValueError, "model/acoustic=none"):
            overfit(
                _compose(
                    "overfit",
                    "runtime=longcat_native",
                    "audio_sequence_layout=flattened",
                )
            )
        with self.assertRaisesRegex(ValueError, "model/acoustic=none"):
            overfit(
                _compose(
                    "overfit",
                    "runtime=longcat_native",
                    "audio_sequence_layout=flattened",
                    "model/acoustic=rvq",
                )
            )

    def test_acoustic_generator_artifact_requires_token_only_longcat(self):
        token = overfit(
            _compose(
                "overfit",
                "runtime=longcat_native",
                "model/acoustic=none",
                "runtime.acoustic_generator_artifact=/tmp/semantic-codec",
            )
        )

        self.assertEqual(
            token.runtime.acoustic_generator_artifact,
            "/tmp/semantic-codec",
        )
        with self.assertRaisesRegex(ValueError, "acoustic_generator_artifact"):
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
                    "runtime.acoustic_generator_artifact=/tmp/semantic-codec",
                )
            )
        with self.assertRaisesRegex(ValueError, "acoustic generator artifacts"):
            overfit(
                _compose(
                    "overfit",
                    "runtime=unicodec",
                    "model/acoustic=none",
                    "runtime.acoustic_generator_artifact=/tmp/semantic-codec",
                )
            )
        with self.assertRaisesRegex(ValueError, "audio_sequence_layout=semantic"):
            overfit(
                _compose(
                    "overfit",
                    "runtime=longcat_native",
                    "audio_sequence_layout=flattened",
                    "model/acoustic=none",
                    "runtime.acoustic_generator_artifact=/tmp/semantic-codec",
                )
            )


if __name__ == "__main__":
    unittest.main()
