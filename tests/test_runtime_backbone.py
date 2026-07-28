from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import torch

from speech_to_speech.runtime import BackboneInitialization, Config, Runtime


class RuntimeBackboneTest(unittest.TestCase):
    def test_random_initialization_loads_config_without_pretrained_weights(self):
        hf_config = Mock()
        backbone = Mock()
        moved = Mock()
        backbone.to.return_value = moved

        with (
            patch(
                "speech_to_speech.runtime.runtime.AutoConfig.from_pretrained",
                return_value=hf_config,
            ) as config_from_pretrained,
            patch(
                "speech_to_speech.runtime.runtime.AutoModelForCausalLM.from_config",
                return_value=backbone,
            ) as from_config,
            patch(
                "speech_to_speech.runtime.runtime.AutoModelForCausalLM.from_pretrained"
            ) as from_pretrained,
        ):
            runtime = Runtime(
                Config(
                    backbone="fake/backbone",
                    backbone_initialization=BackboneInitialization.RANDOM,
                    device="cuda",
                    dtype="bfloat16",
                    attn_implementation="flash_attention_2",
                )
            )

            loaded = runtime.backbone

        config_from_pretrained.assert_called_once_with("fake/backbone")
        from_config.assert_called_once_with(
            hf_config,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        from_pretrained.assert_not_called()
        backbone.to.assert_called_once_with("cuda")
        self.assertIs(loaded, moved)


if __name__ == "__main__":
    unittest.main()
