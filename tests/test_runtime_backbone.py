from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from speech_to_speech.runtime import BackboneInitialization, BackboneType, Config, Runtime


class RuntimeBackboneTest(unittest.TestCase):
    def test_random_initialization_loads_config_without_pretrained_weights(self):
        hf_config = Mock()
        backbone = Mock()
        moved = Mock()
        backbone.to.return_value = moved

        with (
            patch(
                "speech_to_speech.runtime.backbone.hf.AutoConfig.from_pretrained",
                return_value=hf_config,
            ) as config_from_pretrained,
            patch(
                "speech_to_speech.runtime.backbone.hf.AutoModelForCausalLM.from_config",
                return_value=backbone,
            ) as from_config,
            patch(
                "speech_to_speech.runtime.backbone.hf.AutoModelForCausalLM.from_pretrained"
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

        config_from_pretrained.assert_called_once_with(
            "fake/backbone",
            trust_remote_code=False,
        )
        from_config.assert_called_once_with(
            hf_config,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        from_pretrained.assert_not_called()
        backbone.to.assert_called_once_with("cuda")
        self.assertIs(loaded, moved)

    def test_pretrained_initialization_forwards_trust_remote_code(self):
        backbone = Mock()

        with (
            patch(
                "speech_to_speech.runtime.backbone.hf.AutoModelForCausalLM.from_pretrained",
                return_value=backbone,
            ) as from_pretrained,
            patch(
                "speech_to_speech.runtime.backbone.hf.AutoModelForCausalLM.from_config"
            ) as from_config,
            patch(
                "speech_to_speech.runtime.backbone.hf.AutoConfig.from_pretrained"
            ) as config_from_pretrained,
        ):
            runtime = Runtime(
                Config(
                    backbone="fake/remote-backbone",
                    backbone_trust_remote_code=True,
                )
            )

            loaded = runtime.backbone

        from_pretrained.assert_called_once_with(
            "fake/remote-backbone",
            trust_remote_code=True,
        )
        from_config.assert_not_called()
        config_from_pretrained.assert_not_called()
        self.assertIs(loaded, backbone)

    def test_qwen_omni_uses_processor_tokenizer_and_thinker_model(self):
        tokenizer = Mock()
        tokenizer.bos_token_id = 1
        processor = Mock(tokenizer=tokenizer)
        backbone = Mock()
        body = Mock()
        backbone.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=3584),
        )
        backbone.model = body
        model_class = Mock()
        model_class.from_pretrained.return_value = backbone

        with (
            patch(
                "speech_to_speech.runtime.backbone.hf.AutoProcessor.from_pretrained",
                return_value=processor,
            ) as processor_from_pretrained,
            patch(
                "speech_to_speech.runtime.backbone.hf._omni_model_factory",
                return_value=model_class,
            ),
        ):
            runtime = Runtime(
                Config(
                    backbone_type=BackboneType.QWEN2_5_OMNI_THINKER,
                    backbone="Qwen/Qwen2.5-Omni-7B",
                    backbone_body="model",
                    dtype="bfloat16",
                )
            )

            loaded_tokenizer = runtime.text_tokenizer
            loaded_backbone = runtime.backbone
            hidden_size = runtime.backbone_adapter.hidden_size

        processor_from_pretrained.assert_called_once_with(
            "Qwen/Qwen2.5-Omni-7B",
            trust_remote_code=False,
        )
        model_class.from_pretrained.assert_called_once_with(
            "Qwen/Qwen2.5-Omni-7B",
            trust_remote_code=False,
            dtype=torch.bfloat16,
        )
        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertIs(loaded_backbone, backbone)
        self.assertEqual(hidden_size, 3584)

    def test_qwen_omni_random_initialization_uses_thinker_config_factory(self):
        hf_config = Mock()
        backbone = Mock()
        model_class = Mock()
        model_class._from_config.return_value = backbone

        with (
            patch(
                "speech_to_speech.runtime.backbone.hf.AutoConfig.from_pretrained",
                return_value=hf_config,
            ) as config_from_pretrained,
            patch(
                "speech_to_speech.runtime.backbone.hf._omni_model_factory",
                return_value=model_class,
            ),
        ):
            runtime = Runtime(
                Config(
                    backbone_type=BackboneType.QWEN2_5_OMNI_THINKER,
                    backbone="Qwen/Qwen2.5-Omni-7B",
                    backbone_initialization=BackboneInitialization.RANDOM,
                    dtype="bfloat16",
                )
            )

            loaded = runtime.backbone

        config_from_pretrained.assert_called_once_with(
            "Qwen/Qwen2.5-Omni-7B",
            trust_remote_code=False,
        )
        model_class._from_config.assert_called_once_with(
            hf_config,
            dtype=torch.bfloat16,
        )
        model_class.from_pretrained.assert_not_called()
        self.assertIs(loaded, backbone)


if __name__ == "__main__":
    unittest.main()
