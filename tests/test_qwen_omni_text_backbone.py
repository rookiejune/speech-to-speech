from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import torch
from omegaconf import OmegaConf
from torch import nn
from transformers import PretrainedConfig

from speech_to_speech.runtime.backbone import hf
from speech_to_speech.runtime.backbone.config import AdapterConfig, BackboneInitialization, BackboneType


class QwenOmniTextBackboneTest(unittest.TestCase):
    def test_uses_processor_text_tokenizer(self):
        tokenizer = Mock(bos_token_id=1)
        processor = SimpleNamespace(tokenizer=tokenizer)
        adapter = hf.HuggingFaceBackboneAdapter(
            AdapterConfig(
                type=BackboneType.QWEN2_5_OMNI_TEXT,
                path="fake/qwen-omni",
            )
        )

        with (
            patch.object(hf.AutoProcessor, "from_pretrained", return_value=processor) as load,
            patch.object(hf.AutoTokenizer, "from_pretrained") as load_tokenizer,
        ):
            loaded = adapter.text_tokenizer

        load.assert_called_once_with("fake/qwen-omni", trust_remote_code=False)
        load_tokenizer.assert_not_called()
        self.assertIs(loaded, tokenizer)

    def test_pretrained_loads_only_nested_text_model_weights(self):
        text_config = PretrainedConfig(hidden_size=32)
        root_config = SimpleNamespace(
            thinker_config=SimpleNamespace(text_config=text_config)
        )
        backbone = object()
        factory = Mock()
        factory.from_pretrained.return_value = backbone
        adapter = hf.HuggingFaceBackboneAdapter(
            AdapterConfig(
                type=BackboneType.QWEN2_5_OMNI_TEXT,
                path="fake/qwen-omni",
                trust_remote_code=True,
            )
        )

        with (
            patch.object(hf.AutoConfig, "from_pretrained", return_value=root_config) as load,
            patch.object(hf, "_omni_text_model_factory", return_value=factory),
        ):
            loaded = adapter._pretrained(
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )

        load.assert_called_once_with("fake/qwen-omni", trust_remote_code=True)
        factory.from_pretrained.assert_called_once_with(
            "fake/qwen-omni",
            config=text_config,
            key_mapping={r"^thinker\.model\.": ""},
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        factory._from_config.assert_not_called()
        self.assertIs(loaded, backbone)

    def test_random_initialization_uses_nested_text_config(self):
        text_config = PretrainedConfig(hidden_size=32)
        root_config = SimpleNamespace(
            thinker_config=SimpleNamespace(text_config=text_config)
        )
        backbone = object()
        factory = Mock()
        factory._from_config.return_value = backbone
        adapter = hf.HuggingFaceBackboneAdapter(
            AdapterConfig(
                type=BackboneType.QWEN2_5_OMNI_TEXT,
                path="fake/qwen-omni",
                initialization=BackboneInitialization.RANDOM,
            )
        )

        with (
            patch.object(hf.AutoConfig, "from_pretrained", return_value=root_config) as load,
            patch.object(hf, "_omni_text_model_factory", return_value=factory),
        ):
            loaded = adapter._random(
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )

        load.assert_called_once_with("fake/qwen-omni", trust_remote_code=False)
        factory._from_config.assert_called_once_with(
            text_config,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        factory.from_pretrained.assert_not_called()
        self.assertIs(loaded, backbone)

    def test_factory_builds_only_tiny_text_body_on_meta(self):
        from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
            Qwen2_5OmniTextConfig,
        )

        config = Qwen2_5OmniTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=32,
        )

        with torch.device("meta"):
            model = cast(
                nn.Module,
                hf._omni_text_model_factory()._from_config(
                    config,
                    dtype=torch.bfloat16,
                    attn_implementation="eager",
                ),
            )

        self.assertEqual(type(model).__name__, "Qwen2_5OmniThinkerTextModel")
        self.assertEqual(
            [name for name, _ in model.named_children()],
            ["embed_tokens", "layers", "norm", "rotary_emb"],
        )
        self.assertTrue(all(parameter.device.type == "meta" for parameter in model.parameters()))
        self.assertTrue(all(parameter.dtype is torch.bfloat16 for parameter in model.parameters()))

    def test_preset_selects_bicodec_text_only_runtime(self):
        path = Path(__file__).resolve().parents[1] / "configs/runtime/qwen2_5_omni_text.yaml"
        config = OmegaConf.load(path)

        self.assertEqual(config.codec, "bicodec")
        self.assertEqual(config.backbone_type, "qwen2_5_omni_text")
        self.assertEqual(config.backbone_module, "")
        self.assertEqual(config.backbone_body, "base_model")
        self.assertTrue(config.gradient_checkpointing)


if __name__ == "__main__":
    unittest.main()
