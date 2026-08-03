from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer

from speech_to_speech.runtime import BackboneInitialization, Config, Runtime


class GradientCheckpointingBackbone(PreTrainedModel):
    config_class = PretrainedConfig
    supports_gradient_checkpointing = True

    def __init__(self) -> None:
        super().__init__(PretrainedConfig(use_cache=True))
        self.gradient_checkpointing_calls = 0
        self.gradient_checkpointing_kwargs: dict[str, object] | None = None
        self.input_require_grads_calls = 0
        self.moves: list[str] = []

    def gradient_checkpointing_enable(self, *args: object, **kwargs: object) -> None:
        del args
        self.gradient_checkpointing_calls += 1
        self.gradient_checkpointing_kwargs = kwargs

    def enable_input_require_grads(self) -> None:
        self.input_require_grads_calls += 1

    def to(self, device: str):  # type: ignore[override]
        self.moves.append(device)
        return self


class ExternalAdapter(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = SimpleNamespace(use_cache=True)


class CheckpointingLayer(GradientCheckpointingLayer):
    pass


class LayerCheckpointingBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = CheckpointingLayer()
        self.config = SimpleNamespace(use_cache=True)


class BodyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
        self.config = SimpleNamespace(hidden_size=4)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


class RuntimeBackboneTest(unittest.TestCase):
    def test_loading_uses_the_base_model_body_directly(self):
        backbone = BodyBackbone()

        with patch(
            "speech_to_speech.runtime.backbone.hf.AutoModel.from_pretrained",
            return_value=backbone,
        ) as load:
            loaded = Runtime(Config(backbone="fake/backbone")).backbone

        load.assert_called_once_with(
            "fake/backbone",
            trust_remote_code=False,
        )
        self.assertIs(loaded, backbone)
        self.assertIn("layers.0.weight", backbone.state_dict())
        self.assertFalse(any(name.startswith("model.") for name in backbone.state_dict()))

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
                "speech_to_speech.runtime.backbone.hf.AutoModel.from_config",
                return_value=backbone,
            ) as from_config,
            patch(
                "speech_to_speech.runtime.backbone.hf.AutoModel.from_pretrained"
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
            trust_remote_code=False,
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
                "speech_to_speech.runtime.backbone.hf.AutoModel.from_pretrained",
                return_value=backbone,
            ) as from_pretrained,
            patch(
                "speech_to_speech.runtime.backbone.hf.AutoModel.from_config"
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

    def test_gradient_checkpointing_uses_hf_backbone_hook_and_disables_cache(self):
        backbone = GradientCheckpointingBackbone()

        with patch(
            "speech_to_speech.runtime.backbone.hf.AutoModel.from_pretrained",
            return_value=backbone,
        ):
            runtime = Runtime(
                Config(
                    backbone="fake/backbone",
                    gradient_checkpointing=True,
                )
            )

            loaded = runtime.backbone

        self.assertEqual(backbone.gradient_checkpointing_calls, 1)
        self.assertEqual(
            backbone.gradient_checkpointing_kwargs,
            {"gradient_checkpointing_kwargs": {"use_reentrant": False}},
        )
        self.assertEqual(backbone.input_require_grads_calls, 1)
        self.assertFalse(backbone.config.use_cache)
        self.assertIs(loaded, backbone)

    def test_gradient_checkpointing_enters_external_adapter_wrapper(self):
        backbone = GradientCheckpointingBackbone()
        adapter = ExternalAdapter(backbone)

        with patch(
            "speech_to_speech.runtime.backbone.hf.AutoModel.from_pretrained",
            return_value=adapter,
        ):
            runtime = Runtime(
                Config(
                    backbone="fake/wrapped-backbone",
                    gradient_checkpointing=True,
                )
            )

            loaded = runtime.backbone

        self.assertEqual(backbone.gradient_checkpointing_calls, 1)
        self.assertEqual(
            backbone.gradient_checkpointing_kwargs,
            {"gradient_checkpointing_kwargs": {"use_reentrant": False}},
        )
        self.assertEqual(backbone.input_require_grads_calls, 1)
        self.assertFalse(adapter.config.use_cache)
        self.assertFalse(backbone.config.use_cache)
        self.assertIs(loaded, adapter)

    def test_gradient_checkpointing_rejects_legacy_hf_hook(self):
        class LegacyGradientCheckpointingBackbone(GradientCheckpointingBackbone):
            def gradient_checkpointing_enable(  # pyright: ignore[reportIncompatibleMethodOverride]
                self,
            ) -> None:
                self.gradient_checkpointing_calls += 1

        backbone = LegacyGradientCheckpointingBackbone()

        with patch(
            "speech_to_speech.runtime.backbone.hf.AutoModel.from_pretrained",
            return_value=backbone,
        ):
            runtime = Runtime(
                Config(
                    backbone="fake/legacy-backbone",
                    gradient_checkpointing=True,
                )
            )

            with self.assertRaisesRegex(TypeError, "gradient_checkpointing_kwargs"):
                _ = runtime.backbone

        self.assertEqual(backbone.gradient_checkpointing_calls, 0)

    def test_gradient_checkpointing_can_target_checkpointing_layers(self):
        backbone = LayerCheckpointingBackbone()

        with patch(
            "speech_to_speech.runtime.backbone.hf.AutoModel.from_pretrained",
            return_value=backbone,
        ):
            runtime = Runtime(
                Config(
                    backbone="fake/layer-backbone",
                    gradient_checkpointing=True,
                )
            )

            loaded = runtime.backbone

        self.assertTrue(backbone.block.gradient_checkpointing)
        self.assertTrue(callable(backbone.block._gradient_checkpointing_func))
        self.assertFalse(backbone.config.use_cache)
        self.assertIs(loaded, backbone)

if __name__ == "__main__":
    unittest.main()
