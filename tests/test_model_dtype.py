from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch
from anydataset.types import Modality
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from anytrain.lightning import apply_parameter_trainability
from anytrain.module.idspace import Layout
from torch import Tensor, nn

from speech_to_speech.model import (
    AdapterType,
    AudioInputAdapterConfig,
    AudioInputAdapterType,
    AudioOutputAdapterConfig,
    AudioOutputAdapterType,
    Config,
    Model,
)
from speech_to_speech.model.acoustic import DecoderConfig
from speech_to_speech.model.acoustic.flow import FlowModel
from speech_to_speech.model.acoustic.rvq import RVQModel
from speech_to_speech.runtime.audio_tokenizer import NativeAudioTokenizer
from speech_to_speech.training.parameter_policy import (
    PARAMETER_POLICY_SPECS,
    ParameterGroup,
    ParameterPolicyName,
    ParameterPolicyTrainability,
    parameter_group,
)


class ModelDtypeTest(unittest.TestCase):
    def test_speech_interface_uses_fp32_storage_with_bf16_backbone(self):
        model = _rvq_model()
        apply_parameter_trainability(
            model,
            ParameterPolicyTrainability(
                PARAMETER_POLICY_SPECS[ParameterPolicyName.SPEECH_INTERFACE]
            ),
        )

        backbone = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter_group(name) is ParameterGroup.BACKBONE
        ]
        speech = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter_group(name) is not ParameterGroup.BACKBONE
        ]

        self.assertTrue(backbone)
        self.assertTrue(speech)
        self.assertTrue(all(value.dtype is torch.bfloat16 for value in backbone))
        self.assertTrue(all(not value.requires_grad for value in backbone))
        self.assertTrue(all(value.dtype is torch.float32 for value in speech))
        self.assertTrue(any(value.requires_grad for value in speech))

    def test_rvq_boundaries_and_optimizer_state_remain_fp32(self):
        model = _rvq_model()
        apply_parameter_trainability(
            model,
            ParameterPolicyTrainability(
                PARAMETER_POLICY_SPECS[ParameterPolicyName.SPEECH_INTERFACE]
            ),
        )
        hidden = torch.randn(1, 2, 4, dtype=torch.bfloat16)
        positions = torch.tensor([[1, 2]])
        targets = torch.zeros(1, 2, 1, dtype=torch.long)

        semantic = model.semantic_audio_logits(hidden)
        acoustic = model.acoustic_logits(hidden, positions, targets)

        self.assertEqual(semantic.dtype, torch.float32)
        self.assertTrue(torch.isfinite(semantic).all())
        self.assertTrue(all(value.dtype is torch.float32 for value in acoustic))
        self.assertTrue(all(torch.isfinite(value).all() for value in acoustic))

        optimizer = torch.optim.AdamW(
            [value for value in model.parameters() if value.requires_grad],
            lr=2e-5,
        )
        loss = semantic.square().mean() + sum(
            value.square().mean() for value in acoustic
        )
        loss.backward()
        optimizer.step()

        states = [
            value
            for state in optimizer.state.values()
            for key, value in state.items()
            if key in {"exp_avg", "exp_avg_sq"}
        ]
        self.assertTrue(states)
        self.assertTrue(all(value.dtype is torch.float32 for value in states))

    def test_combined_vocabulary_logits_promote_fp32_audio_head(self):
        model = _rvq_model()
        hidden = torch.randn(1, 2, 4, dtype=torch.bfloat16)

        dense = model.token_logits(hidden)
        selected, _ = model.selected_logits(hidden, torch.tensor([0, 4, 8]))
        audio, _ = model.selected_logits(hidden, torch.tensor([4, 8]))

        self.assertEqual(dense.dtype, torch.float32)
        self.assertEqual(selected.dtype, torch.float32)
        self.assertEqual(audio.dtype, torch.float32)
        self.assertTrue(torch.isfinite(dense).all())
        self.assertTrue(torch.isfinite(selected).all())
        self.assertTrue(torch.isfinite(audio).all())

    def test_flow_inputs_follow_fp32_decoder_storage(self):
        model = _flow_model()
        hidden = torch.randn(1, 2, 4, dtype=torch.bfloat16)
        positions = torch.tensor([[1, 2]])
        codes = torch.zeros(1, 2, 1, dtype=torch.long)

        condition = model.target_frame_condition(hidden, positions)
        with patch.object(
            model,
            "_decoder_input",
            wraps=model._decoder_input,
        ) as decoder_input:
            target = model.acoustic_target_latent(codes)

        self.assertTrue(
            all(
                value.dtype is torch.float32
                for value in model.acoustic_decoder.parameters()
            )
        )
        self.assertEqual(condition.dtype, torch.float32)
        self.assertEqual(target.dtype, torch.float32)
        decoder_input.assert_not_called()

    def test_audio_embeddings_merge_into_bf16_backbone_input(self):
        model = _flow_model()
        token_labels = torch.tensor([[0, 4]])
        positions = torch.tensor([[0, 1]])

        condition = model.target_frame_label_condition(token_labels, positions)

        self.assertEqual(condition.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(condition).all())

    def test_model_bfloat16_runs_mlp_speech_interfaces(self) -> None:
        model = _interface_model(
            audio_input=AudioInputAdapterType.MLP,
            audio_output=AudioOutputAdapterType.MLP,
        ).bfloat16()
        model.eval()

        rows = model.tokens.audio_projection(model.tokens.audio_embedding.weight)
        embedded = model._input_embedding(
            torch.tensor([[0, 4, 5]]),
            torch.tensor([[1]]),
        )
        logits = model.token_logits(
            torch.randn(1, 3, 4, dtype=torch.bfloat16),
            Modality.AUDIO,
        )
        tied_model = _interface_model(
            audio_input=AudioInputAdapterType.MLP,
            audio_output=AudioOutputAdapterType.NONE,
        ).bfloat16()
        tied_logits = tied_model.semantic_audio_logits(
            torch.randn(1, 3, 4, dtype=torch.bfloat16)
        )

        self.assertEqual(rows.dtype, torch.bfloat16)
        self.assertEqual(embedded.dtype, torch.bfloat16)
        self.assertEqual(logits.dtype, torch.bfloat16)
        self.assertEqual(tied_logits.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(embedded).all())
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(tied_logits).all())

    def test_model_bfloat16_runs_transformer_speech_interfaces(self) -> None:
        model = _interface_model(
            audio_input=AudioInputAdapterType.TRANSFORMER,
            audio_output=AudioOutputAdapterType.TRANSFORMER,
        ).bfloat16()
        model.eval()

        embedded = model._input_embedding(
            torch.tensor([[0, 4, 5]]),
            torch.tensor([[-1, 1]]),
        )
        hidden = torch.randn(1, 3, 4, dtype=torch.bfloat16)
        projected, past = model.project_audio_hidden(
            hidden,
            attention_mask=torch.tensor([[False, True, True]]),
            use_cache=True,
        )
        continued, _ = model.project_audio_hidden(
            torch.randn(1, 1, 4, dtype=torch.bfloat16),
            attention_mask=torch.ones(1, 1, dtype=torch.bool),
            past_key_values=past,
            use_cache=True,
        )
        logits = model.semantic_audio_logits(projected)

        self.assertEqual(embedded.dtype, torch.bfloat16)
        self.assertEqual(projected.dtype, torch.bfloat16)
        self.assertEqual(continued.dtype, torch.bfloat16)
        self.assertEqual(logits.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(embedded).all())
        self.assertTrue(torch.isfinite(projected).all())
        self.assertTrue(torch.isfinite(continued).all())
        self.assertTrue(torch.isfinite(logits).all())


class _Backbone(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.input_embeddings = nn.Embedding(4, 4)
        self.output_embeddings = nn.Linear(4, 4, bias=False)
        self.to(dtype=dtype)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.input_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.output_embeddings


class _Codec:
    sample_rate = 24_000
    frame_rate = 50.0
    semantic_feature_dim = 4
    semantic_codebook_sizes = (3,)
    codebook_sizes = (3, 3)
    semantic_codebook = torch.randn(3, 4, dtype=torch.bfloat16)
    acoustic_feature_dim = 4
    acoustic_codebook_sizes = (3,)
    acoustic_layout = AcousticLayout.FRAME_ALIGNED
    acoustic_unit_length = None

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        del sample_rate
        return audio.new_zeros(1, 2, dtype=torch.long)

    def decode(self, codes: Tensor) -> Tensor:
        return codes.new_zeros(1, dtype=torch.float32)

    def tokenize(self, audio: Tensor, sample_rate: int) -> SemanticAcousticCodes:
        del sample_rate
        shape = (audio.size(0), 1, 1)
        semantic = torch.zeros(shape, dtype=torch.long, device=audio.device)
        acoustic = torch.zeros(shape, dtype=torch.long, device=audio.device)
        return SemanticAcousticCodes(semantic=semantic, acoustic=acoustic)

    def detokenize(self, codes: SemanticAcousticCodes) -> Tensor:
        return self.decode_features(
            codes.semantic,
            self.acoustic_codes_to_features(codes.acoustic),
        )

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        return torch.nn.functional.one_hot(
            acoustic_codes[..., 0],
            num_classes=self.acoustic_feature_dim,
        ).to(dtype=torch.bfloat16)

    def decode_features(
        self,
        semantic_codes: Tensor,
        acoustic_features: Tensor,
    ) -> Tensor:
        del semantic_codes
        return acoustic_features.new_zeros(1)


class _FlowRuntime:
    def sample(self, *args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(final=torch.zeros(1))


def _runtime(*, backbone_dtype: torch.dtype = torch.bfloat16) -> SimpleNamespace:
    return SimpleNamespace(
        backbone=_Backbone(backbone_dtype),
        codec=_Codec(),
        audio_tokenizer=NativeAudioTokenizer(vocab_size=3),
        layout=Layout(text=(0, 4), audio=(4, 10)),
        flow_matching=_FlowRuntime(),
        codec_audio_range=(4, 7),
    )


def _interface_model(
    *,
    audio_input: AudioInputAdapterType,
    audio_output: AudioOutputAdapterType,
) -> Model:
    return Model(
        Config(
            semantic_audio_adapter=AdapterType.MLP,
            audio_input_adapter=AudioInputAdapterConfig(
                type=audio_input,
                layers=1,
                heads=2,
                ffn_ratio=2,
            ),
            audio_output_adapter=AudioOutputAdapterConfig(
                type=audio_output,
                layers=1,
                heads=2,
                ffn_ratio=2,
            ),
        ),
        runtime=cast(Any, _runtime(backbone_dtype=torch.float32)),
    )


def _rvq_model() -> RVQModel:
    return RVQModel(
        Config(),
        runtime=cast(Any, _runtime()),
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
    )


def _flow_model() -> FlowModel:
    return FlowModel(
        Config(),
        runtime=cast(Any, _runtime()),
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
    )


if __name__ == "__main__":
    unittest.main()
