from __future__ import annotations

from collections.abc import Mapping

import torch
from semantic_acoustic_codec.config import DecoderConfig as SacDecoderConfig
from semantic_acoustic_codec.model import FMFeatureGenerator
from semantic_acoustic_codec.model.dit import DiTDecoder
from semantic_acoustic_codec.runtime.artifact import AcousticGeneratorArtifact
from torch import Tensor, nn

from ...generation.types import AcousticGeneration
from ...runtime.types import acoustic_codec
from .._contract_state import flow_acoustic_contract
from .._helper import register
from ..base import Config
from ..protocol import FlowModelRuntime, FlowSamplingRuntime
from ._config import DecoderConfig, FlowRepaConfig, decoder_options
from ._codec import code_features
from .base import AcousticModel
from .initialization import flow_generator


class AcousticFlow(nn.Module):
    """S2S sampling wrapper around SAC ``FMFeatureGenerator``."""

    feature_mean: Tensor
    feature_std: Tensor

    def __init__(
        self,
        condition_dim: int,
        latent_dim: int,
        runtime: FlowSamplingRuntime,
        *,
        hidden_dim: int | None = None,
        layers: int = 8,
        heads: int = 8,
        ffn_ratio: int = 4,
        repa_feature_dim: int | None = None,
        repa_student_layer: int | None = None,
        feature_mean: tuple[float, ...] | None = None,
        feature_std: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.generator = FMFeatureGenerator(
            condition_dim,
            latent_dim,
            SacDecoderConfig(
                hidden_dim=hidden_dim,
                layers=layers,
                heads=heads,
                ffn_ratio=ffn_ratio,
                repa_feature_dim=repa_feature_dim,
                repa_student_layer=repa_student_layer,
                repa_loss_weight=0.0,
            ),
        )
        self.runtime = runtime
        register(
            self,
            "feature_mean",
            _feature_stat(latent_dim, feature_mean, fill=0.0),
        )
        register(
            self,
            "feature_std",
            _feature_stat(latent_dim, feature_std, fill=1.0),
        )

    @property
    def decoder(self) -> DiTDecoder:
        return self.generator.core

    @torch.no_grad()
    def sample(
        self,
        condition: Tensor,
        *,
        mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        if mask is not None:
            if mask.shape != condition.shape[:2]:
                raise ValueError("acoustic frame mask must align with condition.")
            if mask.dtype != torch.bool:
                raise TypeError("acoustic frame mask must be boolean.")
        parameter = next(self.decoder.parameters())
        condition = condition.to(dtype=parameter.dtype)
        latent = torch.randn(
            (*condition.shape[:2], self.decoder.decoder.latent_dim),
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        output = self.runtime.sample(
            self.decoder,
            latent,
            condition=condition,
            mask=mask,
        ).final
        output = output * self.feature_std + self.feature_mean
        if mask is not None:
            output = output.masked_fill(~mask[..., None], 0)
        return output


class FlowModel(AcousticModel):
    """Token model composition with a flow-matching acoustic decoder."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        runtime: FlowModelRuntime,
        decoder: DecoderConfig | Mapping[str, object] | None = None,
        repa: FlowRepaConfig | None = None,
        initialization: AcousticGeneratorArtifact | None = None,
    ) -> None:
        codec = acoustic_codec(runtime.codec)
        options = decoder_options(decoder)
        generator = flow_generator(options, repa, initialization)
        super().__init__(
            config=config,
            runtime=runtime,
            condition_dim=(
                None if initialization is None else initialization.spec.condition_dim
            ),
        )
        backbone_weight = self.backbone.get_input_embeddings().weight
        condition_dim = self.acoustic_condition.condition_dim
        self.acoustic_flow = AcousticFlow(
            condition_dim,
            codec.acoustic_feature_dim,
            runtime.flow_matching,
            hidden_dim=options.hidden_dim,
            layers=options.layers,
            heads=options.heads,
            ffn_ratio=options.ffn_ratio,
            repa_feature_dim=None if repa is None else repa["feature_dim"],
            repa_student_layer=None if repa is None else repa["student_layer"],
            feature_mean=(
                None if initialization is None else initialization.spec.feature_mean
            ),
            feature_std=(
                None if initialization is None else initialization.spec.feature_std
            ),
        ).to(device=backbone_weight.device, dtype=torch.float32)
        if generator is not None:
            self.acoustic_decoder.load_state_dict(generator.core.state_dict())

    @property
    def acoustic_decoder(self) -> DiTDecoder:
        return self.acoustic_flow.decoder

    def _acoustic_checkpoint_components(self) -> Mapping[str, object]:
        return flow_acoustic_contract(
            self.acoustic_condition,
            self.acoustic_flow,
        )

    def _decoder_module(self) -> nn.Module:
        return self.acoustic_decoder

    def acoustic_target_latent(self, target_acoustic_codes: Tensor) -> Tensor:
        if target_acoustic_codes.dim() != 3:
            raise ValueError("target acoustic codes must have shape [B, F, N].")
        safe_codes = target_acoustic_codes.clamp_min(0)
        decoder_parameter = next(self.acoustic_decoder.parameters())
        features = code_features(
            self.acoustic_codec,
            self.backbone,
            safe_codes,
            like=decoder_parameter,
        )
        features = (
            features - self.acoustic_flow.feature_mean
        ) / self.acoustic_flow.feature_std
        return features.masked_fill(
            (target_acoustic_codes < 0).all(dim=-1)[..., None], 0
        )

    @torch.no_grad()
    def sample_acoustic_features(
        self,
        condition: Tensor,
        *,
        mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return self.acoustic_flow.sample(
            condition,
            mask=mask,
            generator=generator,
        )

    @torch.no_grad()
    def generate_audio_features(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        prompt_attention_mask: Tensor | None = None,
        audio_input_positions: Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> AcousticGeneration:
        return self._generate_audio_features(
            prompt_ids,
            sample=self.sample_acoustic_features,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_attention_mask=prompt_attention_mask,
            audio_input_positions=audio_input_positions,
            do_sample=do_sample,
            use_cache=use_cache,
        )


def _feature_stat(
    feature_dim: int,
    value: tuple[float, ...] | None,
    *,
    fill: float,
) -> Tensor:
    if value is None:
        return torch.full((1, 1, feature_dim), fill, dtype=torch.float32)
    if len(value) != feature_dim:
        raise ValueError("acoustic feature normalization must match feature dimension.")
    result = torch.tensor(value, dtype=torch.float32).view(1, 1, -1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("acoustic feature normalization must be finite.")
    if fill == 1.0 and not bool((result > 0).all()):
        raise ValueError("acoustic feature standard deviation must be positive.")
    return result
