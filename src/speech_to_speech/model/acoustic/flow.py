from __future__ import annotations

from collections.abc import Mapping

import torch
from semantic_acoustic_codec.config import DecoderConfig as SacDecoderConfig
from semantic_acoustic_codec.model import FMFeatureGenerator
from semantic_acoustic_codec.model.dit import DiTDecoder
from semantic_acoustic_codec.runtime import AcousticGeneratorArtifact
from torch import Tensor, nn

from ...generation.types import AcousticGeneration
from ...runtime.types import AcousticCodec, acoustic_codec
from .._helper import register
from ..base import Config, Model
from ..protocol import FlowModelRuntime, FlowSamplingRuntime
from ._config import DecoderConfig, FlowRepaConfig, decoder_options
from ._codec import code_features
from .condition import HiddenConditionAdapter


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


class FlowModel(Model):
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
        super().__init__(config=config, runtime=runtime)
        options = decoder_options(decoder)
        backbone_weight = self.backbone.get_input_embeddings().weight
        condition_dim = self.backbone.config.hidden_size
        feature_mean = None
        feature_std = None
        if initialization is not None:
            generator = initialization.generator
            if not isinstance(generator, FMFeatureGenerator):
                raise TypeError("Flow initialization requires an FMFeatureGenerator artifact.")
            _validate_decoder_options(options, initialization)
            _validate_repa(repa, initialization)
            condition_dim = initialization.spec.condition_dim
            feature_mean = initialization.spec.feature_mean
            feature_std = initialization.spec.feature_std
        self.acoustic_condition = HiddenConditionAdapter(
            self.backbone.config.hidden_size,
            condition_dim,
        ).to(device=backbone_weight.device, dtype=torch.float32)
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
            feature_mean=feature_mean,
            feature_std=feature_std,
        ).to(device=backbone_weight.device, dtype=torch.float32)
        if initialization is not None:
            generator = initialization.generator
            if not isinstance(generator, FMFeatureGenerator):
                raise AssertionError("Flow initialization type changed after validation.")
            self.acoustic_decoder.load_state_dict(generator.core.state_dict())

    @property
    def acoustic_codec(self) -> AcousticCodec:
        return acoustic_codec(self.runtime.codec)

    @property
    def acoustic_decoder(self) -> DiTDecoder:
        return self.acoustic_flow.decoder

    def acoustic_target_latent(self, target_acoustic_codes: Tensor) -> Tensor:
        if target_acoustic_codes.dim() != 3:
            raise ValueError("target acoustic codes must have shape [B, F, N].")
        safe_codes = target_acoustic_codes.clamp_min(0)
        features = self._decoder_input(code_features(self.acoustic_codec, self.backbone, safe_codes))
        features = (
            features - self.acoustic_flow.feature_mean
        ) / self.acoustic_flow.feature_std
        return features.masked_fill(
            (target_acoustic_codes < 0).all(dim=-1)[..., None], 0
        )

    def target_frame_condition(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
    ) -> Tensor:
        return self.acoustic_condition(
            super().target_frame_condition(hidden_states, target_positions)
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
        generated, condition, frame_mask = self.generate_audio_condition(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_attention_mask=prompt_attention_mask,
            audio_input_positions=audio_input_positions,
            do_sample=do_sample,
            use_cache=use_cache,
        )
        condition = self.acoustic_condition(condition)
        return AcousticGeneration(
            sequence=generated,
            features=self.sample_acoustic_features(condition, mask=frame_mask),
            frame_counts=frame_mask.sum(dim=1),
        )

    def _decoder_input(self, value: Tensor) -> Tensor:
        parameter = next(self.acoustic_decoder.parameters())
        return value.to(dtype=parameter.dtype)


def _validate_decoder_options(
    options: DecoderConfig,
    initialization: AcousticGeneratorArtifact,
) -> None:
    decoder = initialization.spec.decoder
    expected = (options.hidden_dim, options.layers, options.heads, options.ffn_ratio)
    actual = (decoder.hidden_dim, decoder.layers, decoder.heads, decoder.ffn_ratio)
    if expected != actual:
        raise ValueError(
            "acoustic decoder config does not match initialization artifact: "
            f"{expected!r} != {actual!r}."
        )


def _validate_repa(
    repa: FlowRepaConfig | None,
    initialization: AcousticGeneratorArtifact,
) -> None:
    decoder = initialization.spec.decoder
    expected = (
        None if repa is None else repa["feature_dim"],
        None if repa is None else repa["student_layer"],
    )
    actual = (decoder.repa_feature_dim, decoder.repa_student_layer)
    if expected != actual:
        raise ValueError(
            "Flow REPA config does not match initialization artifact: "
            f"{expected!r} != {actual!r}."
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
