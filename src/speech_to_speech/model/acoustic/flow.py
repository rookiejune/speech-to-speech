from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ...generation.types import AcousticGeneration
from ..base import Config, TokenModel
from ..protocol import FlowModelRuntime, FlowSamplingRuntime
from ._config import DecoderConfig, FlowRepaConfig, decoder_options
from .dit import AcousticDiT


class AcousticFlow(nn.Module):
    """Acoustic flow decoder and sampling shared by training compositions."""

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
    ) -> None:
        super().__init__()
        self.decoder = AcousticDiT(
            condition_dim,
            latent_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            heads=heads,
            ffn_ratio=ffn_ratio,
            repa_feature_dim=repa_feature_dim,
            repa_student_layer=repa_student_layer,
        )
        self.runtime = runtime

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
        latent = torch.randn(
            (*condition.shape[:2], self.decoder.latent_dim),
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
        if mask is not None:
            output = output.masked_fill(~mask[..., None], 0)
        return output


class FlowModel(TokenModel):
    """Token model composition with a flow-matching acoustic decoder."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        runtime: FlowModelRuntime,
        decoder: DecoderConfig | Mapping[str, object] | None = None,
        repa: FlowRepaConfig | None = None,
    ) -> None:
        super().__init__(config=config, runtime=runtime)
        options = decoder_options(decoder)
        backbone_weight = self.backbone.get_input_embeddings().weight
        self.acoustic_flow = AcousticFlow(
            self.backbone.config.hidden_size,
            self.runtime.codec.acoustic_feature_dim,
            runtime.flow_matching,
            hidden_dim=options.hidden_dim,
            layers=options.layers,
            heads=options.heads,
            ffn_ratio=options.ffn_ratio,
            repa_feature_dim=None if repa is None else repa["feature_dim"],
            repa_student_layer=None if repa is None else repa["student_layer"],
        ).to(device=backbone_weight.device, dtype=backbone_weight.dtype)

    @property
    def acoustic_decoder(self) -> AcousticDiT:
        return self.acoustic_flow.decoder

    def acoustic_target_latent(self, target_acoustic_codes: Tensor) -> Tensor:
        if target_acoustic_codes.dim() != 3:
            raise ValueError("target acoustic codes must have shape [B, F, N].")
        safe_codes = target_acoustic_codes.clamp_min(0)
        features = self.acoustic_code_features(safe_codes)
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
        acoustic_prompt_codes: Tensor | None = None,
        acoustic_prompt_positions: Tensor | None = None,
        acoustic_prompt_mask: Tensor | None = None,
        prompt_attention_mask: Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> AcousticGeneration:
        generated, condition, frame_mask = self.generate_audio_condition(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            acoustic_prompt_codes=acoustic_prompt_codes,
            acoustic_prompt_positions=acoustic_prompt_positions,
            acoustic_prompt_mask=acoustic_prompt_mask,
            prompt_attention_mask=prompt_attention_mask,
            do_sample=do_sample,
            use_cache=use_cache,
        )
        return AcousticGeneration(
            sequence=generated,
            features=self.sample_acoustic_features(condition, mask=frame_mask),
            frame_counts=frame_mask.sum(dim=1),
        )
