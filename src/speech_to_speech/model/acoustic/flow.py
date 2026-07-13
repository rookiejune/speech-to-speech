from __future__ import annotations

import torch
from torch import Tensor, nn

from ..base import SemanticModel
from ..protocol import FlowSamplingRuntime
from .dit import AcousticDiT

AcousticFlowDecoder = AcousticDiT


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
        repa_dim: int | None = None,
        repa_layer: int | None = None,
    ) -> None:
        super().__init__()
        self.decoder = AcousticFlowDecoder(
            condition_dim,
            latent_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            heads=heads,
            ffn_ratio=ffn_ratio,
            repa_dim=repa_dim,
            repa_layer=repa_layer,
        )
        self.runtime = runtime

    @torch.no_grad()
    def sample(
        self,
        condition: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        latent = torch.randn(
            (*condition.shape[:2], self.decoder.latent_dim),
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        return self.runtime.sample(
            self.decoder,
            latent,
            condition=condition,
        ).final


class SpeechToSpeechFlowModel(SemanticModel):
    """Speech-to-speech composition using a flow-matching decoder."""

    def __init__(self, config=None, runtime_snapshot=None) -> None:
        super().__init__(config=config, runtime_snapshot=runtime_snapshot)
        backbone_weight = self.backbone.get_input_embeddings().weight
        self.acoustic_flow = AcousticFlow(
            self.backbone.config.hidden_size,
            self.runtime.codec.acoustic_feature_dim,
            self.runtime.flow_matching,
            hidden_dim=self.config.acoustic_decoder_dim,
            layers=self.config.acoustic_decoder_layers,
            heads=self.config.acoustic_decoder_heads,
            ffn_ratio=self.config.acoustic_decoder_ffn_ratio,
            repa_dim=self.config.acoustic_repa_dim,
            repa_layer=self.config.acoustic_repa_layer,
        ).to(device=backbone_weight.device, dtype=backbone_weight.dtype)

    @property
    def acoustic_decoder(self) -> AcousticFlowDecoder:
        return self.acoustic_flow.decoder

    def acoustic_target_latent(self, acoustic_labels: Tensor) -> Tensor:
        if acoustic_labels.dim() != 3:
            raise ValueError("acoustic labels must have shape [B, F, N].")
        safe_labels = acoustic_labels.clamp_min(0)
        features = self._acoustic_features(safe_labels)
        return features.masked_fill((acoustic_labels < 0).all(dim=-1)[..., None], 0)

    @torch.no_grad()
    def sample_acoustic(self, condition: Tensor) -> Tensor:
        return self.acoustic_flow.sample(condition)

    @torch.no_grad()
    def generate_audio(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        acoustic_input_ids: Tensor | None = None,
        acoustic_input_positions: Tensor | None = None,
        acoustic_input_mask: Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> tuple[Tensor, Tensor]:
        generated, condition, spans = self._generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            acoustic_input_ids=acoustic_input_ids,
            acoustic_input_positions=acoustic_input_positions,
            acoustic_input_mask=acoustic_input_mask,
            stop_token_id=self.runtime.eoa_token_id,
            allowed_token_ids=self.runtime.audio_generation_allowed_ids,
            do_sample=do_sample,
            use_cache=use_cache,
            collect_audio_condition=True,
        )
        if condition is None or spans is None:
            raise ValueError(
                "semantic generation produced no codec-decodable audio tokens."
            )
        return generated, self.sample_acoustic(condition)
