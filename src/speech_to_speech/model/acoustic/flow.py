from __future__ import annotations

import torch
from torch import Tensor

from ..base import SemanticModel
from .dit import AcousticDiT

AcousticFlowDecoder = AcousticDiT


class SpeechToSpeechFlowModel(SemanticModel):
    """Speech-to-speech composition using a flow-matching decoder."""

    def __init__(self, config=None, runtime_snapshot=None) -> None:
        super().__init__(config=config, runtime_snapshot=runtime_snapshot)
        backbone_weight = self.backbone.get_input_embeddings().weight
        self.acoustic_decoder = AcousticFlowDecoder(
            self.backbone.config.hidden_size,
            self.runtime.codec.acoustic_feature_dim,
            hidden_dim=self.config.acoustic_decoder_dim,
            layers=self.config.acoustic_decoder_layers,
            heads=self.config.acoustic_decoder_heads,
            ffn_ratio=self.config.acoustic_decoder_ffn_ratio,
            repa_dim=self.config.acoustic_repa_dim,
            repa_layer=self.config.acoustic_repa_layer,
        ).to(device=backbone_weight.device, dtype=backbone_weight.dtype)

    def acoustic_target_latent(self, acoustic_labels: Tensor) -> Tensor:
        if acoustic_labels.dim() != 3:
            raise ValueError("acoustic labels must have shape [B, F, N].")
        safe_labels = acoustic_labels.clamp_min(0)
        features = self._acoustic_features(safe_labels)
        return features.masked_fill((acoustic_labels < 0).all(dim=-1)[..., None], 0)

    @torch.no_grad()
    def sample_acoustic(self, condition: Tensor) -> Tensor:
        latent = torch.randn(
            (*condition.shape[:2], self.acoustic_decoder.latent_dim),
            device=condition.device,
            dtype=condition.dtype,
        )
        return self.runtime.flow_matching.sample(
            self.acoustic_decoder,
            latent,
            condition=condition,
        ).final

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
