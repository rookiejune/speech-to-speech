from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from semantic_acoustic_codec.model import AcousticRVQDecoder
from torch import Tensor

from ...generation.types import AcousticGeneration
from ..base import Config, TokenModel
from ..protocol import TokenModelRuntime
from ._config import DecoderConfig, decoder_options


class RVQModel(TokenModel):
    """Token model composition with a discrete RVQ acoustic decoder."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        runtime: TokenModelRuntime,
        decoder: DecoderConfig | Mapping[str, object] | None = None,
        codebook_embeddings: Sequence[Tensor] | None = None,
    ) -> None:
        super().__init__(config=config, runtime=runtime)
        options = decoder_options(decoder)
        sizes = self.runtime.codec.acoustic_codebook_sizes
        backbone_weight = self.backbone.get_input_embeddings().weight
        self.acoustic_decoder = AcousticRVQDecoder(
            self.backbone.config.hidden_size,
            len(sizes),
            sizes,
            codebook_embeddings=codebook_embeddings,
            hidden_dim=options.hidden_dim,
            layers=options.layers,
            heads=options.heads,
            ffn_ratio=options.ffn_ratio,
        ).to(device=backbone_weight.device, dtype=backbone_weight.dtype)

    def acoustic_logits(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
        target_acoustic_codes: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        condition = self.target_frame_condition(hidden_states, target_positions)
        return self.acoustic_decoder(
            condition,
            target_acoustic_codes,
            mask=target_positions.ge(0),
        )

    @torch.no_grad()
    def sample_acoustic_codes(
        self,
        condition: Tensor,
        *,
        mask: Tensor | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return self.acoustic_decoder.generate(
            condition,
            mask=mask,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )

    @torch.no_grad()
    def sample_acoustic_features(
        self,
        condition: Tensor,
        *,
        mask: Tensor | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        codes = self.sample_acoustic_codes(
            condition,
            mask=mask,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        features = self.acoustic_code_features(codes)
        if mask is not None:
            features = features.masked_fill(~mask[..., None], 0)
        return features

    @torch.no_grad()
    def generate_audio_features(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        prompt_attention_mask: Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> AcousticGeneration:
        generated, condition, frame_mask = self.generate_audio_condition(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_attention_mask=prompt_attention_mask,
            do_sample=do_sample,
            use_cache=use_cache,
        )
        features = self.sample_acoustic_features(
            condition,
            mask=frame_mask,
            temperature=temperature,
            top_p=top_p,
        )
        return AcousticGeneration(
            sequence=generated,
            features=features,
            frame_counts=frame_mask.sum(dim=1),
        )
