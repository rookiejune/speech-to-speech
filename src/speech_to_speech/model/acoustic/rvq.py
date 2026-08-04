from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial

import torch
from anytrain.loss import PackedCodebookLogits
from semantic_acoustic_generator.model import AcousticRVQDecoder
from semantic_acoustic_generator.runtime.artifact import AcousticGeneratorArtifact
from torch import Tensor

from ..generation import AcousticGeneration
from ...runtime.codec_contract import acoustic_codec
from ..checkpoint_contract import rvq_acoustic_contract
from ..base import Config
from ...runtime.protocol import TokenModelRuntime
from .config import DecoderConfig, decoder_options
from .base import code_features
from .base import AcousticModel
from .factory import rvq_generator


class RVQModel(AcousticModel):
    """Token model composition with a discrete RVQ acoustic decoder."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        runtime: TokenModelRuntime,
        decoder: DecoderConfig | Mapping[str, object] | None = None,
        codebook_embeddings: Sequence[Tensor] | None = None,
        initialization: AcousticGeneratorArtifact | None = None,
    ) -> None:
        codec = acoustic_codec(runtime.codec)
        options = decoder_options(decoder)
        generator = rvq_generator(options, initialization)
        super().__init__(
            config=config,
            runtime=runtime,
            condition_dim=(
                None if initialization is None else initialization.spec.condition_dim
            ),
        )
        sizes = codec.acoustic_codebook_sizes
        backbone_weight = self.backbone.get_input_embeddings().weight
        condition_dim = self.acoustic_condition.condition_dim
        self.acoustic_decoder = AcousticRVQDecoder(
            condition_dim,
            len(sizes),
            sizes,
            codebook_embeddings=codebook_embeddings,
            hidden_dim=options.hidden_dim,
            layers=options.layers,
            heads=options.heads,
            ffn_ratio=options.ffn_ratio,
        ).to(device=backbone_weight.device, dtype=torch.float32)
        if generator is not None:
            self.acoustic_decoder.load_state_dict(generator.core.state_dict())

    def _acoustic_checkpoint_components(self) -> Mapping[str, object]:
        return rvq_acoustic_contract(
            self.acoustic_condition,
            self.acoustic_decoder,
        )

    def _decoder_module(self) -> AcousticRVQDecoder:
        return self.acoustic_decoder

    def acoustic_logits(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
        target_acoustic_codes: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        condition = self._decoder_input(
            self.target_frame_condition(hidden_states, target_positions)
        )
        return self.acoustic_decoder(
            condition,
            target_acoustic_codes,
            mask=target_positions.ge(0),
        )

    def acoustic_packed_logits(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
        target_acoustic_codes: Tensor,
        *,
        mask: Tensor | None = None,
        validate: bool = True,
    ) -> PackedCodebookLogits:
        condition = self._decoder_input(
            self.target_frame_condition(hidden_states, target_positions)
        )
        return self.acoustic_decoder.forward_packed(
            condition,
            target_acoustic_codes,
            mask=target_positions.ge(0) if mask is None else mask,
            validate=validate,
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
            self._decoder_input(condition),
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
        features = code_features(self.acoustic_codec, self.backbone, codes)
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
        audio_input_positions: Tensor | None = None,
        do_sample: bool = True,
        use_cache: bool = True,
    ) -> AcousticGeneration:
        sample = partial(
            self.sample_acoustic_features,
            temperature=temperature,
            top_p=top_p,
        )
        return self._generate_audio_features(
            prompt_ids,
            sample=sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_attention_mask=prompt_attention_mask,
            audio_input_positions=audio_input_positions,
            do_sample=do_sample,
            use_cache=use_cache,
        )
