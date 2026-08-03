from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from anytrain.loss import PackedCodebookLogits
from semantic_acoustic_codec.model import AcousticRVQDecoder, RVQCodeGenerator
from semantic_acoustic_codec.runtime.artifact import AcousticGeneratorArtifact
from torch import Tensor

from ...generation.types import AcousticGeneration
from ...runtime.types import AcousticCodec, acoustic_codec
from ..base import Config, Model
from ..protocol import TokenModelRuntime
from ._config import DecoderConfig, decoder_options
from ._codec import code_features
from .condition import HiddenConditionAdapter


class RVQModel(Model):
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
        super().__init__(config=config, runtime=runtime)
        options = decoder_options(decoder)
        sizes = codec.acoustic_codebook_sizes
        backbone_weight = self.backbone.get_input_embeddings().weight
        condition_dim = self.backbone.config.hidden_size
        if initialization is not None:
            generator = initialization.generator
            if not isinstance(generator, RVQCodeGenerator):
                raise TypeError("RVQ initialization requires an RVQCodeGenerator artifact.")
            if not isinstance(generator.core, AcousticRVQDecoder):
                raise ValueError(
                    "joint S2S RVQ initialization currently requires the codebook_ar predictor."
                )
            _validate_decoder_options(options, initialization)
            condition_dim = initialization.spec.condition_dim
        self.acoustic_condition = HiddenConditionAdapter(
            self.backbone.config.hidden_size,
            condition_dim,
        ).to(device=backbone_weight.device, dtype=torch.float32)
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
        if initialization is not None:
            generator = initialization.generator
            if not isinstance(generator, RVQCodeGenerator) or not isinstance(
                generator.core, AcousticRVQDecoder
            ):
                raise AssertionError("RVQ initialization type changed after validation.")
            self.acoustic_decoder.load_state_dict(generator.core.state_dict())

    @property
    def acoustic_codec(self) -> AcousticCodec:
        return acoustic_codec(self.runtime.codec)

    def target_frame_condition(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
    ) -> Tensor:
        return self.acoustic_condition(
            super().target_frame_condition(hidden_states, target_positions)
        )

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
