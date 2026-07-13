from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor, nn
from transformers import Qwen3Config, Qwen3Model

from .._sampling import top_p_filter
from ..base import SemanticModel


class AcousticRVQDecoder(nn.Module):
    """Frame-parallel, codebook-autoregressive acoustic code predictor.

    ``codebook_embeddings`` uses the codec's local codebook order and contains
    one ``[size_q, embedding_dim]`` tensor per codebook. The
    embeddings are copied into trainable model embeddings; the codec remains
    unchanged.
    """

    def __init__(
        self,
        condition_dim: int,
        codebooks: int,
        codebook_size: int | Sequence[int],
        *,
        codebook_embeddings: Sequence[Tensor] | None = None,
        hidden_dim: int | None = None,
        layers: int = 8,
        heads: int = 8,
        ffn_ratio: int = 4,
    ) -> None:
        super().__init__()
        if condition_dim <= 0 or codebooks <= 0:
            raise ValueError("condition_dim and codebooks must be positive.")
        if layers <= 0 or heads <= 0 or ffn_ratio <= 0:
            raise ValueError("decoder depth, heads, and FFN ratio must be positive.")
        hidden_dim = condition_dim if hidden_dim is None else hidden_dim
        if hidden_dim <= 0:
            raise ValueError("decoder hidden dimension must be positive.")
        attention_heads = _heads(hidden_dim, heads)
        sizes = (
            (codebook_size,) * codebooks
            if isinstance(codebook_size, int)
            else tuple(codebook_size)
        )
        if len(sizes) != codebooks or any(size <= 0 for size in sizes):
            raise ValueError(
                "codebook_size must provide one positive size per codebook."
            )
        if codebook_embeddings is not None:
            if len(codebook_embeddings) != codebooks:
                raise ValueError(
                    "codebook_embeddings must provide one tensor per codebook."
                )
            if any(not torch.is_floating_point(value) for value in codebook_embeddings):
                raise TypeError("codebook_embeddings must be floating point.")
            if any(value.dim() != 2 for value in codebook_embeddings):
                raise ValueError(
                    "each codebook embedding must have shape [size_q, dim]."
                )
            if any(
                value.size(0) != size for value, size in zip(codebook_embeddings, sizes)
            ):
                raise ValueError("codebook embeddings must match codebook sizes.")
            embedding_dim = codebook_embeddings[0].size(-1)
            if any(value.size(-1) != embedding_dim for value in codebook_embeddings):
                raise ValueError(
                    "all codebook embeddings must have the same dimension."
                )
        else:
            embedding_dim = hidden_dim

        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.codebooks = codebooks
        self.codebook_sizes = sizes
        self.embedding_dim = embedding_dim
        self.codebook_embeddings = nn.ModuleList(
            nn.Embedding(size, embedding_dim) for size in sizes
        )
        if codebook_embeddings is None:
            for module in self.codebook_embeddings:
                embedding = cast(nn.Embedding, cast(object, module))
                nn.init.normal_(embedding.weight, std=embedding_dim**-0.5)
        else:
            with torch.no_grad():
                for index, module in enumerate(self.codebook_embeddings):
                    embedding = cast(nn.Embedding, cast(object, module))
                    embedding.weight.copy_(codebook_embeddings[index])

        self.embedding_projections = nn.ModuleList(
            nn.Identity()
            if embedding_dim == hidden_dim
            else nn.Linear(embedding_dim, hidden_dim)
            for _ in range(codebooks)
        )
        self.condition = (
            nn.Identity()
            if condition_dim == hidden_dim
            else nn.Linear(condition_dim, hidden_dim)
        )
        self.codebook_bos = nn.Parameter(torch.zeros(codebooks, hidden_dim))
        config = Qwen3Config(
            vocab_size=1,
            hidden_size=hidden_dim,
            intermediate_size=hidden_dim * ffn_ratio,
            num_hidden_layers=layers,
            num_attention_heads=attention_heads,
            num_key_value_heads=attention_heads,
            head_dim=hidden_dim // attention_heads,
            use_cache=True,
        )
        self.decoder = Qwen3Model(config)
        self.decoder.embed_tokens.requires_grad_(False)
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, size) for size in sizes)

    def _validate_condition(self, condition: Tensor) -> None:
        if condition.dim() != 3 or condition.size(-1) != self.condition_dim:
            raise ValueError("condition must have shape [batch, frame, condition_dim].")

    def _embedding(self, codebook: int, ids: Tensor) -> Tensor:
        if ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
            raise TypeError("acoustic code labels must contain integer ids.")
        if bool((ids < 0).any()) or bool((ids >= self.codebook_sizes[codebook]).any()):
            raise ValueError("acoustic code label is outside the codec codebook.")
        embedding = cast(
            nn.Embedding,
            cast(object, self.codebook_embeddings[codebook]),
        )
        projection = cast(
            nn.Module,
            cast(object, self.embedding_projections[codebook]),
        )
        value = embedding(ids)
        return projection(value)

    def forward(
        self,
        condition: Tensor,
        acoustic_labels: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """Return one teacher-forced ``[B, F, K_q]`` tensor per codebook."""
        self._validate_condition(condition)
        if acoustic_labels is not None:
            if acoustic_labels.shape != (
                condition.size(0),
                condition.size(1),
                self.codebooks,
            ):
                raise ValueError("acoustic_labels must have shape [B, F, codebooks].")
            if bool((acoustic_labels < 0).any()):
                raise ValueError("acoustic_labels cannot contain padding values.")

        condition_hidden = self.condition(condition)
        inputs = [condition_hidden + self.codebook_bos[0]]
        for codebook in range(1, self.codebooks):
            if acoustic_labels is None:
                previous = torch.zeros(
                    condition.shape[:2], dtype=torch.long, device=condition.device
                )
            else:
                previous = acoustic_labels[..., codebook - 1].clamp_min(0)
            inputs.append(
                condition_hidden
                + self.codebook_bos[codebook]
                + self._embedding(codebook - 1, previous)
            )
        decoder_input = torch.stack(inputs, dim=2).flatten(0, 1)
        hidden = self.decoder(
            inputs_embeds=decoder_input,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state.unflatten(0, condition.shape[:2])
        return tuple(
            cast(nn.Linear, cast(object, self.heads[codebook]))(
                hidden[..., codebook, :]
            )
            for codebook in range(self.codebooks)
        )

    @torch.no_grad()
    def generate(
        self,
        condition: Tensor,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> Tensor:
        """Sample codebooks autoregressively while keeping frames parallel."""
        self._validate_condition(condition)
        if temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError(
                "temperature must be positive and top_p must be in (0, 1]."
            )

        condition_hidden = self.condition(condition)
        output: list[Tensor] = []
        past_key_values = None
        for codebook in range(self.codebooks):
            decoder_input = condition_hidden + self.codebook_bos[codebook]
            if output:
                decoder_input = decoder_input + self._embedding(
                    codebook - 1, output[-1]
                )
            state_output = self.decoder(
                inputs_embeds=decoder_input.flatten(0, 1)[:, None],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = state_output.past_key_values
            if past_key_values is None:
                raise RuntimeError("RVQ decoder did not return a generation cache.")
            state = state_output.last_hidden_state[:, -1].unflatten(
                0, condition.shape[:2]
            )
            head = cast(nn.Linear, cast(object, self.heads[codebook]))
            logits = head(state) / temperature
            if top_p < 1.0:
                logits = top_p_filter(logits, top_p)
            value = torch.distributions.Categorical(logits=logits).sample()
            output.append(value)
        return torch.stack(output, dim=-1)


class SpeechToSpeechRVQModel(SemanticModel):
    """Semantic model composition using a discrete RVQ acoustic decoder."""

    def __init__(
        self,
        config=None,
        runtime_snapshot=None,
        *,
        codebook_embeddings: Sequence[Tensor] | None = None,
    ) -> None:
        super().__init__(config=config, runtime_snapshot=runtime_snapshot)
        sizes = self.runtime.codec.acoustic_codebook_sizes
        backbone_weight = self.backbone.get_input_embeddings().weight
        self.acoustic_decoder = AcousticRVQDecoder(
            self.backbone.config.hidden_size,
            len(sizes),
            sizes,
            codebook_embeddings=codebook_embeddings,
            hidden_dim=self.config.acoustic_decoder_dim,
            layers=self.config.acoustic_decoder_layers,
            heads=self.config.acoustic_decoder_heads,
            ffn_ratio=self.config.acoustic_decoder_ffn_ratio,
        ).to(device=backbone_weight.device, dtype=backbone_weight.dtype)

    def acoustic_logits(
        self,
        hidden_states: Tensor,
        target_positions: Tensor,
        acoustic_labels: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        condition = self.target_frame_condition(hidden_states, target_positions)
        return self.acoustic_decoder(condition, acoustic_labels)

    @torch.no_grad()
    def sample_acoustic(self, condition: Tensor, **kwargs: float) -> Tensor:
        return self.acoustic_decoder.generate(condition, **kwargs)

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
        prompt_attention_mask: Tensor | None = None,
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
            prompt_attention_mask=prompt_attention_mask,
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
        codes = self.sample_acoustic(
            condition,
            temperature=temperature,
            top_p=top_p,
        )
        return generated, self._acoustic_features(codes)


def _heads(hidden_dim: int, requested: int) -> int:
    for heads in range(min(hidden_dim, requested), 0, -1):
        if hidden_dim % heads == 0:
            return heads
    raise RuntimeError("a positive hidden dimension must have an attention head divisor")
