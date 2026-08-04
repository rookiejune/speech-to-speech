from __future__ import annotations

import torch
import torch.nn.functional as F
from anydataset.types import Modality
from anytrain.module.idspace import Layout
from torch import Tensor, nn

from ._helper import CastOutput
from .audio_output import AudioOutputAdapter, AudioOutputAdapterType
from .embedding.audio import SemanticAudioEmbedding, require_semantic_audio_embedding
from .embedding.fsq import FsqEmbedding, FsqNeighbors
from .generation import TokenKind


class TokenInterface(nn.Module):
    """Own the trainable audio token interface around a text backbone.

    The backbone remains the sole owner of its text embedding. Text lookup and
    tied text logits therefore receive that embedding explicitly at call time.
    """

    audio_embedding: SemanticAudioEmbedding

    def __init__(
        self,
        layout: Layout,
        *,
        audio_embedding: SemanticAudioEmbedding,
        audio_projection: CastOutput,
        audio_head: AudioOutputAdapter,
    ) -> None:
        super().__init__()
        if frozenset(layout.block_names) != {
            Modality.TEXT.value,
            Modality.AUDIO.value,
        }:
            raise ValueError("token layout must contain exactly text and audio blocks.")
        audio = require_semantic_audio_embedding(
            audio_embedding,
            "semantic audio embedding",
        )
        if not isinstance(audio, nn.Module):
            raise TypeError("semantic audio embedding must also be an nn.Module.")
        _validate_audio_block(layout, audio.num_embeddings)

        self.layout = layout
        self.audio_embedding = audio
        self.audio_projection = audio_projection
        self.audio_head = audio_head

    @property
    def semantic_audio_embedding(self) -> SemanticAudioEmbedding:
        return require_semantic_audio_embedding(
            self.audio_embedding,
            "semantic audio embedding",
        )

    @property
    def vocab_size(self) -> int:
        return self.layout.vocab_size

    def embed(
        self,
        input_ids: Tensor,
        text_embedding: nn.Embedding,
        *,
        input_modalities: frozenset[Modality] | None = None,
        validate: bool = True,
        audio_override_mask: Tensor | None = None,
    ) -> Tensor:
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty.")
        _validate_text_embedding(self.layout, text_embedding)
        selected = _checked_modalities(input_modalities)
        if validate:
            routed = self.selected_modalities(input_ids)
            if selected is not None and selected != routed:
                raise ValueError("input_modalities does not match the routed input ids.")
            selected = routed
        elif selected is None:
            raise ValueError("input_modalities is required when validate=False.")

        override = _checked_override_mask(audio_override_mask, input_ids)
        output = text_embedding.weight.new_zeros(
            *input_ids.shape,
            text_embedding.embedding_dim,
        )
        text_start, text_end = self.layout.blocks[Modality.TEXT.value]
        if Modality.TEXT in selected:
            text_mask = input_ids.ge(text_start) & input_ids.lt(text_end)
            output[text_mask] = text_embedding(input_ids[text_mask] - text_start)

        audio_start, audio_end = self.layout.blocks[Modality.AUDIO.value]
        if Modality.AUDIO in selected:
            audio_mask = input_ids.ge(audio_start) & input_ids.lt(audio_end)
            if override is not None:
                audio_mask &= ~override
            rows = self.audio_rows(input_ids[audio_mask] - audio_start)
            output[audio_mask] = self.audio_projection(rows)
        return output

    def selected_modalities(self, input_ids: Tensor) -> frozenset[Modality]:
        """Validate global IDs and return the exact routed modalities in one sync."""
        if input_ids.numel() == 0:
            raise ValueError("input_ids must not be empty.")
        covered = torch.zeros_like(input_ids, dtype=torch.bool)
        hits: list[Tensor] = []
        for name in (Modality.TEXT.value, Modality.AUDIO.value):
            start, end = self.layout.blocks[name]
            mask = input_ids.ge(start) & input_ids.lt(end)
            hits.append(mask.any())
            covered |= mask

        first_uncovered = (~covered).reshape(-1).to(dtype=torch.int64).argmax()
        first_uncovered_id = input_ids.reshape(-1).gather(
            0,
            first_uncovered.reshape(1),
        )
        summary = torch.cat(
            (
                torch.stack((*hits, covered.all())).to(dtype=torch.int64),
                first_uncovered_id.to(dtype=torch.int64),
            )
        ).detach().cpu().tolist()
        text_hit, audio_hit, all_covered, bad_id = summary
        if not all_covered:
            raise ValueError(f"input_ids contains id outside space: {bad_id}.")
        return frozenset(
            modality
            for modality, hit in (
                (Modality.TEXT, text_hit),
                (Modality.AUDIO, audio_hit),
            )
            if hit
        )

    def audio_rows(self, local_ids: Tensor) -> Tensor:
        embedding = self.semantic_audio_embedding
        if isinstance(embedding, FsqEmbedding):
            return embedding.rows(local_ids)
        return embedding(local_ids)

    def audio_neighbor_targets(self, local_ids: Tensor) -> FsqNeighbors | None:
        embedding = self.semantic_audio_embedding
        if not isinstance(embedding, FsqEmbedding):
            return None
        return embedding.neighbors(local_ids)

    def text_logits(
        self,
        text_embedding: nn.Embedding,
        hidden_state: Tensor,
        local_ids: Tensor | None = None,
    ) -> Tensor:
        text_rows = _text_rows(self.layout, text_embedding)
        weight = (
            text_rows
            if local_ids is None
            else text_rows.index_select(0, local_ids)
        )
        return F.linear(hidden_state.to(dtype=weight.dtype), weight)

    def semantic_audio_logits(
        self,
        hidden_state: Tensor,
        local_ids: Tensor | None = None,
    ) -> Tensor:
        embedding = self.semantic_audio_embedding
        if self.audio_head.config.type is not AudioOutputAdapterType.NONE:
            return _raw_audio_logits(embedding, hidden_state, local_ids)
        if (
            local_ids is None
            and isinstance(embedding, FsqEmbedding)
            and isinstance(self.audio_projection.module, nn.Identity)
        ):
            return embedding.logits(hidden_state)

        rows = (
            embedding.weight
            if local_ids is None
            else self.audio_rows(local_ids)
        )
        weight = self.audio_projection(
            rows,
            cast_output=False,
        )
        return F.linear(hidden_state.to(dtype=weight.dtype), weight)

    def project_audio_hidden(
        self,
        hidden_state: Tensor,
        *,
        attention_mask: Tensor | None = None,
        selection_mask: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, object | None]:
        return self.audio_head(
            hidden_state,
            attention_mask=attention_mask,
            selection_mask=selection_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def token_logits(
        self,
        text_embedding: nn.Embedding,
        hidden_state: Tensor,
        modality: Modality | None = None,
        *,
        attention_mask: Tensor | None = None,
        audio_hidden_state: Tensor | None = None,
    ) -> Tensor:
        if modality is Modality.TEXT:
            return self.text_logits(text_embedding, hidden_state)
        if modality is Modality.AUDIO:
            adapted = (
                audio_hidden_state
                if audio_hidden_state is not None
                else self.project_audio_hidden(
                    hidden_state,
                    attention_mask=attention_mask,
                )[0]
            )
            return self.semantic_audio_logits(adapted)
        if modality is not None:
            raise ValueError(f"unsupported token modality: {modality.value}")

        adapted = (
            audio_hidden_state
            if audio_hidden_state is not None
            else self.project_audio_hidden(
                hidden_state,
                attention_mask=attention_mask,
            )[0]
        )
        text = self.text_logits(text_embedding, hidden_state)
        audio = self.semantic_audio_logits(adapted)
        dtype = torch.promote_types(text.dtype, audio.dtype)
        logits = torch.full(
            (*hidden_state.shape[:-1], self.layout.vocab_size),
            float("-inf"),
            dtype=dtype,
            device=hidden_state.device,
        )
        text_start, text_end = self.layout.blocks[Modality.TEXT.value]
        audio_start, audio_end = self.layout.blocks[Modality.AUDIO.value]
        logits[..., text_start:text_end] = text.to(dtype=dtype)
        logits[..., audio_start:audio_end] = audio.to(dtype=dtype)
        return logits

    def modality_logits(
        self,
        text_embedding: nn.Embedding,
        hidden_state: Tensor,
        modality: Modality,
        *,
        blocked_token_ids: tuple[int, int],
        attention_mask: Tensor | None = None,
        audio_hidden_state: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, object | None]:
        if modality is Modality.TEXT:
            start, _ = self.layout.blocks[Modality.TEXT.value]
            logits = self.text_logits(text_embedding, hidden_state)
            for token_id in blocked_token_ids:
                logits[..., token_id - start] = float("-inf")
            return logits, None
        if modality is Modality.AUDIO:
            start, _ = self.layout.blocks[Modality.AUDIO.value]
            audio_past = None
            if audio_hidden_state is None:
                audio_hidden_state, audio_past = self.project_audio_hidden(
                    hidden_state,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
            logits = self.semantic_audio_logits(audio_hidden_state)
            for token_id in blocked_token_ids:
                logits[..., token_id - start] = float("-inf")
            return logits, audio_past
        raise ValueError(f"unsupported generation modality: {modality.value}")

    def selected_logits(
        self,
        text_embedding: nn.Embedding,
        hidden_state: Tensor,
        token_ids: Tensor,
        *,
        token_kind: TokenKind | None = None,
        attention_mask: Tensor | None = None,
        audio_hidden_state: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
        validate: bool = True,
    ) -> tuple[Tensor, object | None]:
        text_start, text_end = self.layout.blocks[Modality.TEXT.value]
        audio_start, audio_end = self.layout.blocks[Modality.AUDIO.value]
        actual_kind, text_mask, audio_mask = (
            _selected_token_routing(self.layout, token_ids, token_kind)
            if validate
            else (
                _trusted_selected_token_kind(token_ids, token_kind),
                None,
                None,
            )
        )
        if actual_kind == Modality.TEXT.value:
            return (
                self.text_logits(
                    text_embedding,
                    hidden_state,
                    token_ids - text_start,
                ),
                None,
            )
        if actual_kind == Modality.AUDIO.value:
            return self._selected_audio_logits(
                hidden_state,
                token_ids - audio_start,
                attention_mask=attention_mask,
                audio_hidden_state=audio_hidden_state,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

        if text_mask is None or audio_mask is None:
            text_mask, audio_mask = _selected_token_masks(self.layout, token_ids)
        text = self.text_logits(
            text_embedding,
            hidden_state,
            token_ids[text_mask] - text_start,
        )
        audio, audio_past = self._selected_audio_logits(
            hidden_state,
            token_ids[audio_mask] - audio_start,
            attention_mask=attention_mask,
            audio_hidden_state=audio_hidden_state,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        dtype = torch.promote_types(text.dtype, audio.dtype)
        logits = torch.full(
            (*hidden_state.shape[:-1], token_ids.numel()),
            float("-inf"),
            dtype=dtype,
            device=hidden_state.device,
        )
        logits[..., text_mask] = text.to(dtype=dtype)
        logits[..., audio_mask] = audio.to(dtype=dtype)
        return logits, audio_past

    def _selected_audio_logits(
        self,
        hidden_state: Tensor,
        local_ids: Tensor,
        *,
        attention_mask: Tensor | None,
        audio_hidden_state: Tensor | None,
        past_key_values: object | None,
        use_cache: bool,
    ) -> tuple[Tensor, object | None]:
        audio_past = None
        if audio_hidden_state is None:
            audio_hidden_state, audio_past = self.project_audio_hidden(
                hidden_state,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        return self.semantic_audio_logits(audio_hidden_state, local_ids), audio_past

    def select_audio_head_cache(
        self,
        past_key_values: object | None,
        indices: Tensor,
    ) -> object | None:
        return self.audio_head.batch_select_past(past_key_values, indices)


def _raw_audio_logits(
    embedding: SemanticAudioEmbedding,
    hidden_state: Tensor,
    local_ids: Tensor | None,
) -> Tensor:
    if isinstance(embedding, FsqEmbedding):
        return embedding.logits(hidden_state, local_ids)
    weight = embedding.weight
    if local_ids is not None:
        weight = weight.index_select(0, local_ids)
    return F.linear(hidden_state.to(dtype=weight.dtype), weight)


def _validate_audio_block(layout: Layout, rows: int) -> None:
    start, end = layout.blocks[Modality.AUDIO.value]
    if rows != end - start:
        raise ValueError("audio embedding rows must match its layout block size.")


def _validate_text_embedding(layout: Layout, embedding: nn.Embedding) -> None:
    start, end = layout.blocks[Modality.TEXT.value]
    if start < 0 or embedding.num_embeddings < end - start:
        raise ValueError("text embedding does not cover its layout block.")


def _text_rows(layout: Layout, embedding: nn.Embedding) -> Tensor:
    _validate_text_embedding(layout, embedding)
    start, end = layout.blocks[Modality.TEXT.value]
    return embedding.weight[: end - start]


def _selected_token_routing(
    layout: Layout,
    token_ids: Tensor,
    hinted_kind: TokenKind | None,
) -> tuple[TokenKind, Tensor, Tensor]:
    _validate_selected_token_ids(token_ids)
    text_mask, audio_mask = _selected_token_masks(layout, token_ids)
    text_only, audio_only, all_covered = (
        torch.stack(
            (
                text_mask.all(),
                audio_mask.all(),
                (text_mask | audio_mask).all(),
            )
        )
        .detach()
        .cpu()
        .tolist()
    )
    if not all_covered:
        raise ValueError("selected token ids contain an invalid vocabulary id.")

    actual_kind: TokenKind
    if text_only:
        actual_kind = "text"
    elif audio_only:
        actual_kind = "audio"
    else:
        actual_kind = "mixed"
    if hinted_kind not in {None, "text", "audio", "mixed"}:
        raise ValueError(f"unsupported selected token kind: {hinted_kind!r}")
    if hinted_kind is not None and hinted_kind != actual_kind:
        raise ValueError(
            f"selected token kind {hinted_kind!r} does not match "
            f"the actual kind {actual_kind!r}."
        )
    return actual_kind, text_mask, audio_mask


def _trusted_selected_token_kind(
    token_ids: Tensor,
    hinted_kind: TokenKind | None,
) -> TokenKind:
    _validate_selected_token_ids(token_ids)
    if hinted_kind is None:
        raise ValueError(
            "selected token kind is required when token id validation is disabled."
        )
    if hinted_kind not in {"text", "audio", "mixed"}:
        raise ValueError(f"unsupported selected token kind: {hinted_kind!r}")
    return hinted_kind


def _validate_selected_token_ids(token_ids: Tensor) -> None:
    if token_ids.dim() != 1 or token_ids.numel() == 0:
        raise ValueError("selected token ids must be a non-empty 1D tensor.")
    if token_ids.dtype not in {torch.int32, torch.int64}:
        raise TypeError("selected token ids must use an int32 or int64 dtype.")


def _selected_token_masks(layout: Layout, token_ids: Tensor) -> tuple[Tensor, Tensor]:
    text_start, text_end = layout.blocks[Modality.TEXT.value]
    audio_start, audio_end = layout.blocks[Modality.AUDIO.value]
    text_mask = token_ids.ge(text_start) & token_ids.lt(text_end)
    audio_mask = token_ids.ge(audio_start) & token_ids.lt(audio_end)
    return text_mask, audio_mask


def _checked_modalities(
    value: frozenset[Modality] | None,
) -> frozenset[Modality] | None:
    if value is None:
        return None
    if not isinstance(value, frozenset) or any(
        not isinstance(modality, Modality) for modality in value
    ):
        raise TypeError("input_modalities must be a frozenset of Modality values.")
    unknown = value - {Modality.TEXT, Modality.AUDIO}
    if unknown:
        names = sorted(modality.value for modality in unknown)
        raise ValueError(f"input_modalities contains unsupported values: {names!r}.")
    if not value:
        raise ValueError("input_modalities must not be empty.")
    return value


def _checked_override_mask(value: Tensor | None, input_ids: Tensor) -> Tensor | None:
    if value is None:
        return None
    if value.dtype != torch.bool:
        raise TypeError("audio_override_mask must be boolean.")
    if value.shape != input_ids.shape:
        raise ValueError("audio_override_mask must align with input_ids.")
    if value.device != input_ids.device:
        raise ValueError("audio_override_mask must be on the input device.")
    return value


__all__ = ["TokenInterface"]
