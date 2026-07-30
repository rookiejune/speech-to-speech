from __future__ import annotations

import torch
import torch.nn.functional as F
from anydataset.types import Modality
from anytrain.module.idspace import Embedding
from torch import Tensor

from .audio_output import AudioOutputAdapter
from ._helper import require_embedding
from .protocol import TokenModelRuntime


class VocabularyHeadMixin:
    runtime: TokenModelRuntime
    token_embedding: Embedding
    audio_output_adapter: AudioOutputAdapter

    def text_logits(
        self,
        hidden_state: Tensor,
        local_ids: Tensor | None = None,
    ) -> Tensor:
        weight = require_embedding(
            self.token_embedding.embeddings["text"],
            "text token embedding",
        ).weight
        if local_ids is not None:
            weight = weight.index_select(0, local_ids)
        return F.linear(hidden_state.to(dtype=weight.dtype), weight)

    def semantic_audio_logits(
        self,
        hidden_state: Tensor,
        local_ids: Tensor | None = None,
    ) -> Tensor:
        """Compute audio logits from already-adapted hidden states."""
        weight = require_embedding(
            self.token_embedding.embeddings["audio"],
            "semantic audio embedding",
        ).weight
        if local_ids is not None:
            weight = weight.index_select(0, local_ids)
        return F.linear(hidden_state.to(dtype=weight.dtype), weight)

    def project_audio_hidden(
        self,
        hidden_state: Tensor,
        *,
        attention_mask: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, object | None]:
        return self.audio_output_adapter(
            hidden_state,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    def token_logits(
        self,
        hidden_state: Tensor,
        modality: Modality | None = None,
        *,
        attention_mask: Tensor | None = None,
        audio_hidden_state: Tensor | None = None,
    ) -> Tensor:
        if modality is Modality.TEXT:
            return self.text_logits(hidden_state)
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
        text = self.text_logits(hidden_state)
        audio = self.semantic_audio_logits(adapted)
        dtype = torch.promote_types(text.dtype, audio.dtype)
        logits = torch.full(
            (*hidden_state.shape[:-1], self.runtime.layout.vocab_size),
            float("-inf"),
            dtype=dtype,
            device=hidden_state.device,
        )
        text_start, text_end = self.runtime.layout.blocks["text"]
        audio_start, audio_end = self.runtime.layout.blocks["audio"]
        logits[..., text_start:text_end] = text.to(dtype=dtype)
        logits[..., audio_start:audio_end] = audio.to(dtype=dtype)
        return logits

    def modality_logits(
        self,
        hidden_state: Tensor,
        modality: Modality,
        *,
        attention_mask: Tensor | None = None,
        audio_hidden_state: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, object | None]:
        audio_past = None
        if modality is Modality.TEXT:
            start, _ = self.runtime.layout.blocks[Modality.TEXT.value]
            logits = self.text_logits(hidden_state)
            for token_id in (self.runtime.pad_token_id, self.runtime.bos_token_id):
                logits[..., token_id - start] = float("-inf")
            return logits, None
        if modality is Modality.AUDIO:
            start, _ = self.runtime.layout.blocks[Modality.AUDIO.value]
            if audio_hidden_state is None:
                audio_hidden_state, audio_past = self.project_audio_hidden(
                    hidden_state,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
            logits = self.semantic_audio_logits(audio_hidden_state)
            for token_id in (
                self.runtime.boa_token_id,
                self.runtime.mask_token_id,
            ):
                logits[..., token_id - start] = float("-inf")
            return logits, audio_past
        raise ValueError(f"unsupported generation modality: {modality.value}")

    def selected_logits(
        self,
        hidden_state: Tensor,
        token_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        audio_hidden_state: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, object | None]:
        text_start, text_end = self.runtime.layout.blocks["text"]
        audio_start, audio_end = self.runtime.layout.blocks["audio"]
        text_mask = token_ids.ge(text_start) & token_ids.lt(text_end)
        audio_mask = token_ids.ge(audio_start) & token_ids.lt(audio_end)
        if not bool((text_mask | audio_mask).all()):
            raise ValueError("selected token ids contain an invalid vocabulary id.")
        audio_past = None
        if bool(audio_mask.any()) and audio_hidden_state is None:
            audio_hidden_state, audio_past = self.project_audio_hidden(
                hidden_state,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        if bool(text_mask.all()):
            return self.text_logits(hidden_state, token_ids - text_start), None
        if bool(audio_mask.all()):
            assert audio_hidden_state is not None
            return (
                self.semantic_audio_logits(
                    audio_hidden_state,
                    token_ids - audio_start,
                ),
                audio_past,
            )
        assert audio_hidden_state is not None
        text = self.text_logits(hidden_state, token_ids[text_mask] - text_start)
        audio = self.semantic_audio_logits(
            audio_hidden_state,
            token_ids[audio_mask] - audio_start,
        )
        dtype = torch.promote_types(text.dtype, audio.dtype)
        logits = torch.empty(
            (*hidden_state.shape[:-1], token_ids.numel()),
            dtype=dtype,
            device=hidden_state.device,
        )
        logits[..., text_mask] = text.to(dtype=dtype)
        logits[..., audio_mask] = audio.to(dtype=dtype)
        return logits, audio_past
