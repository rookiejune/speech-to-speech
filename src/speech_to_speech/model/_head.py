from __future__ import annotations

import torch
import torch.nn.functional as F
from anydataset.types import Modality
from torch import nn

from .protocol import TokenModelRuntime
from ..runtime.types import Backbone


class VocabularyHeadMixin:
    runtime: TokenModelRuntime
    backbone: Backbone
    semantic_audio_embedding: nn.Embedding
    semantic_audio_output_adapter: nn.Module

    def text_logits(
        self,
        hidden_state: torch.Tensor,
        local_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_start, text_end = self.runtime.layout.blocks["text"]
        output = self.backbone.get_output_embeddings()
        weight = output.weight[: text_end - text_start]
        bias = output.bias
        bias = None if bias is None else bias[: text_end - text_start]
        if local_ids is not None:
            weight = weight.index_select(0, local_ids)
            bias = None if bias is None else bias.index_select(0, local_ids)
        return F.linear(hidden_state, weight, bias)

    def semantic_audio_logits(
        self,
        hidden_state: torch.Tensor,
        local_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = self.semantic_audio_embedding.weight
        projected = self.semantic_audio_output_adapter(
            hidden_state.to(dtype=weight.dtype)
        )
        if local_ids is not None:
            weight = weight.index_select(0, local_ids)
        return F.linear(projected, weight)

    def token_logits(
        self,
        hidden_state: torch.Tensor,
        modality: Modality | None = None,
    ) -> torch.Tensor:
        if modality is Modality.TEXT:
            return self.text_logits(hidden_state)
        if modality is Modality.AUDIO:
            return self.semantic_audio_logits(hidden_state)
        if modality is not None:
            raise ValueError(f"unsupported token modality: {modality.value}")
        text = self.text_logits(hidden_state)
        audio = self.semantic_audio_logits(hidden_state)
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
        hidden_state: torch.Tensor,
        modality: Modality,
    ) -> torch.Tensor:
        if modality is Modality.TEXT:
            start, _ = self.runtime.layout.blocks[Modality.TEXT.value]
            logits = self.token_logits(hidden_state, modality)
            for token_id in (self.runtime.pad_token_id, self.runtime.bos_token_id):
                logits[..., token_id - start] = float("-inf")
            return logits
        if modality is Modality.AUDIO:
            start, _ = self.runtime.layout.blocks[Modality.AUDIO.value]
            logits = self.token_logits(hidden_state, modality)
            logits[..., self.runtime.boa_token_id - start] = float("-inf")
            return logits
        raise ValueError(f"unsupported generation modality: {modality.value}")

    def selected_logits(
        self,
        hidden_state: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        text_start, text_end = self.runtime.layout.blocks["text"]
        audio_start, audio_end = self.runtime.layout.blocks["audio"]
        text_mask = token_ids.ge(text_start) & token_ids.lt(text_end)
        audio_mask = token_ids.ge(audio_start) & token_ids.lt(audio_end)
        if not bool((text_mask | audio_mask).all()):
            raise ValueError("selected token ids contain an invalid vocabulary id.")
        if bool(text_mask.all()):
            return self.text_logits(hidden_state, token_ids - text_start)
        if bool(audio_mask.all()):
            return self.semantic_audio_logits(hidden_state, token_ids - audio_start)
        text = self.text_logits(hidden_state, token_ids[text_mask] - text_start)
        audio = self.semantic_audio_logits(hidden_state, token_ids[audio_mask] - audio_start)
        dtype = torch.promote_types(text.dtype, audio.dtype)
        logits = torch.empty(
            (*hidden_state.shape[:-1], token_ids.numel()),
            dtype=dtype,
            device=hidden_state.device,
        )
        logits[..., text_mask] = text.to(dtype=dtype)
        logits[..., audio_mask] = audio.to(dtype=dtype)
        return logits
