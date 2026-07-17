from __future__ import annotations

import torch
import torch.nn.functional as F
from anydataset.types import Modality
from torch import nn

from .protocol import TokenModelRuntime


class VocabularyHeadMixin:
    runtime: TokenModelRuntime
    semantic_audio_embedding: nn.Embedding
    semantic_audio_output_adapter: nn.Module

    def text_logits(
        self,
        hidden_state: torch.Tensor,
        local_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_start, text_end = self.runtime.layout.blocks["text"]
        output = self.runtime.backbone.get_output_embeddings()
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
        projected = self.semantic_audio_output_adapter(hidden_state)
        weight = self.semantic_audio_embedding.weight
        if local_ids is not None:
            weight = weight.index_select(0, local_ids)
        return F.linear(projected, weight)

    def token_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        logits = hidden_state.new_full(
            (*hidden_state.shape[:-1], self.runtime.layout.vocab_size),
            float("-inf"),
        )
        text_start, text_end = self.runtime.layout.blocks["text"]
        audio_start, audio_end = self.runtime.layout.blocks["audio"]
        logits[..., text_start:text_end] = self.text_logits(hidden_state)
        logits[..., audio_start:audio_end] = self.semantic_audio_logits(hidden_state)
        return logits

    def modality_logits(
        self,
        hidden_state: torch.Tensor,
        modality: Modality,
    ) -> torch.Tensor:
        if modality is Modality.TEXT:
            start, _ = self.runtime.layout.blocks[Modality.TEXT.value]
            logits = self.text_logits(hidden_state)
            for token_id in (self.runtime.pad_token_id, self.runtime.bos_token_id):
                logits[..., token_id - start] = float("-inf")
            return logits
        if modality is Modality.AUDIO:
            start, _ = self.runtime.layout.blocks[Modality.AUDIO.value]
            logits = self.semantic_audio_logits(hidden_state)
            logits[..., self.runtime.boa_token_id - start] = float("-inf")
            return logits
        raise ValueError(f"unsupported generation modality: {modality.value}")

    def selected_logits(
        self,
        hidden_state: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = hidden_state.new_empty(*hidden_state.shape[:-1], token_ids.numel())
        text_start, text_end = self.runtime.layout.blocks["text"]
        audio_start, audio_end = self.runtime.layout.blocks["audio"]
        text_mask = token_ids.ge(text_start) & token_ids.lt(text_end)
        audio_mask = token_ids.ge(audio_start) & token_ids.lt(audio_end)
        if not bool((text_mask | audio_mask).all()):
            raise ValueError("selected token ids contain an invalid vocabulary id.")
        if bool(text_mask.any()):
            logits[..., text_mask] = self.text_logits(
                hidden_state, token_ids[text_mask] - text_start
            )
        if bool(audio_mask.any()):
            logits[..., audio_mask] = self.semantic_audio_logits(
                hidden_state, token_ids[audio_mask] - audio_start
            )
        return logits
