from __future__ import annotations

from dataclasses import dataclass

import torch
from anytrain.idspace import IdSpaceEmbedding, Modality, ModalityBlock
from torch import Tensor, nn
from transformers.cache_utils import Cache

from ...types.datamodule import GenerationBatch
from ...types.model import (
    AcousticCondition,
    AcousticConditionGeneration,
    AcousticFeatureGenerator,
    AudioBoundary,
    SemanticBPE,
    SemanticGeneration,
    SpecialToken,
    WaveformCodec,
    WaveformGeneration,
)
from ..token_space import AudioLMHead


@dataclass(frozen=True)
class Generator:
    qwen3: nn.Module
    embed_tokens: IdSpaceEmbedding
    output_adapter: nn.Module
    lm_head: AudioLMHead

    @torch.no_grad()
    def acoustic_condition(
        self,
        batch: GenerationBatch,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        return_token_ids: bool = False,
    ) -> AcousticConditionGeneration:
        _validate_generation_args(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        _validate_generation_batch(batch)

        block = self.embed_tokens.space.modality_block(Modality.AUDIO)
        pad_id = self.embed_tokens.space.special_token_id(SpecialToken.PAD)
        eoa_id = self.embed_tokens.space.special_token_id(AudioBoundary.EOA)

        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        pending_condition = torch.zeros_like(finished)
        hidden_steps: list[Tensor] = []
        mask_steps: list[Tensor] = []
        token_steps: list[Tensor] = []
        last_hidden, past_key_values = self._prefill(input_ids, attention_mask)

        for _ in range(max_new_tokens):
            _append_condition_step(
                hidden_steps,
                mask_steps,
                last_hidden,
                pending_condition,
            )

            logits = self.lm_head(self.output_adapter(last_hidden))
            next_ids = self.lm_head.to_global_ids(
                _sample_next_head_ids(logits, temperature=temperature, top_p=top_p)
            )
            if not isinstance(next_ids, Tensor):
                raise TypeError("sampled ids must be returned as a Tensor.")
            active = ~finished
            next_ids = torch.where(
                active,
                next_ids.to(device=input_ids.device),
                input_ids.new_full((), pad_id),
            )
            token_steps.append(next_ids)

            is_eoa = next_ids.eq(eoa_id)
            pending_condition = active & _is_audio_bpe_id(next_ids, block)
            finished = finished | (active & is_eoa)
            input_ids = torch.cat((input_ids, next_ids.unsqueeze(1)), dim=1)
            attention_mask = torch.cat(
                (attention_mask, active.to(dtype=torch.long).unsqueeze(1)),
                dim=1,
            )
            if bool(finished.all()):
                break
            last_hidden, past_key_values = self._next_hidden(
                input_ids,
                next_ids,
                attention_mask,
                past_key_values,
            )

        if bool(pending_condition.any()):
            _append_condition_step(
                hidden_steps,
                mask_steps,
                last_hidden,
                pending_condition,
            )

        hidden_states, mask = _stack_condition_steps(
            hidden_steps,
            mask_steps,
            batch_size=batch.input_ids.size(0),
            hidden_size=last_hidden.size(-1),
            device=batch.input_ids.device,
            dtype=last_hidden.dtype,
        )
        token_ids = torch.stack(token_steps, dim=1) if return_token_ids and token_steps else None
        return AcousticConditionGeneration(
            hidden_states=hidden_states,
            mask=mask,
            token_ids=token_ids,
        )

    @torch.no_grad()
    def semantic(
        self,
        batch: GenerationBatch,
        *,
        bpe: SemanticBPE,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> SemanticGeneration:
        generation = self.acoustic_condition(
            batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            return_token_ids=True,
        )
        if generation.token_ids is None:
            raise RuntimeError("semantic generation must return token ids.")
        semantic_ids, semantic_mask, _, _ = _semantic_frames_from_generation(
            generation.token_ids,
            generation.hidden_states,
            generation.mask,
            bpe=bpe,
            block=self.embed_tokens.space.modality_block(Modality.AUDIO),
        )
        return SemanticGeneration(
            semantic_ids=semantic_ids,
            semantic_mask=semantic_mask,
            token_ids=generation.token_ids,
        )

    def waveform(
        self,
        batch: GenerationBatch,
        *,
        bpe: SemanticBPE,
        codec: WaveformCodec,
        acoustic_generator: AcousticFeatureGenerator | None,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> WaveformGeneration:
        if acoustic_generator is None:
            raise RuntimeError(
                "full waveform generation requires an acoustic feature generator."
            )

        generation = self.acoustic_condition(
            batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            return_token_ids=True,
        )
        if generation.token_ids is None:
            raise RuntimeError("semantic generation must return token ids.")
        semantic_ids, semantic_mask, hidden_states, chunk_lengths = _semantic_frames_from_generation(
            generation.token_ids,
            generation.hidden_states,
            generation.mask,
            bpe=bpe,
            block=self.embed_tokens.space.modality_block(Modality.AUDIO),
        )
        condition = AcousticCondition(
            hidden_states=hidden_states,
            semantic_ids=semantic_ids,
            mask=semantic_mask,
            chunk_lengths=chunk_lengths,
        )
        acoustic_features = acoustic_generator(condition)
        acoustic_features = _validate_acoustic_features(acoustic_features, condition.mask)
        audio = _decode_waveform(codec, condition.semantic_ids, acoustic_features)
        return WaveformGeneration(
            audio=audio,
            audio_mask=torch.ones(
                audio.shape[:2],
                dtype=torch.bool,
                device=audio.device,
            ),
            semantic_ids=condition.semantic_ids,
            semantic_mask=condition.mask,
            acoustic_features=acoustic_features,
            condition_hidden_states=condition.hidden_states,
            token_ids=generation.token_ids,
        )

    def _prefill(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Cache | None]:
        outputs = self.qwen3(
            attention_mask=attention_mask,
            inputs_embeds=self.embed_tokens(input_ids),
            use_cache=True,
            cache_position=torch.arange(input_ids.size(1), device=input_ids.device),
        )
        return (
            _last_non_padding_hidden(outputs.last_hidden_state, attention_mask),
            getattr(outputs, "past_key_values", None),
        )

    def _next_hidden(
        self,
        input_ids: Tensor,
        next_ids: Tensor,
        attention_mask: Tensor,
        past_key_values: Cache | None,
    ) -> tuple[Tensor, Cache | None]:
        if past_key_values is None:
            return self._prefill(input_ids, attention_mask)

        outputs = self.qwen3(
            attention_mask=attention_mask,
            inputs_embeds=self.embed_tokens(next_ids.unsqueeze(1)),
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=torch.tensor(
                [attention_mask.size(1) - 1],
                dtype=torch.long,
                device=attention_mask.device,
            ),
        )
        hidden_states = outputs.last_hidden_state
        return hidden_states[:, -1], getattr(outputs, "past_key_values", past_key_values)


def _last_non_padding_hidden(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    positions = attention_mask.sum(dim=1).sub(1).clamp_min(0).to(dtype=torch.long)
    return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), positions]


def _validate_generation_args(
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an integer.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if isinstance(temperature, bool) or not isinstance(temperature, int | float):
        raise TypeError("temperature must be a float.")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    if isinstance(top_p, bool) or not isinstance(top_p, int | float):
        raise TypeError("top_p must be a float.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")


def _validate_generation_batch(batch: GenerationBatch) -> None:
    if batch.input_ids.dim() != 2:
        raise ValueError("generation input_ids must have shape (batch, sequence).")
    if batch.attention_mask.shape != batch.input_ids.shape:
        raise ValueError("generation attention_mask must match input_ids shape.")
    if batch.input_ids.device != batch.attention_mask.device:
        raise ValueError("generation input_ids and attention_mask must be on the same device.")
    if (
        batch.input_ids.dtype == torch.bool
        or torch.is_floating_point(batch.input_ids)
        or torch.is_complex(batch.input_ids)
    ):
        raise TypeError("generation input_ids must contain integer ids.")
    if bool(batch.attention_mask.sum(dim=1).eq(0).any()):
        raise ValueError("generation attention_mask must keep at least one token per row.")
    if batch.attention_mask.dtype == torch.bool:
        return
    if torch.is_floating_point(batch.attention_mask) or torch.is_complex(batch.attention_mask):
        raise TypeError("generation attention_mask must contain integer mask values.")


def _append_condition_step(
    hidden_steps: list[Tensor],
    mask_steps: list[Tensor],
    hidden: Tensor,
    mask: Tensor,
) -> None:
    if not bool(mask.any()):
        return
    hidden_steps.append(hidden)
    mask_steps.append(mask)


def _stack_condition_steps(
    hidden_steps: list[Tensor],
    mask_steps: list[Tensor],
    *,
    batch_size: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if not hidden_steps:
        return (
            torch.empty((batch_size, 0, hidden_size), dtype=dtype, device=device),
            torch.empty((batch_size, 0), dtype=torch.bool, device=device),
        )
    return torch.stack(hidden_steps, dim=1), torch.stack(mask_steps, dim=1)


def _is_audio_bpe_id(token_ids: Tensor, block: ModalityBlock) -> Tensor:
    return token_ids.ge(block.start) & token_ids.lt(block.end)


def _semantic_frames_from_generation(
    token_ids: Tensor,
    hidden_states: Tensor,
    token_mask: Tensor,
    *,
    bpe: SemanticBPE,
    block: ModalityBlock,
) -> tuple[Tensor, Tensor, Tensor, tuple[tuple[int, ...], ...]]:
    semantic_rows: list[Tensor] = []
    hidden_rows: list[Tensor] = []
    chunk_lengths: list[tuple[int, ...]] = []
    for row_index, row in enumerate(token_ids.detach().cpu()):
        token_ids_row: list[int] = []
        hidden_parts: list[Tensor] = []
        row_chunk_lengths: list[int] = []
        hidden_index = 0
        for token_id in row.tolist():
            token_id = int(token_id)
            if not block.start <= token_id < block.end:
                continue
            if hidden_index >= hidden_states.size(1):
                raise ValueError("generated BPE hidden states are shorter than token ids.")
            if not bool(token_mask[row_index, hidden_index]):
                raise ValueError("generated BPE hidden mask is missing an active token.")
            local_id = token_id - block.start
            expanded = _single_codebook_ids(bpe.expand_ids([local_id]))
            token_ids_row.extend(expanded)
            row_chunk_lengths.append(len(expanded))
            hidden = hidden_states[row_index, hidden_index]
            hidden_parts.append(hidden.unsqueeze(0).expand(len(expanded), -1))
            hidden_index += 1
        if hidden_index != int(token_mask[row_index].sum().item()):
            raise ValueError("generated BPE hidden mask does not match token ids.")
        semantic_rows.append(torch.tensor(token_ids_row, dtype=torch.long))
        if hidden_parts:
            hidden_rows.append(torch.cat(hidden_parts, dim=0))
        else:
            hidden_rows.append(hidden_states.new_empty((0, hidden_states.size(-1))))
        chunk_lengths.append(tuple(row_chunk_lengths))

    max_length = max((row.numel() for row in semantic_rows), default=0)
    semantic_ids = token_ids.new_zeros((token_ids.size(0), max_length))
    mask = torch.zeros(
        (token_ids.size(0), max_length),
        dtype=torch.bool,
        device=token_ids.device,
    )
    expanded_hidden = hidden_states.new_zeros(
        (token_ids.size(0), max_length, hidden_states.size(-1))
    )
    for index, row in enumerate(semantic_rows):
        row = row.to(device=token_ids.device)
        semantic_ids[index, : row.numel()] = row
        mask[index, : row.numel()] = True
        expanded_hidden[index, : row.numel()] = hidden_rows[index].to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    return semantic_ids, mask, expanded_hidden, tuple(chunk_lengths)


def _validate_acoustic_features(acoustic_features: Tensor, mask: Tensor) -> Tensor:
    if acoustic_features.dim() != 3:
        raise ValueError(
            "acoustic generator must return acoustic features with shape [batch, time, dim]."
        )
    if acoustic_features.shape[:2] != mask.shape:
        raise ValueError("acoustic features must align with semantic mask on batch and time.")
    if not torch.is_floating_point(acoustic_features) or torch.is_complex(acoustic_features):
        raise TypeError("acoustic features must be floating point tensors.")
    return acoustic_features.to(device=mask.device)


def _decode_waveform(codec: WaveformCodec, semantic_ids: Tensor, acoustic_features: Tensor) -> Tensor:
    audio = codec.decode_features(semantic_ids, acoustic_features)
    if not isinstance(audio, Tensor):
        raise TypeError("LongCat codec decode_features() must return a Tensor.")
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
    if audio.dim() != 3:
        raise ValueError("decoded waveform must have shape [batch, channels, time].")
    return audio.detach().float()


def _single_codebook_ids(frames: object) -> list[int]:
    ids: list[int] = []
    if not isinstance(frames, list | tuple):
        raise TypeError("LongCat BPE expand_ids() must return a sequence of frames.")
    for frame in frames:
        if isinstance(frame, int) or not isinstance(frame, list | tuple):
            raise TypeError("LongCat BPE expand_ids() must return frame sequences.")
        if len(frame) != 1:
            raise ValueError("LongCat semantic BPE must use exactly one codebook.")
        ids.append(int(frame[0]))
    return ids


def _sample_next_head_ids(
    logits: Tensor,
    *,
    temperature: float,
    top_p: float,
) -> Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1)
    scaled = logits.float() / temperature
    if top_p < 1:
        scaled = _top_p_logits(scaled, top_p=top_p)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)


def _top_p_logits(logits: Tensor, *, top_p: float) -> Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, torch.finfo(logits.dtype).min)
    filtered = torch.full_like(logits, torch.finfo(logits.dtype).min)
    return filtered.scatter(dim=-1, index=sorted_indices, src=sorted_logits)
