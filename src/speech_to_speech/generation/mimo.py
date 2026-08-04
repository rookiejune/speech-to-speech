"""Synchronous aligned text/audio autoregressive generation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict

from typing_extensions import NotRequired

import torch
from torch import Tensor

from .._tensor import is_signed_integer_dtype
from ..mimo import MimoGenerationStep


class MimoGenerationModel(Protocol):
    def mimo_generation_step(
        self,
        text_input_ids: Tensor,
        audio_input_ids: Tensor,
        *,
        attention_mask: Tensor,
        past_key_values: object | None,
        use_cache: bool,
        audio_features: Tensor | None = None,
        audio_feature_mask: Tensor | None = None,
    ) -> MimoGenerationStep: ...


class _MimoGenerationStepKwargs(TypedDict):
    text_input_ids: Tensor
    audio_input_ids: Tensor
    attention_mask: Tensor
    past_key_values: object | None
    use_cache: bool
    audio_features: NotRequired[Tensor | None]
    audio_feature_mask: NotRequired[Tensor | None]


@dataclass(frozen=True)
class MimoGenerationOptions:
    max_new_tokens: int
    text_eos_token_id: int
    audio_eos_token_id: int
    text_blank_token_id: int
    audio_blank_token_id: int
    audio_bos_token_id: int
    audio_delay_tokens: int = 0
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True
    use_cache: bool = True
    text_allowed_token_ids: Sequence[int] | Tensor | None = None
    audio_allowed_token_ids: Sequence[int] | Tensor | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive integer.")
        for name in (
            "text_eos_token_id",
            "audio_eos_token_id",
            "text_blank_token_id",
            "audio_blank_token_id",
            "audio_bos_token_id",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if (
            isinstance(self.audio_delay_tokens, bool)
            or not isinstance(self.audio_delay_tokens, int)
            or self.audio_delay_tokens < 0
        ):
            raise ValueError("audio_delay_tokens must be a non-negative integer.")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be finite and positive.")
        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not math.isfinite(float(self.top_p))
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be in (0, 1].")
        for name in ("do_sample", "use_cache"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean.")


@dataclass(frozen=True)
class MimoGenerationResult:
    text_sequences: Tensor
    audio_sequences: Tensor
    prompt_length: int
    text_finished: Tensor
    audio_finished: Tensor


@torch.no_grad()
def generate_mimo(
    model: MimoGenerationModel,
    text_prompt_ids: Tensor,
    audio_prompt_ids: Tensor,
    options: MimoGenerationOptions,
    *,
    prompt_attention_mask: Tensor | None = None,
    prompt_audio_features: Tensor | None = None,
    prompt_audio_feature_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> MimoGenerationResult:
    """Generate aligned tracks with deterministic delay/blank transitions."""

    attention = _validate_prompts(
        text_prompt_ids,
        audio_prompt_ids,
        prompt_attention_mask,
    )
    feature_state, feature_mask_state = _validate_prompt_features(
        prompt_audio_features,
        prompt_audio_feature_mask,
        text_prompt_ids.shape,
    )
    text = text_prompt_ids
    audio = audio_prompt_ids
    batch_size, prompt_length = text.shape
    text_finished = torch.zeros(batch_size, dtype=torch.bool, device=text.device)
    audio_finished = torch.zeros_like(text_finished)
    past: object | None = None
    step_text = text
    step_audio = audio

    for step in range(options.max_new_tokens):
        step_kwargs: _MimoGenerationStepKwargs = {
            "text_input_ids": step_text,
            "audio_input_ids": step_audio,
            "attention_mask": attention,
            "past_key_values": past,
            "use_cache": options.use_cache,
        }
        # Cached decoding only needs continuous features on the initial
        # prefill.  Without a cache, carry zero/masked rows for generated
        # positions so the prompt features remain aligned with the full input.
        if feature_state is not None and (past is None or not options.use_cache):
            step_kwargs["audio_features"] = feature_state
            step_kwargs["audio_feature_mask"] = feature_mask_state
        output = model.mimo_generation_step(
            **step_kwargs,
        )
        text_logits = _last_logits(output.text_logits, batch_size, "text")
        audio_logits = _last_logits(output.audio_logits, batch_size, "audio")
        next_text = _sample(
            text_logits,
            allowed_token_ids=options.text_allowed_token_ids,
            temperature=options.temperature,
            top_p=options.top_p,
            do_sample=options.do_sample,
            generator=generator,
        )
        next_text = torch.where(
            text_finished,
            next_text.new_full(next_text.shape, options.text_blank_token_id),
            next_text,
        )

        if step < options.audio_delay_tokens:
            next_audio = audio.new_full((batch_size,), options.audio_blank_token_id)
        elif step == options.audio_delay_tokens:
            next_audio = audio.new_full((batch_size,), options.audio_bos_token_id)
        else:
            next_audio = _sample(
                audio_logits,
                allowed_token_ids=options.audio_allowed_token_ids,
                temperature=options.temperature,
                top_p=options.top_p,
                do_sample=options.do_sample,
                generator=generator,
            )
            next_audio = torch.where(
                audio_finished,
                next_audio.new_full(next_audio.shape, options.audio_blank_token_id),
                next_audio,
            )

        text_finished |= next_text.eq(options.text_eos_token_id)
        if step > options.audio_delay_tokens:
            audio_finished |= next_audio.eq(options.audio_eos_token_id)
        text = torch.cat((text, next_text[:, None]), dim=1)
        audio = torch.cat((audio, next_audio[:, None]), dim=1)
        attention = torch.cat(
            (
                attention,
                torch.ones((batch_size, 1), dtype=torch.bool, device=text.device),
            ),
            dim=1,
        )
        if feature_state is not None:
            feature_state = torch.cat(
                (
                    feature_state,
                    torch.zeros(
                        (batch_size, 1, feature_state.size(-1)),
                        dtype=feature_state.dtype,
                        device=feature_state.device,
                    ),
                ),
                dim=1,
            )
            assert feature_mask_state is not None
            feature_mask_state = torch.cat(
                (
                    feature_mask_state,
                    torch.zeros((batch_size, 1), dtype=torch.bool, device=text.device),
                ),
                dim=1,
            )
        past = output.past_key_values if options.use_cache else None
        if past is not None:
            step_text = next_text[:, None]
            step_audio = next_audio[:, None]
        else:
            step_text = text
            step_audio = audio
        if bool((text_finished & audio_finished).all()):
            break

    return MimoGenerationResult(
        text_sequences=text,
        audio_sequences=audio,
        prompt_length=prompt_length,
        text_finished=text_finished,
        audio_finished=audio_finished,
    )


def _validate_prompts(
    text: Tensor,
    audio: Tensor,
    attention_mask: Tensor | None,
) -> Tensor:
    for name, value in (("text_prompt_ids", text), ("audio_prompt_ids", audio)):
        if value.dim() != 2 or not is_signed_integer_dtype(value.dtype):
            raise ValueError(f"{name} must be a signed integer tensor [B, T].")
    if text.shape != audio.shape or text.size(0) < 1 or text.size(1) < 1:
        raise ValueError("MIMO prompts must be non-empty aligned [B, T] tensors.")
    if text.device != audio.device:
        raise ValueError("MIMO prompts must share a device.")
    if attention_mask is None:
        return torch.ones(text.shape, dtype=torch.bool, device=text.device)
    if attention_mask.shape != text.shape or attention_mask.dtype != torch.bool:
        raise ValueError("prompt_attention_mask must be boolean and align with prompts.")
    if attention_mask.device != text.device:
        raise ValueError("prompt_attention_mask must share the prompt device.")
    if not bool(attention_mask[:, -1].all()):
        raise ValueError("each MIMO prompt must end at an attended position.")
    return attention_mask


def _validate_prompt_features(
    features: Tensor | None,
    feature_mask: Tensor | None,
    shape: torch.Size,
) -> tuple[Tensor | None, Tensor | None]:
    if features is None:
        if feature_mask is not None:
            raise ValueError("prompt_audio_feature_mask requires prompt_audio_features.")
        return None, None
    if features.dim() != 3 or features.shape[:2] != shape:
        raise ValueError("prompt_audio_features must have shape [B, T, D].")
    if not features.is_floating_point() or features.size(-1) < 1:
        raise TypeError("prompt_audio_features must be floating-point with D > 0.")
    if feature_mask is None:
        raise ValueError("prompt_audio_feature_mask is required with prompt features.")
    if feature_mask.shape != shape or feature_mask.dtype != torch.bool:
        raise ValueError("prompt_audio_feature_mask must be boolean and align with prompts.")
    if feature_mask.device != features.device:
        raise ValueError("prompt audio features and mask must share a device.")
    return features, feature_mask


def _last_logits(value: Tensor, batch_size: int, name: str) -> Tensor:
    if value.dim() == 3:
        value = value[:, -1]
    if value.dim() != 2 or value.size(0) != batch_size:
        raise ValueError(f"{name}_logits must resolve to shape [B, V].")
    return value


def _sample(
    logits: Tensor,
    *,
    allowed_token_ids: Sequence[int] | Tensor | None,
    temperature: float,
    top_p: float,
    do_sample: bool,
    generator: torch.Generator | None,
) -> Tensor:
    ids = _allowed_ids(allowed_token_ids, logits)
    selected = logits if ids is None else logits.index_select(-1, ids)
    selected = selected / float(temperature)
    if top_p < 1.0:
        selected = _top_p(selected, float(top_p))
    if do_sample:
        probabilities = torch.softmax(selected, dim=-1, dtype=torch.float32)
        indices = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
    else:
        indices = selected.argmax(dim=-1)
    return indices if ids is None else ids.index_select(0, indices)


def _allowed_ids(values: Sequence[int] | Tensor | None, logits: Tensor) -> Tensor | None:
    if values is None:
        return None
    ids = torch.as_tensor(values, dtype=torch.long, device=logits.device)
    if ids.dim() != 1 or ids.numel() < 1:
        raise ValueError("allowed token ids must be a non-empty vector.")
    if bool((ids < 0).any()) or bool((ids >= logits.size(-1)).any()):
        raise ValueError("allowed token ids must be inside the local vocabulary.")
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError("allowed token ids must not contain duplicates.")
    return ids


def _top_p(logits: Tensor, top_p: float) -> Tensor:
    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter(-1, sorted_indices, sorted_logits)


__all__ = [
    "MimoGenerationModel",
    "MimoGenerationOptions",
    "MimoGenerationResult",
    "MimoGenerationStep",
    "generate_mimo",
]
