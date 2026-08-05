from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import Tensor

from .._tensor import is_signed_integer_dtype
from ..audio import AudioCodes
from ..task import PredictionModality
from ..runtime import AudioSequenceLayout
from ..runtime.audio_tokenizer import BiCodecAudioTokenizer
from ..runtime.codec_contract import (
    AcousticCodec,
    Codec,
    GlobalCodec,
    SemanticCodec,
    acoustic_codec,
    codec_sample_rate,
    frame_codec,
    global_codec,
)
from .decode import (
    decode_generated_audio,
    decode_generated_bicodec_full,
    decode_generated_bicodec_codes_row,
    decode_generated_bicodec_row,
    decode_generated_frame_code_row,
    decode_generated_frame_codes,
    decode_generated_semantic_code_row,
    decode_generated_semantic,
)
from .contract import AcousticFeatureGeneration, TokenGenerator
from ..task import Request
from .contract import AudioOutput, Result


@dataclass(frozen=True)
class _Batch:
    requests: Sequence[Request]
    model: TokenGenerator
    prompt: Tensor
    prompt_mask: Tensor
    audio_input_positions: Tensor | None
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    use_cache: bool


class _Strategy(Protocol):
    def generate(self, batch: _Batch) -> list[Result]: ...


class _Decoder(Protocol):
    sample_rate: int

    def decode(self, token_ids: Tensor, features: Tensor | None) -> Tensor: ...


@dataclass(frozen=True)
class _SemanticDecodeOptions:
    reference_features: Tensor | None
    reference_mask: Tensor | None
    generator: torch.Generator | None


_AUDIO_PREFIX_LENGTH = 2


def generate_audio_responses(
    requests: Sequence[Request],
    model: TokenGenerator,
    prompt: Tensor,
    prompt_mask: Tensor,
    audio_input_positions: Tensor | None,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    use_cache: bool,
) -> list[Result]:
    """Generate and decode one homogeneous audio-target request batch."""
    batch = _Batch(
        requests=requests,
        model=model,
        prompt=prompt,
        prompt_mask=prompt_mask,
        audio_input_positions=audio_input_positions,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        use_cache=use_cache,
    )
    return _strategy(model).generate(batch)


def validate_audio_request(
    request: Request,
    model: TokenGenerator,
    *,
    prediction: PredictionModality,
) -> None:
    """Validate audio generation inputs and self-describing codec prompts."""
    if prediction is not PredictionModality.AUDIO:
        raise ValueError("audio generation validation requires an audio prediction task.")
    if "audio_route" in request:
        raise ValueError(
            "generation request audio_route is internal; use runtime audio_sequence_layout."
        )
    _validate_semantic_decode_options(request, model)


def _strategy(model: TokenGenerator) -> _Strategy:
    if _output_detokenizer(model) is None:
        if model.runtime.acoustic_side_channel and isinstance(
            model,
            AcousticFeatureGeneration,
        ):
            return _AcousticCodes(model)
        return _Codes(model)
    if model.runtime.acoustic_side_channel and isinstance(model, AcousticFeatureGeneration):
        return _Acoustic(model)
    if model.runtime.structured_full_sequence:
        return _SemanticGlobal(model)
    if model.runtime.audio_sequence_layout is AudioSequenceLayout.FLATTENED:
        return _Frame(model)
    return _Semantic(model)


def _output_detokenizer(model: TokenGenerator) -> object | None:
    """Read the optional output decoder without eagerly loading legacy aliases."""
    try:
        return model.runtime.output_audio_detokenizer
    except AttributeError:
        return model.runtime.codec


def _output_detokenizer_configured(model: TokenGenerator) -> bool:
    """Check the canonical config without loading a configured decoder."""
    try:
        return model.runtime.output_audio_detokenizer_name is not None
    except AttributeError:
        return _output_detokenizer(model) is not None


class _Codes:
    """Generate output token codes without requiring a waveform decoder."""

    def __init__(self, model: TokenGenerator) -> None:
        self.model = model

    def generate(self, batch: _Batch) -> list[Result]:
        responses = _token_responses(batch)
        variants = _generation_variants(batch)
        return [
            self._result(
                _direct_audio_payload(response, self.model, variant=variant),
                request,
                response_ids=response,
            )
            for response, request, variant in zip(
                responses,
                batch.requests,
                variants,
            )
        ]

    def _result(
        self,
        token_ids: Tensor,
        request: Request | None,
        *,
        response_ids: Tensor | None = None,
        features: Tensor | None = None,
    ) -> Result:
        result_ids = token_ids if response_ids is None else response_ids
        try:
            codes = self._codes(token_ids, request)
            if features is not None:
                if not isinstance(codes, AudioCodes) or codes.semantic_codes is None:
                    raise ValueError("acoustic features require generated semantic codec codes.")
                if codes.semantic_codes.size(0) != features.size(0):
                    raise ValueError("semantic codes and acoustic features must align on frames.")
        except torch.OutOfMemoryError:
            raise
        except Exception as error:
            return _decode_error_result(result_ids, error)
        return _audio_result(
            result_ids,
            None,
            None,
            features=features,
            codes=codes,
        )

    def _codes(self, token_ids: Tensor, request: Request | None) -> AudioCodes | Tensor:
        runtime = self.model.runtime
        if isinstance(runtime.audio_tokenizer, BiCodecAudioTokenizer):
            return decode_generated_bicodec_codes_row(
                token_ids,
                None if request is None else request["prompt_ids"],
                audio_tokenizer=runtime.audio_tokenizer,
                audio_token_range=runtime.codec_audio_range,
                boa_token_id=runtime.boa_token_id,
                eoa_token_id=runtime.eoa_token_id,
                audio_schema_token_id=runtime.audio_schema_token_id,
            )
        if runtime.audio_sequence_layout is AudioSequenceLayout.FLATTENED:
            return decode_generated_frame_code_row(
                token_ids,
                audio_tokenizer=runtime.audio_tokenizer,
                audio_token_range=runtime.codec_audio_range,
            )
        return AudioCodes(
            semantic_codes=decode_generated_semantic_code_row(
                token_ids,
                audio_tokenizer=runtime.audio_tokenizer,
                audio_token_range=runtime.codec_audio_range,
            )
        )


class _AcousticCodes(_Codes):
    """Generate semantic codes and acoustic features without detokenizing."""

    def generate(self, batch: _Batch) -> list[Result]:
        prompt, prompt_mask = _generate_audio_prefix(batch)
        if batch.max_new_tokens <= _AUDIO_PREFIX_LENGTH:
            return [
                _decode_error_result(
                    prompt[row, batch.prompt.size(1) :],
                    ValueError("audio generation budget ended before any codec payload token."),
                )
                for row in range(prompt.size(0))
            ]
        generator = cast(AcousticFeatureGeneration, self.model)
        generated = generator.generate_audio_features(
            prompt,
            max_new_tokens=batch.max_new_tokens - _AUDIO_PREFIX_LENGTH,
            temperature=batch.temperature,
            top_p=batch.top_p,
            prompt_attention_mask=prompt_mask,
            audio_input_positions=batch.audio_input_positions,
            do_sample=batch.do_sample,
            use_cache=batch.use_cache,
        )
        responses = _responses(
            generated["sequence"],
            batch.prompt.size(1),
            self.model.runtime.eoa_token_id,
            retain_stop=True,
        )
        variants = _generation_variants(batch)
        payloads = [
            _direct_audio_payload(response, self.model, variant=variant)
            for response, variant in zip(responses, variants)
        ]
        counts = _frame_count_values(generated["frame_counts"], rows=len(payloads))
        row_features = _feature_rows(
            generated["features"],
            rows=len(payloads),
            counts=counts,
        )
        return [
            self._result(
                payload,
                request,
                response_ids=response,
                features=features,
            )
            for response, payload, request, features in zip(
                responses,
                payloads,
                batch.requests,
                row_features,
            )
        ]


class _Semantic:
    def __init__(self, model: TokenGenerator) -> None:
        self.model = model
        self.codec: SemanticCodec = model.runtime.semantic_codec
        self.sample_rate = codec_sample_rate(self.codec)

    def generate(self, batch: _Batch) -> list[Result]:
        responses = _token_responses(batch)
        variants = _generation_variants(batch)
        payloads = [
            _direct_audio_payload(row, self.model, variant=variant)
            for row, variant in zip(responses, variants)
        ]
        frame_counts = _frame_counts(payloads, self.model)
        if any(has_semantic_decode_options(request) for request in batch.requests):
            return [
                self._decode_result(payload, request, response_ids=response)
                for response, payload, request in zip(
                    responses,
                    payloads,
                    batch.requests,
                )
            ]
        return _decoded_results(
            responses,
            payloads,
            None,
            frame_counts,
            self,
        )

    def _decode_result(
        self,
        token_ids: Tensor,
        request: Request,
        *,
        response_ids: Tensor | None = None,
    ) -> Result:
        result_ids = token_ids if response_ids is None else response_ids
        options = _semantic_decode_options(request)
        try:
            decoded = decode_generated_semantic(
                token_ids.unsqueeze(0),
                codec=self.codec,
                audio_tokenizer=self.model.runtime.audio_tokenizer,
                audio_token_range=self.model.runtime.codec_audio_range,
                semantic_reference_features=_batched(options.reference_features),
                semantic_reference_mask=_batched(options.reference_mask),
                semantic_decode_generator=options.generator,
            )
            if decoded.dim() < 1 or decoded.size(0) != 1:
                raise ValueError("codec decode must preserve the generation batch axis.")
        except torch.OutOfMemoryError:
            raise
        except Exception as error:
            return _decode_error_result(result_ids, error)
        return Result(
            response_ids=result_ids,
            audio=AudioOutput(
                features=None,
                codes=None,
                waveform=decoded[0],
                sample_rate=self.sample_rate,
            ),
        )

    def decode(self, token_ids: Tensor, features: Tensor | None) -> Tensor:
        if features is not None:
            raise ValueError("semantic-only generation must not provide features.")
        return decode_generated_semantic(
            token_ids,
            codec=self.codec,
            audio_tokenizer=self.model.runtime.audio_tokenizer,
            audio_token_range=self.model.runtime.codec_audio_range,
        )


class _Acoustic:
    def __init__(self, model: TokenGenerator) -> None:
        self.model = model
        self.generator = cast(AcousticFeatureGeneration, model)
        backend = _output_detokenizer(model)
        if backend is None:
            raise RuntimeError("waveform generation requires runtime.audio_output.detokenizer.")
        self.codec: AcousticCodec = acoustic_codec(backend)
        self.sample_rate = codec_sample_rate(self.codec)

    def generate(self, batch: _Batch) -> list[Result]:
        prompt, prompt_mask = _generate_audio_prefix(batch)
        if batch.max_new_tokens <= _AUDIO_PREFIX_LENGTH:
            return [
                _decode_error_result(
                    prompt[row, batch.prompt.size(1) :],
                    ValueError("audio generation budget ended before any codec payload token."),
                )
                for row in range(prompt.size(0))
            ]
        generated = self.generator.generate_audio_features(
            prompt,
            max_new_tokens=batch.max_new_tokens - _AUDIO_PREFIX_LENGTH,
            temperature=batch.temperature,
            top_p=batch.top_p,
            prompt_attention_mask=prompt_mask,
            audio_input_positions=batch.audio_input_positions,
            do_sample=batch.do_sample,
            use_cache=batch.use_cache,
        )
        responses = _responses(
            generated["sequence"],
            batch.prompt.size(1),
            self.model.runtime.eoa_token_id,
            retain_stop=True,
        )
        variants = _generation_variants(batch)
        payloads = [
            _direct_audio_payload(row, self.model, variant=variant)
            for row, variant in zip(responses, variants)
        ]
        return _decoded_results(
            responses,
            payloads,
            generated["features"],
            generated["frame_counts"],
            self,
        )

    def decode(self, token_ids: Tensor, features: Tensor | None) -> Tensor:
        if features is None:
            raise ValueError("acoustic generation requires generated features.")
        return decode_generated_audio(
            token_ids,
            features,
            codec=self.codec,
            audio_tokenizer=self.model.runtime.audio_tokenizer,
            audio_token_range=self.model.runtime.codec_audio_range,
        )


class _Frame:
    def __init__(self, model: TokenGenerator) -> None:
        self.model = model
        backend = _output_detokenizer(model)
        if backend is None:
            raise RuntimeError("waveform generation requires runtime.audio_output.detokenizer.")
        self.codec: Codec = frame_codec(backend)
        self.sample_rate = codec_sample_rate(self.codec)

    def generate(self, batch: _Batch) -> list[Result]:
        responses = _token_responses(batch)
        variants = _generation_variants(batch)
        return [
            self._decode_result(
                _direct_audio_payload(response, self.model, variant=variant),
                response_ids=response,
            )
            for response, variant in zip(responses, variants)
        ]

    def _decode_result(
        self,
        token_ids: Tensor,
        *,
        response_ids: Tensor | None = None,
    ) -> Result:
        result_ids = token_ids if response_ids is None else response_ids
        try:
            waveform = self._decode_row(token_ids)
        except torch.OutOfMemoryError:
            raise
        except Exception as error:
            return _decode_error_result(result_ids, error)
        return _audio_result(result_ids, waveform, self.sample_rate)

    def _decode_row(self, token_ids: Tensor) -> Tensor:
        decoded = self.decode(token_ids.unsqueeze(0), None)
        if decoded.dim() < 1 or decoded.size(0) != 1:
            raise ValueError("codec decode must preserve a batch size of one.")
        return decoded[0]

    def decode(self, token_ids: Tensor, features: Tensor | None) -> Tensor:
        if features is not None:
            raise ValueError("full frame-code generation must not provide features.")
        return decode_generated_frame_codes(
            token_ids,
            codec=self.codec,
            audio_tokenizer=self.model.runtime.audio_tokenizer,
            audio_token_range=self.model.runtime.codec_audio_range,
        )


class _SemanticGlobal:
    def __init__(self, model: TokenGenerator) -> None:
        tokenizer = model.runtime.audio_tokenizer
        if not isinstance(tokenizer, BiCodecAudioTokenizer):
            raise TypeError("semantic-global generation requires BiCodecAudioTokenizer.")
        self.model = model
        self.tokenizer = tokenizer
        backend = _output_detokenizer(model)
        if backend is None:
            raise RuntimeError("waveform generation requires runtime.audio_output.detokenizer.")
        self.codec: GlobalCodec = global_codec(backend)
        self.sample_rate = codec_sample_rate(self.codec)

    def generate(self, batch: _Batch) -> list[Result]:
        responses = _token_responses(batch)
        variants = _generation_variants(batch)
        return [
            self._decode_result(
                _direct_audio_payload(response, self.model, variant=variant),
                request,
                response_ids=response,
            )
            for response, request, variant in zip(
                responses,
                batch.requests,
                variants,
            )
        ]

    def _decode_result(
        self,
        token_ids: Tensor,
        request: Request | None,
        *,
        response_ids: Tensor | None = None,
    ) -> Result:
        result_ids = token_ids if response_ids is None else response_ids
        try:
            waveform, codes = self._decode_row(token_ids, request)
        except torch.OutOfMemoryError:
            raise
        except Exception as error:
            return _decode_error_result(result_ids, error)
        return _audio_result(
            result_ids,
            waveform,
            self.sample_rate,
            codes=codes,
        )

    def _decode_row(
        self,
        token_ids: Tensor,
        request: Request | None,
    ) -> tuple[Tensor, AudioCodes]:
        return decode_generated_bicodec_row(
            token_ids,
            None if request is None else request["prompt_ids"],
            codec=self.codec,
            audio_tokenizer=self.tokenizer,
            audio_token_range=self.model.runtime.codec_audio_range,
            boa_token_id=self.model.runtime.boa_token_id,
            eoa_token_id=self.model.runtime.eoa_token_id,
            audio_schema_token_id=self.model.runtime.audio_schema_token_id,
        )

    def decode(self, token_ids: Tensor, features: Tensor | None) -> Tensor:
        if features is not None:
            raise ValueError("semantic-global generation must not provide features.")
        return decode_generated_bicodec_full(
            token_ids,
            codec=self.codec,
            audio_tokenizer=self.tokenizer,
            audio_token_range=self.model.runtime.codec_audio_range,
        )


def _token_responses(batch: _Batch) -> list[Tensor]:
    prompt, prompt_mask = _generate_audio_prefix(batch)
    remaining = batch.max_new_tokens - _AUDIO_PREFIX_LENGTH
    if remaining <= 0:
        return [prompt[row, batch.prompt.size(1) :] for row in range(prompt.size(0))]
    start, end = batch.model.runtime.codec_audio_range
    sequence = batch.model.generate_tokens(
        prompt,
        max_new_tokens=remaining,
        temperature=batch.temperature,
        top_p=batch.top_p,
        prompt_attention_mask=prompt_mask,
        audio_input_positions=batch.audio_input_positions,
        stop_token_id=batch.model.runtime.eoa_token_id,
        allowed_token_ids=(*range(start, end), batch.model.runtime.eoa_token_id),
        do_sample=batch.do_sample,
        use_cache=batch.use_cache,
    )
    return _responses(
        sequence,
        batch.prompt.size(1),
        batch.model.runtime.eoa_token_id,
        retain_stop=True,
    )


def _generate_audio_prefix(batch: _Batch) -> tuple[Tensor, Tensor]:
    prompt = batch.prompt
    prompt_mask = batch.prompt_mask
    expected_ids = (
        batch.model.runtime.boa_token_id,
        batch.model.runtime.audio_schema_token_id,
    )
    for expected_id in expected_ids[: batch.max_new_tokens]:
        sequence = batch.model.generate_tokens(
            prompt,
            max_new_tokens=1,
            temperature=batch.temperature,
            top_p=batch.top_p,
            prompt_attention_mask=prompt_mask,
            audio_input_positions=batch.audio_input_positions,
            allowed_token_ids=(expected_id,),
            do_sample=batch.do_sample,
            use_cache=batch.use_cache,
        )
        if sequence.shape != (prompt.size(0), prompt.size(1) + 1):
            raise ValueError("audio prefix generation must append exactly one token per row.")
        if not bool(sequence[:, -1].eq(expected_id).all()):
            raise ValueError("audio prefix generation emitted an invalid control token.")
        prompt = sequence
        prompt_mask = torch.cat(
            (
                prompt_mask,
                torch.ones(
                    (prompt_mask.size(0), 1),
                    dtype=torch.bool,
                    device=prompt_mask.device,
                ),
            ),
            dim=1,
        )
    return prompt, prompt_mask


def _responses(
    sequence: Tensor,
    prompt_length: int,
    stop_token_id: int,
    *,
    retain_stop: bool = False,
) -> list[Tensor]:
    return [
        _response(
            sequence[row],
            prompt_length,
            stop_token_id,
            retain_stop=retain_stop,
        )
        for row in range(sequence.size(0))
    ]


def _response(
    sequence: Tensor,
    prompt_length: int,
    stop_token_id: int,
    *,
    retain_stop: bool = False,
) -> Tensor:
    response = sequence[prompt_length:]
    stops = response.eq(stop_token_id).nonzero()
    if stops.numel():
        stop = int(stops[0].item())
        return response[: stop + int(retain_stop)]
    return response


def _direct_audio_payload(
    response_ids: Tensor,
    model: TokenGenerator,
    *,
    variant: str,
) -> Tensor:
    if response_ids.dim() != 1:
        raise ValueError("generated audio response must have shape [tokens].")
    if response_ids.numel() < _AUDIO_PREFIX_LENGTH:
        raise ValueError("generated audio response is missing its audio prefix.")
    if int(response_ids[0].item()) != model.runtime.boa_token_id:
        raise ValueError("generated audio response must begin with BOA.")
    if int(response_ids[1].item()) != model.runtime.audio_schema_token_id:
        raise ValueError("generated audio response has the wrong schema selector.")
    payload = response_ids[_AUDIO_PREFIX_LENGTH:]
    stops = payload.eq(model.runtime.eoa_token_id).nonzero(as_tuple=False)
    if stops.numel():
        stop = int(stops[0].item())
        if stop + 1 != payload.numel():
            raise ValueError("generated audio response contains tokens after EOA.")
        payload = payload[:stop]
    if payload.numel() == 0:
        raise ValueError("generated audio response contains no codec payload.")
    start, end = model.runtime.codec_audio_range
    if bool((payload < start).any()) or bool((payload >= end).any()):
        raise ValueError("generated audio response contains non-codec payload tokens.")
    _validate_complete_payload(payload, model, variant=variant)
    return payload


def _validate_complete_payload(
    token_ids: Tensor,
    model: TokenGenerator,
    *,
    variant: str,
) -> None:
    start, _ = model.runtime.codec_audio_range
    local_ids = token_ids.to(dtype=torch.long) - start
    spec = model.runtime.output_audio_token_spec
    try:
        spec.validate_complete(local_ids, variant=variant)
    except ValueError as error:
        raise ValueError(
            f"generated audio payload violates its selected schema variant {variant!r}. {error}"
        ) from error


def audio_generation_variant(request: Request, model: TokenGenerator) -> str:
    """Resolve the output codec grammar variant from prompt-owned audio spans."""
    spec = model.runtime.output_audio_token_spec
    if not spec.grammar.prompt_continuations:
        return spec.generation_variant(())
    payloads = _prompt_audio_payloads(request["prompt_ids"], model)
    return spec.generation_variant(payloads)


def _generation_variants(batch: _Batch) -> tuple[str, ...]:
    return tuple(audio_generation_variant(request, batch.model) for request in batch.requests)


def _prompt_audio_payloads(
    prompt_ids: Tensor,
    model: TokenGenerator,
) -> tuple[Tensor, ...]:
    if prompt_ids.dim() != 1:
        raise ValueError("generation prompt ids must have shape [tokens].")
    runtime = model.runtime
    codec_start, codec_end = runtime.codec_audio_range
    payloads: list[Tensor] = []
    cursor = 0
    while cursor < prompt_ids.numel():
        starts = prompt_ids[cursor:].eq(runtime.boa_token_id).nonzero(as_tuple=False)
        if starts.numel() == 0:
            break
        span_start = cursor + int(starts[0].item())
        schema_position = span_start + 1
        if (
            schema_position >= prompt_ids.numel()
            or int(prompt_ids[schema_position].item()) != runtime.audio_schema_token_id
        ):
            raise ValueError("prompt audio span has the wrong output schema selector.")
        payload_start = schema_position + 1
        stops = prompt_ids[payload_start:].eq(runtime.eoa_token_id).nonzero(as_tuple=False)
        if stops.numel() == 0:
            raise ValueError("prompt audio span is missing EOA.")
        payload_end = payload_start + int(stops[0].item())
        payload = prompt_ids[payload_start:payload_end]
        if payload.numel() == 0:
            raise ValueError("prompt audio span contains no codec payload.")
        if bool((payload < codec_start).any()) or bool((payload >= codec_end).any()):
            raise ValueError("prompt audio span contains non-codec payload tokens.")
        payloads.append(payload.to(dtype=torch.long) - codec_start)
        cursor = payload_end + 1
    return tuple(payloads)


def _frame_counts(token_rows: list[Tensor], model: TokenGenerator) -> Tensor:
    if any(token_ids.numel() == 0 for token_ids in token_rows):
        raise ValueError("audio generation produced no codec-decodable tokens.")
    start, _ = model.runtime.codec_audio_range
    counts = []
    span_lookup = model.audio_token_frame_spans
    for token_ids in token_rows:
        local = token_ids - start
        if bool((local < 0).any()) or bool((local >= span_lookup.numel()).any()):
            raise ValueError("audio generation produced non-codec audio tokens.")
        spans = span_lookup.index_select(0, local.to(device=span_lookup.device))
        counts.append(spans.sum().to(device=local.device))
    return torch.stack(counts)


def _decoded_results(
    response_rows: list[Tensor],
    decode_rows: list[Tensor],
    features: Tensor | None,
    frame_counts: Tensor,
    decoder: _Decoder,
) -> list[Result]:
    if len(response_rows) != len(decode_rows):
        raise ValueError("audio response and codec payload rows must align.")
    counts = _frame_count_values(frame_counts, rows=len(decode_rows))
    row_features = _feature_rows(
        features,
        rows=len(decode_rows),
        counts=counts,
    )
    decoded_rows = _decode_grouped_rows(
        decode_rows,
        row_features,
        counts,
        decoder,
    )
    results: list[Result] = []
    for row, response_ids in enumerate(response_rows):
        decoded = decoded_rows[row]
        if isinstance(decoded, Exception):
            results.append(_decode_error_result(response_ids, decoded))
        else:
            results.append(
                _audio_result(
                    response_ids,
                    decoded,
                    decoder.sample_rate,
                    features=row_features[row],
                )
            )
    return results


def _audio_result(
    token_ids: Tensor,
    waveform: Tensor | None,
    sample_rate: int | None,
    *,
    features: Tensor | None = None,
    codes: AudioCodes | Tensor | None = None,
) -> Result:
    if (waveform is None) != (sample_rate is None):
        raise ValueError("audio waveform and sample rate must be present together.")
    if waveform is None and codes is None:
        raise ValueError("audio output requires waveform or codec codes.")
    return Result(
        response_ids=token_ids,
        audio=AudioOutput(
            features=features,
            codes=codes,
            waveform=waveform,
            sample_rate=sample_rate,
        ),
    )


def _decode_error_result(token_ids: Tensor, error: Exception) -> Result:
    warnings.warn(
        f"skipping invalid generated audio sequence: {type(error).__name__}: {error}",
        RuntimeWarning,
        stacklevel=2,
    )
    return Result(
        response_ids=token_ids,
        audio=None,
        decode_error={
            "type": type(error).__name__,
            "message": str(error),
        },
    )


def _frame_count_values(frame_counts: Tensor, *, rows: int) -> list[int]:
    frame_counts = _integer_tensor(
        frame_counts,
        "generated audio frame counts",
        dimensions=1,
    )
    if frame_counts.shape != (rows,):
        raise ValueError("generated audio frame counts must provide one value per row.")
    counts = [int(count) for count in frame_counts.detach().cpu().tolist()]
    if any(count < 1 for count in counts):
        raise ValueError("each audio generation row must contain at least one frame.")
    return counts


def _feature_rows(
    features: Tensor | None,
    *,
    rows: int,
    counts: list[int],
) -> list[Tensor | None]:
    if features is not None:
        if features.dim() != 3 or features.size(0) != rows:
            raise ValueError("generated acoustic features must have shape [batch, frames, dim].")
        if any(count > features.size(1) for count in counts):
            raise ValueError("generated frame count exceeds acoustic feature padding.")
        return [features[row, :count] for row, count in enumerate(counts)]
    return [None] * rows


def _decode_grouped_rows(
    token_rows: list[Tensor],
    row_features: list[Tensor | None],
    counts: list[int],
    decoder: _Decoder,
) -> list[Tensor | Exception]:
    groups: dict[tuple[int, int], list[int]] = {}
    for row, (token_ids, count) in enumerate(zip(token_rows, counts)):
        groups.setdefault((token_ids.numel(), count), []).append(row)

    decoded_rows: list[Tensor | Exception | None] = [None] * len(token_rows)
    for rows in groups.values():
        try:
            decoded = _decode_batch(token_rows, row_features, rows, decoder)
        except torch.OutOfMemoryError:
            raise
        except Exception as batch_error:
            if len(rows) == 1:
                decoded_rows[rows[0]] = batch_error
                continue
            for row in rows:
                try:
                    decoded = _decode_batch(
                        token_rows,
                        row_features,
                        [row],
                        decoder,
                    )
                except torch.OutOfMemoryError:
                    raise
                except Exception as row_error:
                    decoded_rows[row] = row_error
                else:
                    decoded_rows[row] = decoded[0]
            continue
        for row, waveform in zip(rows, decoded):
            decoded_rows[row] = waveform

    results: list[Tensor | Exception] = []
    for decoded in decoded_rows:
        if decoded is None:
            raise RuntimeError("codec decode did not produce every generation row.")
        results.append(decoded)
    return results


def _decode_batch(
    token_rows: list[Tensor],
    row_features: list[Tensor | None],
    rows: list[int],
    decoder: _Decoder,
) -> Tensor:
    token_batch = torch.stack([token_rows[row] for row in rows])
    first_features = row_features[rows[0]]
    feature_batch = (
        None
        if first_features is None
        else torch.stack([cast(Tensor, row_features[row]) for row in rows])
    )
    decoded = decoder.decode(token_batch, feature_batch)
    if decoded.dim() < 1 or decoded.size(0) != len(rows):
        raise ValueError("codec decode must preserve the generation batch axis.")
    return decoded


def _integer_tensor(value: object, name: str, *, dimensions: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    if not is_signed_integer_dtype(value.dtype):
        raise TypeError(f"{name} must contain integer ids using a signed dtype.")
    if value.dim() != dimensions:
        raise ValueError(f"{name} must have {dimensions} dimensions.")
    return value


def decode_token_audio_results(
    token_rows: Sequence[Tensor],
    model: TokenGenerator,
    *,
    requests: Sequence[Request] | None = None,
) -> list[Result]:
    """Decode codec-decodable spans already present in generated token rows.

    Used by mixed AR after token generation. Does not collect or consume acoustic
    frame conditions; acoustic feature generators must fail explicitly.
    """
    if model.runtime.acoustic_side_channel and isinstance(model, AcousticFeatureGeneration):
        raise ValueError("token-row audio decode does not support acoustic feature side channel.")
    if requests is not None and len(requests) != len(token_rows):
        raise ValueError("audio decode requests must align with generated token rows.")
    variants = (
        tuple(model.runtime.output_audio_token_spec.generation_variant(()) for _ in token_rows)
        if requests is None
        else tuple(audio_generation_variant(request, model) for request in requests)
    )
    parsed_rows: list[Tensor | Exception] = []
    for row, variant in zip(token_rows, variants):
        try:
            parsed_rows.append(_codec_payload(row, model, variant=variant))
        except Exception as error:
            parsed_rows.append(error)
    results = [Result(response_ids=row, audio=None) for row in token_rows]
    for index, parsed in enumerate(parsed_rows):
        if isinstance(parsed, Exception):
            results[index] = _decode_error_result(token_rows[index], parsed)
    if not any(isinstance(row, Tensor) and row.numel() > 0 for row in parsed_rows):
        return results

    decoder = _token_decoder(model)
    active_index = [
        index
        for index, row in enumerate(parsed_rows)
        if isinstance(row, Tensor) and row.numel() > 0
    ]
    active_rows = [cast(Tensor, parsed_rows[index]) for index in active_index]
    active_responses = [token_rows[index] for index in active_index]
    active_requests = [None if requests is None else requests[index] for index in active_index]
    if isinstance(decoder, _Codes):
        decoded = [
            decoder._result(
                row,
                request,
                response_ids=response,
            )
            for row, request, response in zip(
                active_rows,
                active_requests,
                active_responses,
            )
        ]
    elif isinstance(decoder, _Frame):
        decoded = [
            decoder._decode_result(row, response_ids=response)
            for row, response in zip(active_rows, active_responses)
        ]
    elif isinstance(decoder, _SemanticGlobal):
        decoded = [
            decoder._decode_result(row, request, response_ids=response)
            for row, request, response in zip(
                active_rows,
                active_requests,
                active_responses,
            )
        ]
    elif isinstance(decoder, _Semantic) and any(
        request is not None and has_semantic_decode_options(request) for request in active_requests
    ):
        decoded = [
            decoder._decode_result(
                row,
                request,
                response_ids=response,
            )
            if request is not None
            else _decode_error_result(
                response,
                ValueError("semantic decode options require the generation request."),
            )
            for row, request, response in zip(
                active_rows,
                active_requests,
                active_responses,
            )
        ]
    elif isinstance(decoder, _Semantic):
        decoded = _decoded_results(
            active_responses,
            active_rows,
            None,
            _frame_counts(active_rows, model),
            decoder,
        )
    else:
        raise AssertionError("unsupported token-row audio decoder.")
    for index, result in zip(active_index, decoded):
        results[index] = result
    return results


def _codec_payload(
    token_ids: Tensor,
    model: TokenGenerator,
    *,
    variant: str,
) -> Tensor:
    if token_ids.dim() != 1:
        raise ValueError("generated response ids must have shape [tokens].")
    start, end = model.runtime.codec_audio_range
    payloads: list[Tensor] = []
    cursor = 0
    while cursor < token_ids.numel():
        starts = token_ids[cursor:].eq(model.runtime.boa_token_id).nonzero(as_tuple=False)
        if starts.numel() == 0:
            break
        span_start = cursor + int(starts[0].item())
        schema_position = span_start + 1
        if (
            schema_position >= token_ids.numel()
            or int(token_ids[schema_position].item()) != model.runtime.audio_schema_token_id
        ):
            raise ValueError("generated audio span has the wrong schema selector.")
        payload_start = schema_position + 1
        stops = token_ids[payload_start:].eq(model.runtime.eoa_token_id).nonzero(as_tuple=False)
        if stops.numel():
            payload_end = payload_start + int(stops[0].item())
            cursor = payload_end + 1
        else:
            payload_end = token_ids.numel()
            cursor = payload_end
        payload = token_ids[payload_start:payload_end]
        if payload.numel() == 0:
            raise ValueError("generated audio span contains no codec payload.")
        if bool((payload < start).any()) or bool((payload >= end).any()):
            raise ValueError("generated audio span contains non-codec payload tokens.")
        _validate_complete_payload(payload, model, variant=variant)
        payloads.append(payload)
    if not payloads:
        return token_ids.new_empty(0)
    return torch.cat(payloads)


def _token_decoder(model: TokenGenerator) -> _Decoder | _Codes:
    if _output_detokenizer(model) is None:
        return _Codes(model)
    if model.runtime.structured_full_sequence:
        return _SemanticGlobal(model)
    if model.runtime.audio_sequence_layout is AudioSequenceLayout.FLATTENED:
        return _Frame(model)
    return _Semantic(model)


def has_semantic_decode_options(request: Request) -> bool:
    return (
        request.get("semantic_reference_features") is not None
        or request.get("semantic_reference_mask") is not None
        or request.get("semantic_decode_generator") is not None
    )


def _validate_semantic_decode_options(
    request: Request,
    model: TokenGenerator,
) -> None:
    if not has_semantic_decode_options(request):
        return
    if not _output_detokenizer_configured(model):
        raise ValueError(
            "semantic decode options require runtime.audio_output.detokenizer."
        )
    if (
        model.runtime.audio_sequence_layout is AudioSequenceLayout.FLATTENED
        or model.runtime.structured_full_sequence
    ):
        raise ValueError("semantic decode options require semantic-only audio generation.")
    if model.runtime.acoustic_side_channel and isinstance(
        model,
        AcousticFeatureGeneration,
    ):
        raise ValueError(
            "semantic decode options are not supported with acoustic side-channel generation."
        )

    features = request.get("semantic_reference_features")
    if features is not None:
        if not isinstance(features, Tensor):
            raise TypeError("semantic reference features must be a Tensor.")
        if features.dim() != 2:
            raise ValueError("semantic reference features must have shape [frames, dim].")
    mask = request.get("semantic_reference_mask")
    if mask is not None:
        if not isinstance(mask, Tensor):
            raise TypeError("semantic reference mask must be a Tensor.")
        if mask.dtype != torch.bool:
            raise TypeError("semantic reference mask must be a bool Tensor.")
        if mask.dim() != 1:
            raise ValueError("semantic reference mask must have shape [frames].")
        if features is None:
            raise ValueError("semantic reference mask requires semantic reference features.")
        if mask.size(0) != features.size(0):
            raise ValueError("semantic reference mask must align with reference feature frames.")
    generator = request.get("semantic_decode_generator")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("semantic decode generator must be a torch.Generator.")


def _semantic_decode_options(request: Request) -> _SemanticDecodeOptions:
    return _SemanticDecodeOptions(
        reference_features=request.get("semantic_reference_features"),
        reference_mask=request.get("semantic_reference_mask"),
        generator=request.get("semantic_decode_generator"),
    )


def _batched(value: Tensor | None) -> Tensor | None:
    if value is None:
        return None
    return value.unsqueeze(0)


__all__ = [
    "audio_generation_variant",
    "decode_token_audio_results",
    "generate_audio_responses",
    "has_semantic_decode_options",
    "validate_audio_request",
]
