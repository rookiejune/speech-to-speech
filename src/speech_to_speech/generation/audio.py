from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from anydataset.types import Modality
from torch import Tensor

from .._tensor import is_signed_integer_dtype
from ..audio import AudioCodes
from ..task import PredictionModality
from ..runtime import AudioSequenceLayout
from ..runtime.audio_tokenizer import BiCodecAudioTokenizer
from ..runtime.codec_contract import (
    AcousticCodec,
    Codec,
    SemanticCodec,
    StructuredCodec,
    acoustic_codec,
    codec_sample_rate,
    frame_codec,
    structured_codec,
)
from .decode import (
    decode_generated_audio,
    decode_generated_bicodec_full,
    decode_generated_bicodec_row,
    decode_generated_frame_codes,
    decode_generated_semantic,
)
from .protocol import AcousticFeatureGeneration, TokenGenerator
from ..task import Request
from .result import AudioOutput, Result


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


def validate_audio_request(request: Request, model: TokenGenerator) -> None:
    """Validate audio generation inputs and self-describing codec prompts."""
    prediction = request.get("prediction")
    if prediction is None:
        prediction = request["task"].prediction_modality
    if prediction is not PredictionModality.AUDIO:
        raise ValueError("audio generation validation requires an audio prediction task.")
    if "audio_route" in request:
        raise ValueError(
            "generation request audio_route is internal; use runtime audio_sequence_layout."
        )
    _validate_semantic_decode_options(request, model)


def _strategy(model: TokenGenerator) -> _Strategy:
    if model.runtime.acoustic_side_channel and isinstance(model, AcousticFeatureGeneration):
        return _Acoustic(model)
    if model.runtime.structured_full_sequence:
        return _Structured(model)
    if model.runtime.audio_sequence_layout is AudioSequenceLayout.FLATTENED:
        return _Frame(model)
    return _Semantic(model)


class _Semantic:
    def __init__(self, model: TokenGenerator) -> None:
        self.model = model
        self.codec: SemanticCodec = model.runtime.semantic_codec
        self.sample_rate = codec_sample_rate(self.codec)

    def generate(self, batch: _Batch) -> list[Result]:
        responses = _token_responses(batch)
        frame_counts = _frame_counts(responses, self.model)
        if any(has_semantic_decode_options(request) for request in batch.requests):
            return [
                self._decode_result(token_ids, request)
                for token_ids, request in zip(responses, batch.requests)
            ]
        return _decoded_results(
            responses,
            None,
            frame_counts,
            self,
        )

    def _decode_result(self, token_ids: Tensor, request: Request) -> Result:
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
            return _decode_error_result(token_ids, error)
        return Result(
            response_ids=token_ids,
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
        self.codec: AcousticCodec = acoustic_codec(model.runtime.codec)
        self.sample_rate = codec_sample_rate(self.codec)

    def generate(self, batch: _Batch) -> list[Result]:
        generated = self.generator.generate_audio_features(
            batch.prompt,
            max_new_tokens=batch.max_new_tokens,
            temperature=batch.temperature,
            top_p=batch.top_p,
            prompt_attention_mask=batch.prompt_mask,
            audio_input_positions=batch.audio_input_positions,
            do_sample=batch.do_sample,
            use_cache=batch.use_cache,
        )
        responses = _responses(
            generated["sequence"],
            batch.prompt.size(1),
            self.model.runtime.eoa_token_id,
        )
        return _decoded_results(
            responses,
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
        self.codec: Codec = frame_codec(model.runtime.codec)
        self.sample_rate = codec_sample_rate(self.codec)

    def generate(self, batch: _Batch) -> list[Result]:
        return [self._decode_result(token_ids) for token_ids in _token_responses(batch)]

    def _decode_result(self, token_ids: Tensor) -> Result:
        try:
            waveform = self._decode_row(token_ids)
        except torch.OutOfMemoryError:
            raise
        except Exception as error:
            return _decode_error_result(token_ids, error)
        return _audio_result(token_ids, waveform, self.sample_rate)

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


class _Structured:
    def __init__(self, model: TokenGenerator) -> None:
        tokenizer = model.runtime.audio_tokenizer
        if not isinstance(tokenizer, BiCodecAudioTokenizer):
            raise TypeError("structured audio generation requires BiCodecAudioTokenizer.")
        self.model = model
        self.tokenizer = tokenizer
        self.codec: StructuredCodec = structured_codec(model.runtime.codec)
        self.sample_rate = codec_sample_rate(self.codec)

    def generate(self, batch: _Batch) -> list[Result]:
        return [
            self._decode_result(token_ids, request)
            for token_ids, request in zip(_token_responses(batch), batch.requests)
        ]

    def _decode_result(self, token_ids: Tensor, request: Request | None) -> Result:
        try:
            waveform, codes = self._decode_row(token_ids, request)
        except torch.OutOfMemoryError:
            raise
        except Exception as error:
            return _decode_error_result(token_ids, error)
        return _audio_result(
            token_ids,
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
        )

    def decode(self, token_ids: Tensor, features: Tensor | None) -> Tensor:
        if features is not None:
            raise ValueError("structured audio generation must not provide features.")
        return decode_generated_bicodec_full(
            token_ids,
            codec=self.codec,
            audio_tokenizer=self.tokenizer,
            audio_token_range=self.model.runtime.codec_audio_range,
        )


def _token_responses(batch: _Batch) -> list[Tensor]:
    sequence = batch.model.generate_tokens(
        batch.prompt,
        max_new_tokens=batch.max_new_tokens,
        temperature=batch.temperature,
        top_p=batch.top_p,
        prompt_attention_mask=batch.prompt_mask,
        audio_input_positions=batch.audio_input_positions,
        stop_token_id=batch.model.runtime.eoa_token_id,
        generation_modality=Modality.AUDIO,
        do_sample=batch.do_sample,
        use_cache=batch.use_cache,
    )
    return _responses(
        sequence,
        batch.prompt.size(1),
        batch.model.runtime.eoa_token_id,
    )


def _responses(sequence: Tensor, prompt_length: int, stop_token_id: int) -> list[Tensor]:
    return [
        _response(sequence[row], prompt_length, stop_token_id) for row in range(sequence.size(0))
    ]


def _response(sequence: Tensor, prompt_length: int, stop_token_id: int) -> Tensor:
    response = sequence[prompt_length:]
    stops = response.eq(stop_token_id).nonzero()
    if stops.numel():
        return response[: int(stops[0].item())]
    return response


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
    token_rows: list[Tensor],
    features: Tensor | None,
    frame_counts: Tensor,
    decoder: _Decoder,
) -> list[Result]:
    counts = _frame_count_values(frame_counts, rows=len(token_rows))
    row_features = _feature_rows(
        features,
        rows=len(token_rows),
        counts=counts,
    )
    decoded_rows = _decode_grouped_rows(
        token_rows,
        row_features,
        counts,
        decoder,
    )
    results: list[Result] = []
    for row, token_ids in enumerate(token_rows):
        decoded = decoded_rows[row]
        if isinstance(decoded, Exception):
            results.append(_decode_error_result(token_ids, decoded))
        else:
            results.append(
                _audio_result(
                    token_ids,
                    decoded,
                    decoder.sample_rate,
                    features=row_features[row],
                )
            )
    return results


def _audio_result(
    token_ids: Tensor,
    waveform: Tensor,
    sample_rate: int,
    *,
    features: Tensor | None = None,
    codes: AudioCodes | None = None,
) -> Result:
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
        "skipping invalid generated audio sequence: "
        f"{type(error).__name__}: {error}",
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
) -> list[Result]:
    """Decode codec-decodable spans already present in generated token rows.

    Used by mixed AR after token generation. Does not collect or consume acoustic
    frame conditions; acoustic feature generators must fail explicitly.
    """
    if model.runtime.acoustic_side_channel and isinstance(model, AcousticFeatureGeneration):
        raise ValueError("token-row audio decode does not support acoustic feature side channel.")
    codec_rows = [_codec_payload(row, model) for row in token_rows]
    results = [Result(response_ids=row, audio=None) for row in token_rows]
    if not any(row.numel() > 0 for row in codec_rows):
        return results

    decoder = _token_decoder(model)
    active_index = [index for index, row in enumerate(codec_rows) if row.numel() > 0]
    active_rows = [codec_rows[index] for index in active_index]
    if isinstance(decoder, _Frame):
        decoded = [decoder._decode_result(row) for row in active_rows]
    elif isinstance(decoder, _Structured):
        decoded = [decoder._decode_result(row, None) for row in active_rows]
    else:
        decoded = _decoded_results(
            active_rows,
            None,
            _frame_counts(active_rows, model),
            decoder,
        )
    for index, result in zip(active_index, decoded):
        result["response_ids"] = token_rows[index]
        results[index] = result
    return results


def _codec_payload(token_ids: Tensor, model: TokenGenerator) -> Tensor:
    start, end = model.runtime.codec_audio_range
    mask = token_ids.ge(start) & token_ids.lt(end)
    return token_ids[mask]


def _token_decoder(model: TokenGenerator) -> _Decoder:
    if model.runtime.structured_full_sequence:
        return _Structured(model)
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
    "decode_token_audio_results",
    "generate_audio_responses",
    "has_semantic_decode_options",
    "validate_audio_request",
]
