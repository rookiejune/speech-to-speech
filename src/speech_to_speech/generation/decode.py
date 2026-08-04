from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from torch import Generator, Tensor

from .._tensor import is_signed_integer_dtype
from ..codes import AudioCodes
from ..runtime.audio_tokenizer import (
    BiCodecAudioTokenizer,
    semantic_codes_from_audio_tokens,
)
from ..runtime.types import (
    AcousticCodec,
    AudioTokenizer,
    Codec,
    CodecBackend,
    SemanticCodec,
    StructuredCodec,
    frame_codec,
    structured_codec,
)


def decode_generated_audio(
    audio_token_ids: Tensor,
    acoustic_features: Tensor,
    *,
    codec: AcousticCodec,
    audio_tokenizer: AudioTokenizer,
    audio_token_range: tuple[int, int],
) -> Tensor:
    """Decode generated audio tokens and acoustic features into waveforms."""
    local_ids = _local_ids(audio_token_ids, audio_token_range)
    return _decode_audio(
        local_ids,
        acoustic_features,
        codec=codec,
        audio_tokenizer=audio_tokenizer,
    )


def _decode_audio(
    local_ids: Tensor,
    acoustic_features: Tensor,
    *,
    codec: AcousticCodec,
    audio_tokenizer: AudioTokenizer,
) -> Tensor:
    rows = [semantic_codes_from_audio_tokens(audio_tokenizer, row) for row in local_ids]
    if not rows or len({tuple(row.shape) for row in rows}) != 1:
        raise ValueError(
            "audio token rows must expand to the same frame and codebook shape."
        )
    semantic_codes = torch.stack(rows)
    if semantic_codes.shape[:2] != acoustic_features.shape[:2]:
        raise ValueError(
            "semantic codes and acoustic features must align on [batch, frame]."
        )
    return codec.decode_features(semantic_codes, acoustic_features)


def decode_generated_semantic(
    audio_token_ids: Tensor,
    *,
    codec: SemanticCodec,
    audio_tokenizer: AudioTokenizer,
    audio_token_range: tuple[int, int],
    semantic_reference_features: Tensor | None = None,
    semantic_reference_mask: Tensor | None = None,
    semantic_decode_generator: Generator | None = None,
) -> Tensor:
    """Decode semantic-only codec tokens directly into waveforms."""
    local_ids = _local_ids(audio_token_ids, audio_token_range)
    semantic_codes = torch.stack(
        [semantic_codes_from_audio_tokens(audio_tokenizer, row) for row in local_ids]
    )
    return codec.decode(
        semantic_codes,
        reference_features=semantic_reference_features,
        reference_mask=semantic_reference_mask,
        generator=semantic_decode_generator,
    )


def decode_generated_frame_codes(
    audio_token_ids: Tensor,
    *,
    codec: Codec,
    audio_tokenizer: AudioTokenizer,
    audio_token_range: tuple[int, int],
) -> Tensor:
    """Decode generated full frame-code tokens with a ``FrameCodec`` backend."""
    local_ids = _local_ids(audio_token_ids, audio_token_range)
    rows = [_decoded_frames(audio_tokenizer, row) for row in local_ids]
    if not rows or len({tuple(row.shape) for row in rows}) != 1:
        raise ValueError(
            "full codec token rows must expand to the same frame and codebook shape."
        )
    return codec.decode(torch.stack(rows))


def decode_generated_bicodec_full(
    audio_token_ids: Tensor,
    *,
    codec: StructuredCodec,
    audio_tokenizer: BiCodecAudioTokenizer,
    audio_token_range: tuple[int, int],
) -> Tensor:
    """Decode a structured BiCodec full sequence without collapsing its axes."""
    local_ids = _local_ids(audio_token_ids, audio_token_range)
    rows = [audio_tokenizer.decode_full(row) for row in local_ids]
    semantic_rows = [cast(Tensor, row.semantic_codes) for row in rows]
    global_rows = [cast(Tensor, row.global_codes) for row in rows]
    if not rows or len({tuple(row.shape) for row in semantic_rows}) != 1:
        raise ValueError("BiCodec semantic token rows must have the same shape.")
    if len({tuple(row.shape) for row in global_rows}) != 1:
        raise ValueError("BiCodec global token rows must have the same shape.")
    codes = SemanticAcousticCodes(
        semantic=torch.stack(semantic_rows),
        acoustic=torch.stack(global_rows),
    )
    return codec.detokenize(codes)


def decode_generated_bicodec_row(
    audio_token_ids: Tensor,
    prompt_ids: Tensor | None,
    *,
    codec: StructuredCodec,
    audio_tokenizer: BiCodecAudioTokenizer,
    audio_token_range: tuple[int, int],
    boa_token_id: int,
    eoa_token_id: int,
) -> tuple[Tensor, AudioCodes]:
    """Resolve one self-describing BiCodec response against its prompt."""
    if audio_token_ids.dim() != 1:
        raise ValueError("BiCodec decode expects one generated token row.")
    local_ids = _local_ids(audio_token_ids[None], audio_token_range)[0]
    output = audio_tokenizer.decode_streams(local_ids)
    if output.semantic_codes is None:
        raise ValueError("BiCodec generated output is missing semantic codes.")
    prompt_global = _prompt_bicodec_global(
        prompt_ids,
        audio_tokenizer=audio_tokenizer,
        audio_token_range=audio_token_range,
        boa_token_id=boa_token_id,
        eoa_token_id=eoa_token_id,
    )
    if (prompt_global is None) == (output.global_codes is None):
        raise ValueError(
            "BiCodec decode requires exactly one global stream owner across "
            "prompt and generated output."
        )
    global_codes = (
        output.global_codes if output.global_codes is not None else prompt_global
    )
    if global_codes is None:
        raise AssertionError("BiCodec global stream ownership was not resolved.")
    resolved = AudioCodes(
        semantic_codes=output.semantic_codes.to(
            device=audio_token_ids.device,
            dtype=torch.long,
        ),
        global_codes=global_codes.to(
            device=audio_token_ids.device,
            dtype=torch.long,
        ),
    )
    resolved_semantic = cast(Tensor, resolved.semantic_codes)
    waveform = codec.detokenize(
        SemanticAcousticCodes(
            semantic=resolved_semantic.unsqueeze(0),
            acoustic=cast(Tensor, resolved.global_codes).unsqueeze(0),
        )
    )
    if waveform.dim() < 1 or waveform.size(0) != 1:
        raise ValueError("BiCodec detokenize must preserve a batch size of one.")
    return waveform[0], resolved


def _prompt_bicodec_global(
    prompt_ids: Tensor | None,
    *,
    audio_tokenizer: BiCodecAudioTokenizer,
    audio_token_range: tuple[int, int],
    boa_token_id: int,
    eoa_token_id: int,
) -> Tensor | None:
    if prompt_ids is None:
        return None
    if prompt_ids.dim() != 1:
        raise ValueError("BiCodec prompt ids must have shape [tokens].")
    start, end = audio_token_range
    global_streams: list[Tensor] = []
    cursor = 0
    while cursor < prompt_ids.numel():
        starts = prompt_ids[cursor:].eq(boa_token_id).nonzero(as_tuple=False)
        if starts.numel() == 0:
            break
        span_start = cursor + int(starts[0].item()) + 1
        stops = prompt_ids[span_start:].eq(eoa_token_id).nonzero(as_tuple=False)
        if stops.numel() == 0:
            break
        span_end = span_start + int(stops[0].item())
        payload = prompt_ids[span_start:span_end]
        if payload.numel() < 1:
            raise ValueError("BiCodec prompt audio span must not be empty.")
        if bool((payload < start).any()) or bool((payload >= end).any()):
            raise ValueError("BiCodec prompt audio span contains non-codec tokens.")
        decoded = audio_tokenizer.decode_streams(payload.to(dtype=torch.long) - start)
        if decoded.global_codes is not None:
            global_streams.append(decoded.global_codes)
        cursor = span_end + 1
    if len(global_streams) > 1:
        raise ValueError("BiCodec prompt contains more than one global stream.")
    return None if not global_streams else global_streams[0]


def decode_generated_codes(
    audio_token_ids: Tensor,
    acoustic_codes: Tensor,
    *,
    codec: AcousticCodec,
    audio_tokenizer: AudioTokenizer,
    audio_token_range: tuple[int, int],
) -> Tensor:
    """Decode generated audio tokens and acoustic codes into waveforms."""
    local_ids = _local_ids(audio_token_ids, audio_token_range)
    return _decode_audio(
        local_ids,
        codec.acoustic_codes_to_features(acoustic_codes),
        codec=codec,
        audio_tokenizer=audio_tokenizer,
    )


def decode_reference_codes(
    codes: object,
    *,
    codec: CodecBackend,
) -> Tensor:
    """Decode one prepared-code sample through its actual backend capability."""
    if isinstance(codes, (Mapping, AudioCodes)):
        structured_backend = structured_codec(codec)
        structured = _structured_codes(codes, structured_backend.acoustic_layout)
        anycodec_codes = structured.to_anycodec(structured_backend.acoustic_layout)
        return structured_backend.detokenize(
            SemanticAcousticCodes(
                semantic=anycodec_codes.semantic.unsqueeze(0),
                acoustic=anycodec_codes.acoustic.unsqueeze(0),
            )
        )
    if not isinstance(codes, Tensor):
        raise TypeError("frame codec reference codes must be a Tensor.")
    _reference_code_tensor(codes)
    return frame_codec(codec).decode(codes.unsqueeze(0))


def _local_ids(audio_token_ids: Tensor, audio_token_range: tuple[int, int]) -> Tensor:
    if not isinstance(audio_token_ids, Tensor):
        raise TypeError("audio token ids must be a Tensor.")
    if not is_signed_integer_dtype(audio_token_ids.dtype):
        raise TypeError(
            "audio token ids must contain integer ids using a signed dtype."
        )
    if audio_token_ids.dim() != 2:
        raise ValueError("audio token ids must have shape [batch, tokens].")
    if audio_token_ids.size(0) < 1 or audio_token_ids.size(1) < 1:
        raise ValueError("audio token ids must contain at least one token row.")

    global_start, global_end = audio_token_range
    if bool((audio_token_ids < global_start).any()) or bool(
        (audio_token_ids >= global_end).any()
    ):
        raise ValueError("audio token ids must be codec-decodable global audio ids.")
    return audio_token_ids.to(dtype=torch.long) - global_start


def _decoded_frames(tokenizer: AudioTokenizer, token_ids: Tensor) -> Tensor:
    decoded = tokenizer.decode(token_ids)
    if isinstance(decoded, Tensor):
        frames = decoded
    else:
        frames = torch.as_tensor(decoded, device=token_ids.device)
    if frames.dim() != 2:
        raise ValueError("full codec tokens must decode to [frames, codebooks].")
    if not is_signed_integer_dtype(frames.dtype):
        raise TypeError("decoded full codec codes must use a signed integer dtype.")
    return frames.to(device=token_ids.device, dtype=torch.long)


def _structured_codes(value: object, layout: AcousticLayout) -> AudioCodes:
    if isinstance(value, AudioCodes):
        codes = value
    elif isinstance(value, Mapping):
        semantic = value.get("semantic")
        if not isinstance(semantic, Tensor):
            raise TypeError(
                "anydataset structured reference codes must contain Tensor semantic."
            )
        if layout is AcousticLayout.FIXED_LENGTH:
            global_codes = value.get("acoustic")
            if not isinstance(global_codes, Tensor):
                raise TypeError(
                    "anydataset fixed reference codes must contain Tensor acoustic."
                )
            codes = AudioCodes(
                semantic_codes=semantic,
                global_codes=global_codes,
            )
        elif layout is AcousticLayout.FRAME_ALIGNED:
            acoustic = value.get("acoustic")
            if not isinstance(acoustic, Tensor):
                raise TypeError(
                    "anydataset frame-aligned reference codes must contain Tensor acoustic."
                )
            codes = AudioCodes(
                semantic_codes=semantic,
                acoustic_codes=acoustic,
            )
        else:
            raise ValueError(f"unsupported acoustic layout: {layout!r}.")
    else:
        raise TypeError("structured reference codes require AudioCodes fields.")
    semantic = codes.semantic_codes
    if semantic is None:
        raise ValueError("structured reference codes are missing semantic codes.")
    _reference_code_tensor(semantic)
    secondary = (
        codes.global_codes
        if layout is AcousticLayout.FIXED_LENGTH
        else codes.acoustic_codes
    )
    if secondary is None:
        raise ValueError("structured reference codes are missing non-semantic codes.")
    _reference_code_tensor(secondary)
    return codes


def _reference_code_tensor(value: Tensor) -> None:
    if value.dim() != 2:
        raise ValueError("reference codes must have shape [units, codebooks].")
    if not is_signed_integer_dtype(value.dtype):
        raise TypeError("reference codes must use a signed integer dtype.")
