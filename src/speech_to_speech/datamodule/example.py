from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from anydataset import AudioView, Batch, FieldGroup, FieldRef, Modality, Role, Sample

from ..types.datamodule import (
    AutoregressionExample,
    LongCatBPETokenizer,
    LongCatPair,
    LongCatSide,
    SpeechPair,
    TranslationExample,
)


def autoregression_example_from_sample(
    sample: Sample,
    role: Role,
) -> AutoregressionExample:
    return AutoregressionExample(audio_ids=_role_ids(sample, role))


def translation_example_from_sample(sample: Sample) -> TranslationExample:
    source_ids, target_ids = _source_target_ids(sample)
    return TranslationExample(
        source_ids=source_ids,
        target_ids=target_ids,
    )


def speech_pair_from_sample(sample: Sample) -> SpeechPair:
    source_ids, target_ids = _source_target_ids(sample)
    return SpeechPair(
        source_ids=source_ids,
        target_ids=target_ids,
    )


def longcat_pair_from_sample(sample: Sample) -> LongCatPair:
    return LongCatPair(
        source=_longcat_side(sample, Role.SOURCE),
        target=_longcat_side(sample, Role.TARGET),
    )


def autoregression_examples_from_batch(
    batch: Batch,
    role: Role,
) -> list[AutoregressionExample]:
    return [
        AutoregressionExample(audio_ids=audio_ids)
        for audio_ids in _role_id_rows(batch, role)
    ]


def translation_examples_from_batch(batch: Batch) -> list[TranslationExample]:
    source_rows = _role_id_rows(batch, Role.SOURCE)
    target_rows = _role_id_rows(batch, Role.TARGET)
    _check_row_count(source_rows, target_rows)
    return [
        TranslationExample(source_ids=source_ids, target_ids=target_ids)
        for source_ids, target_ids in zip(source_rows, target_rows, strict=True)
    ]


def speech_pairs_from_batch(batch: Batch) -> list[SpeechPair]:
    source_rows = _role_id_rows(batch, Role.SOURCE)
    target_rows = _role_id_rows(batch, Role.TARGET)
    _check_row_count(source_rows, target_rows)
    return [
        SpeechPair(source_ids=source_ids, target_ids=target_ids)
        for source_ids, target_ids in zip(source_rows, target_rows, strict=True)
    ]


def encode_autoregression_example(
    example: AutoregressionExample,
    tokenizer: LongCatBPETokenizer,
    *,
    device: torch.device | str | None = None,
) -> AutoregressionExample:
    audio_ids, audio_weights = _encode_ids_and_weights(
        example.audio_ids,
        tokenizer,
        device=device,
    )
    return AutoregressionExample(
        audio_ids=audio_ids,
        audio_weights=audio_weights,
    )


def encode_translation_example(
    example: TranslationExample,
    tokenizer: LongCatBPETokenizer,
    *,
    device: torch.device | str | None = None,
) -> TranslationExample:
    source_ids, _ = _encode_ids_and_weights(
        example.source_ids,
        tokenizer,
        device=device,
    )
    target_ids, target_weights = _encode_ids_and_weights(
        example.target_ids,
        tokenizer,
        device=device,
    )
    return TranslationExample(
        source_ids=source_ids,
        target_ids=target_ids,
        target_weights=target_weights,
    )


def _source_target_ids(sample: Sample) -> tuple[Tensor, Tensor]:
    return (
        _role_ids(sample, Role.SOURCE),
        _role_ids(sample, Role.TARGET),
    )


def _role_ids(sample: Sample, role: Role) -> Tensor:
    return _semantic_ids(sample[role, Modality.AUDIO].views[AudioView.LONGCAT])


def _longcat_side(sample: Sample, role: Role) -> LongCatSide:
    view = sample[role, Modality.AUDIO].views[AudioView.LONGCAT]
    return LongCatSide(
        semantic_ids=_semantic_ids(view),
        acoustic_ids=_acoustic_ids(view),
    )


def _role_id_rows(batch: Batch, role: Role) -> list[Tensor]:
    ref = (role, Modality.AUDIO)
    view = batch.sample[ref].views[AudioView.LONGCAT]
    mask = batch.masks.get(
        FieldRef(
            ref=ref,
            group=FieldGroup.VIEWS,
            key=AudioView.LONGCAT,
        )
    )
    return _semantic_id_rows(view, mask)


def _semantic_id_rows(view: object, mask: Tensor | None) -> list[Tensor]:
    if isinstance(view, Tensor):
        return _tensor_rows(view, mask)
    if isinstance(view, Mapping):
        return _tensor_rows(_semantic_ids(view), mask)
    if isinstance(view, Sequence) and not isinstance(view, str | bytes):
        if mask is not None:
            raise TypeError(
                "LongCat non-tensor batch view must not have a tensor mask."
            )
        return [_semantic_ids(value) for value in view]
    return [_semantic_ids(view)]


def _tensor_rows(value: Tensor, mask: Tensor | None) -> list[Tensor]:
    if mask is None:
        if value.dim() == 0:
            raise ValueError("LongCat batch tensor must have a batch dimension.")
        return [row for row in value]
    if value.shape != mask.shape:
        raise ValueError("LongCat batch tensor and mask must have the same shape.")
    if value.dim() == 0:
        raise ValueError("LongCat batch tensor must have a batch dimension.")
    return [_trim_row(row, row_mask) for row, row_mask in zip(value, mask, strict=True)]


def _trim_row(row: Tensor, mask: Tensor) -> Tensor:
    if row.shape != mask.shape:
        raise ValueError("LongCat row tensor and mask must have the same shape.")
    if row.dim() == 0:
        return row
    if row.dim() == 1:
        return row[mask.to(dtype=torch.bool)]

    dims = tuple(range(mask.dim() - 1))
    time_mask = mask.to(dtype=torch.bool).any(dim=dims)
    return row[..., time_mask]


def _semantic_ids(view: object) -> Tensor:
    if isinstance(view, Tensor):
        return view
    if not isinstance(view, Mapping):
        raise TypeError("LongCat audio view must be a Tensor or mapping.")

    value = view["semantic_codes"]
    if not isinstance(value, Tensor):
        raise TypeError("LongCat semantic_codes must be a Tensor.")
    return value


def _acoustic_ids(view: object) -> Tensor:
    if not isinstance(view, Mapping):
        raise TypeError("LongCat audio view must be a mapping with acoustic_codes.")

    value = view["acoustic_codes"]
    if not isinstance(value, Tensor):
        raise TypeError("LongCat acoustic_codes must be a Tensor.")
    return value


def _encode_ids_and_weights(
    ids: Tensor,
    tokenizer: LongCatBPETokenizer,
    *,
    device: torch.device | str | None,
) -> tuple[Tensor, Tensor]:
    frames = [[int(value)] for value in ids.reshape(-1).detach().cpu().tolist()]
    encoded = tokenizer.encode_frames(frames)
    token_ids = torch.tensor(encoded, dtype=torch.long, device=device)
    return token_ids, _expanded_lengths(encoded, tokenizer, device=device)


def _expanded_lengths(
    encoded: Sequence[int],
    tokenizer: LongCatBPETokenizer,
    *,
    device: torch.device | str | None,
) -> Tensor:
    expanded = tokenizer.expand_ids([int(token_id) for token_id in encoded])
    lengths = [len(span) for span in expanded]
    if len(lengths) != len(encoded):
        raise ValueError("LongCat BPE expand_ids() must return one span per token.")
    if any(length <= 0 for length in lengths):
        raise ValueError("LongCat BPE expanded spans must be non-empty.")
    return torch.tensor(lengths, dtype=torch.float, device=device)


def _check_row_count(
    source_rows: Sequence[Tensor], target_rows: Sequence[Tensor]
) -> None:
    if len(source_rows) != len(target_rows):
        raise ValueError("source and target batch rows must have the same length.")
