from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, device as TorchDevice

from ..types.datamodule import LongCatBatchSide, LongCatSide


def collate_longcat_sides(
    sides: Sequence[LongCatSide | None],
    *,
    device: TorchDevice,
) -> LongCatBatchSide | None:
    present = [side for side in sides if side is not None]
    if not present:
        return None

    semantic_rows = [_semantic_row(side.semantic_ids) for side in present]
    acoustic_rows = [_acoustic_row(side.acoustic_ids) for side in present]
    for semantic, acoustic in zip(semantic_rows, acoustic_rows, strict=True):
        if semantic.numel() != acoustic.size(-1):
            raise ValueError("LongCat semantic and acoustic lengths must match.")

    max_semantic_length = max(row.numel() for row in semantic_rows)
    max_acoustic_length = max(row.size(-1) for row in acoustic_rows)
    codebook_count = acoustic_rows[0].size(0)
    if any(row.size(0) != codebook_count for row in acoustic_rows):
        raise ValueError("LongCat acoustic codebook count must be consistent within a batch.")

    semantic_ids = torch.zeros(
        (len(sides), max_semantic_length),
        dtype=torch.long,
        device=device,
    )
    semantic_mask = torch.zeros(
        (len(sides), max_semantic_length),
        dtype=torch.bool,
        device=device,
    )
    acoustic_ids = torch.zeros(
        (len(sides), codebook_count, max_acoustic_length),
        dtype=torch.long,
        device=device,
    )
    acoustic_mask = torch.zeros(
        (len(sides), max_acoustic_length),
        dtype=torch.bool,
        device=device,
    )

    present_index = 0
    for row_index, side in enumerate(sides):
        if side is None:
            continue
        semantic = semantic_rows[present_index].to(device=device)
        acoustic = acoustic_rows[present_index].to(device=device)
        present_index += 1
        semantic_ids[row_index, : semantic.numel()] = semantic
        semantic_mask[row_index, : semantic.numel()] = True
        acoustic_ids[row_index, :, : acoustic.size(-1)] = acoustic
        acoustic_mask[row_index, : acoustic.size(-1)] = True

    return LongCatBatchSide(
        semantic_ids=semantic_ids,
        semantic_mask=semantic_mask,
        acoustic_ids=acoustic_ids,
        acoustic_mask=acoustic_mask,
    )


def _semantic_row(ids: Tensor) -> Tensor:
    if ids.dim() == 0:
        raise ValueError("LongCat semantic ids must have a time dimension.")
    return ids.reshape(-1).detach().to(dtype=torch.long)


def _acoustic_row(ids: Tensor) -> Tensor:
    if ids.dim() == 3 and ids.size(0) == 1:
        ids = ids.squeeze(0)
    if ids.dim() != 2:
        raise ValueError("LongCat acoustic ids must have shape [nq, time].")
    return ids.detach().to(dtype=torch.long)
