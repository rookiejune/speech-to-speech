from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from anytrain.perf import training_flops_from_forward
from lightning import pytorch as pl
from torch import Tensor

from .._flops import adapter, flow_decoder, rvq_decoder
from .._tensor import is_signed_integer_dtype
from ..model.acoustic.dit import AcousticDiT
from ..model.acoustic.flow import AcousticFlow
from ..model.acoustic.rvq import AcousticRVQDecoder
from .model import AcousticFlowScreening, AcousticRVQScreening


class TrainingFlops:
    """Estimate local-rank codec-oracle training FLOPs for one batch.

    The estimate follows the conventional model-MFU matmul boundary. It counts
    linear projections and attention matrix multiplications, then estimates
    backward as twice the forward cost. Lookup, scatter, normalization,
    activation, loss, and frozen codec dequantization operations are excluded.
    """

    def __call__(
        self,
        *,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> float:
        del trainer, outputs, batch_idx
        codes, mask = _batch(batch)
        if isinstance(pl_module, AcousticFlowScreening):
            forward = _flow(pl_module, codes, mask)
        elif isinstance(pl_module, AcousticRVQScreening):
            forward = _rvq(pl_module, codes, mask)
        else:
            raise TypeError(
                "codec oracle training FLOPs require AcousticFlowScreening "
                "or AcousticRVQScreening."
            )
        return training_flops_from_forward(float(forward), backward_multiplier=2.0)


def _batch(batch: Any) -> tuple[Tensor, Tensor]:
    if not isinstance(batch, Mapping):
        raise TypeError("codec oracle FLOPs batch must be a mapping.")
    if "codes" not in batch or "mask" not in batch:
        raise KeyError("codec oracle FLOPs batch requires codes and mask.")
    codes = batch["codes"]
    mask = batch["mask"]
    if not isinstance(codes, Tensor) or not isinstance(mask, Tensor):
        raise TypeError("codec oracle FLOPs codes and mask must be tensors.")
    if codes.dim() != 3 or mask.dim() != 2 or codes.shape[:2] != mask.shape:
        raise ValueError(
            "codec oracle FLOPs codes and mask must have shapes [B, F, Q] and [B, F]."
        )
    if not is_signed_integer_dtype(codes.dtype):
        raise TypeError("codec oracle FLOPs codes must use a signed integer dtype.")
    if mask.dtype != torch.bool:
        raise TypeError("codec oracle FLOPs mask must be boolean.")
    if codes.device != mask.device:
        raise ValueError("codec oracle FLOPs codes and mask must use the same device.")
    if codes.size(0) < 1 or codes.size(1) < 1:
        raise ValueError(
            "codec oracle FLOPs batch and frame dimensions must be positive."
        )
    if not bool(mask.any(dim=1).all()):
        raise ValueError(
            "each codec oracle FLOPs batch row must contain a valid frame."
        )
    return codes, mask


def _flow(module: AcousticFlowScreening, codes: Tensor, mask: Tensor) -> int:
    del mask
    model = module.model
    flow = model.acoustic_flow
    if not isinstance(flow, AcousticFlow) or not isinstance(flow.decoder, AcousticDiT):
        raise TypeError(
            "Flow FLOPs require the standard AcousticFlow/AcousticDiT model."
        )
    decoder = flow.decoder
    if decoder.repa_projection is not None or decoder.repa_student_layer is not None:
        raise ValueError("Flow FLOPs do not support a REPA decoder.")

    sizes = model.runtime.codec.acoustic_codebook_sizes
    _codebooks(codes, sizes)
    batch, frames = codes.shape[:2]
    rows = batch * frames
    latent = decoder.latent_dim
    condition = decoder.condition.in_features
    if model.runtime.codec.acoustic_feature_dim != latent:
        raise ValueError(
            "Flow decoder latent size does not match the codec feature size."
        )

    forward = adapter(
        model.semantic_audio_adapter,
        rows=rows,
        in_features=model.semantic_audio_embedding.embedding_dim,
        out_features=condition,
        name="semantic audio adapter",
    )
    forward += flow_decoder(decoder, batch=batch, frames=frames)
    return forward


def _rvq(module: AcousticRVQScreening, codes: Tensor, mask: Tensor) -> int:
    model = module.model
    decoder = model.acoustic_decoder
    if not isinstance(decoder, AcousticRVQDecoder):
        raise TypeError("RVQ FLOPs require the standard AcousticRVQDecoder model.")
    sizes = model.runtime.codec.acoustic_codebook_sizes
    if tuple(sizes) != tuple(decoder.codebook_sizes):
        raise ValueError("RVQ decoder codebooks do not match the runtime codec.")
    _codebooks(codes, sizes)

    batch, frames = codes.shape[:2]
    packed = int(mask.sum().item())
    condition = decoder.condition_dim
    forward = adapter(
        model.semantic_audio_adapter,
        rows=batch * frames,
        in_features=model.semantic_audio_embedding.embedding_dim,
        out_features=condition,
        name="semantic audio adapter",
    )
    forward += rvq_decoder(decoder, valid_frames=packed)
    return forward


def _codebooks(codes: Tensor, sizes: tuple[int, ...]) -> None:
    if (
        not isinstance(sizes, tuple)
        or not sizes
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in sizes
        )
    ):
        raise ValueError(
            "codec acoustic codebook sizes must be a non-empty integer tuple."
        )
    expected = len(sizes) + 1
    if codes.size(-1) != expected:
        raise ValueError(
            f"codec oracle batch has {codes.size(-1) - 1} acoustic codebooks; "
            f"the model requires {expected - 1}."
        )


__all__ = ["TrainingFlops"]
