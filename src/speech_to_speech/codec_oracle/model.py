from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from anytrain.framework.flow_matching import ContinuousFlowRuntime
from lightning import pytorch as pl
from torch import Tensor, nn

from ..loss.flow_matching import AcousticFlowLoss
from ..model import AcousticFlowDecoder
from .trace import timed
from .types import Initialization, matched_random_weight


class FlowOracle(pl.LightningModule):
    def __init__(
        self,
        codebook: Tensor,
        feature_dim: int,
        *,
        initialization: Initialization,
        seed: int,
        dequantize: Callable[[Tensor], Tensor],
        flow_runtime: ContinuousFlowRuntime,
        learning_rate: float,
        weight_decay: float,
        target_mean: Tensor,
        target_std: Tensor,
    ) -> None:
        super().__init__()
        weight = embedding_weight(codebook, initialization, seed=seed)
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)
        self.decoder = AcousticFlowDecoder(weight.size(-1), feature_dim)
        self.flow_runtime = flow_runtime
        self.objective = AcousticFlowLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.target_mean = nn.Buffer(target_mean)
        self.target_std = nn.Buffer(target_std)
        self.dequantize = dequantize
        self._logged_dequantize = False

    def condition(self, semantic_codes: Tensor) -> Tensor:
        return self.embedding(semantic_codes)

    def features(self, acoustic_codes: Tensor) -> Tensor:
        return self.dequantize(acoustic_codes).float()

    def normalized_features(self, acoustic_codes: Tensor) -> Tensor:
        return (self.features(acoustic_codes) - self.target_mean) / self.target_std

    def training_step(
        self,
        batch: Mapping[str, Tensor],
        batch_idx: int,
    ) -> dict[str, Any]:
        del batch_idx
        codes = batch["codes"]
        mask = batch["mask"]
        safe_codes = codes.masked_fill(~mask[..., None], 0)
        semantic_codes = safe_codes[..., 0]
        acoustic_codes = safe_codes[..., 1:]
        condition = self.condition(semantic_codes)
        if not self._logged_dequantize:
            with timed(
                "train.first_dequantize",
                code_shape=list(acoustic_codes.shape),
            ):
                target = self.normalized_features(acoustic_codes)
            self._logged_dequantize = True
        else:
            target = self.normalized_features(acoustic_codes)
        target = target.masked_fill(~mask[..., None], 0)
        item = self.objective(
            self.decoder,
            condition,
            target,
            mask,
            self.flow_runtime,
        )
        loss = item.loss.mean()
        self.log("train/flow_loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log("train/batch_size", float(codes.size(0)), on_step=True, sync_dist=True)
        self.log("train/valid_frames", mask.sum().float(), on_step=True, sync_dist=True)
        return {"loss": loss, "flow_matching": item}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [*self.embedding.parameters(), *self.decoder.parameters()],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @torch.no_grad()
    def sample(self, semantic_codes: Tensor, *, seed: int) -> Tensor:
        condition = self.condition(semantic_codes)
        generator = torch.Generator(device=condition.device).manual_seed(seed)
        noise = torch.randn(
            (*condition.shape[:2], self.decoder.latent_dim),
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        normalized = self.flow_runtime.sample(
            self.decoder,
            noise,
            condition=condition,
        ).final
        return normalized * self.target_std + self.target_mean


class TokenOracle(pl.LightningModule):
    def __init__(
        self,
        codebook: Tensor,
        max_length: int,
        *,
        initialization: Initialization,
        seed: int,
        layers: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        weight = embedding_weight(codebook, initialization, seed=seed)
        special = matched_random_weight(
            weight,
            seed=seed + 1,
            rows=1,
        )
        self.vocab_size = weight.size(0)
        self.bos_id = self.vocab_size
        self.embedding = nn.Embedding.from_pretrained(
            torch.cat((weight, special), dim=0),
            freeze=False,
        )
        self.position = nn.Embedding(max_length, weight.size(-1))
        layer = nn.TransformerEncoderLayer(
            d_model=weight.size(-1),
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(weight.size(-1), self.vocab_size, bias=False)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, codes: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        if frame_mask is None:
            frame_mask = torch.ones_like(codes, dtype=torch.bool)
        safe_codes = codes.masked_fill(~frame_mask, 0)
        inputs = torch.cat(
            (
                codes.new_full((codes.size(0), 1), self.bos_id),
                safe_codes[:, :-1],
            ),
            dim=1,
        )
        positions = torch.arange(codes.size(1), device=codes.device)
        hidden = self.embedding(inputs) + self.position(positions)[None]
        causal_mask = torch.ones(
            (codes.size(1), codes.size(1)),
            dtype=torch.bool,
            device=codes.device,
        ).triu(diagonal=1)
        return self.head(
            self.backbone(
                hidden,
                mask=causal_mask,
                src_key_padding_mask=~frame_mask,
                is_causal=True,
            )
        )

    def training_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        codes = batch["codes"]
        frame_mask = batch["mask"]
        logits = self(codes, frame_mask)
        labels = codes.masked_fill(~frame_mask, -100)
        loss = F.cross_entropy(logits.transpose(1, 2), labels)
        accuracy = (
            logits.argmax(dim=-1).eq(codes).masked_select(frame_mask).float().mean()
        )
        self.log("train/token_loss", loss, on_step=True, prog_bar=True, sync_dist=True)
        self.log(
            "train/token_accuracy",
            accuracy,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("train/batch_size", float(codes.size(0)), on_step=True, sync_dist=True)
        self.log(
            "train/valid_frames",
            frame_mask.sum().float(),
            on_step=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @torch.no_grad()
    def teacher_forced_ids(self, codes: Tensor) -> Tensor:
        return self(codes).argmax(dim=-1)


def embedding_weight(
    codebook: Tensor,
    initialization: Initialization,
    *,
    seed: int,
) -> Tensor:
    if codebook.dim() != 2 or not torch.is_floating_point(codebook):
        raise ValueError(
            "codec codebook must have shape [vocab, dim] and floating dtype."
        )
    return initialization.weight(codebook, seed=seed)


def feature_stats(target: Tensor, *, enabled: bool) -> tuple[Tensor, Tensor]:
    if not enabled:
        return target.new_zeros((1, 1, target.size(-1))), target.new_ones(
            (1, 1, target.size(-1))
        )
    mean = target.mean(dim=(0, 1), keepdim=True)
    std = target.std(dim=(0, 1), correction=0, keepdim=True).clamp_min(1e-5)
    return mean, std
