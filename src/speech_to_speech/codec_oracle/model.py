from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from anytrain.framework.flow_matching import ContinuousFlowRuntime
from lightning import pytorch as pl
from torch import Tensor, nn

from ..loss.flow_matching import AcousticFlowLoss
from ..model import SpeechToSpeechFlowModel
from .trace import timed
from .types import Initialization


class AcousticFlowScreening(pl.LightningModule):
    def __init__(
        self,
        model: SpeechToSpeechFlowModel,
        *,
        initialization: Initialization,
        seed: int,
        flow_runtime: ContinuousFlowRuntime,
        learning_rate: float,
        weight_decay: float,
        target_mean: Tensor,
        target_std: Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.model.semantic_audio_embedding.requires_grad_(True)
        self.model.semantic_audio_adapter.requires_grad_(True)
        self.model.acoustic_flow.requires_grad_(True)
        weight = initialization.weight(
            self.model.semantic_audio_embedding.weight.detach(),
            seed=seed,
        )
        with torch.no_grad():
            self.model.semantic_audio_embedding.weight.copy_(weight)
        self.flow_runtime = flow_runtime
        self.objective = AcousticFlowLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.target_mean = nn.Buffer(target_mean)
        self.target_std = nn.Buffer(target_std)
        self._logged_dequantize = False

    def condition(self, semantic_codes: Tensor) -> Tensor:
        start, _ = self.model.layout.blocks["audio"]
        token_labels = semantic_codes + start
        positions = torch.arange(
            semantic_codes.size(1),
            device=semantic_codes.device,
        ).expand_as(semantic_codes)
        return self.model.target_frame_label_condition(token_labels, positions)

    def features(self, acoustic_codes: Tensor) -> Tensor:
        return self.model.acoustic_target_latent(acoustic_codes).float()

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
            self.model.acoustic_decoder,
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
            [
                *self.model.semantic_audio_embedding.parameters(),
                *self.model.semantic_audio_adapter.parameters(),
                *self.model.acoustic_flow.parameters(),
            ],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @torch.no_grad()
    def sample(self, semantic_codes: Tensor, *, seed: int) -> Tensor:
        condition = self.condition(semantic_codes)
        generator = torch.Generator(device=condition.device).manual_seed(seed)
        normalized = self.model.acoustic_flow.sample(condition, generator=generator)
        return normalized * self.target_std + self.target_mean
