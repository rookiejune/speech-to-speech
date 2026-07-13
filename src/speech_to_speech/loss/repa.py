from __future__ import annotations

from typing import Protocol, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from transformers import WavLMModel

from ..runtime.types import Codec
from .types import LossItem


class Teacher(Protocol):
    @property
    def feature_dim(self) -> int: ...

    def __call__(
        self,
        semantic_ids: Tensor,
        acoustic_ids: Tensor,
        mask: Tensor,
    ) -> Tensor: ...


class WavLMTeacher(nn.Module):
    """Frozen online WavLM teacher over codec-decoded target waveforms."""

    def __init__(
        self,
        codec: Codec,
        *,
        checkpoint: str = "microsoft/wavlm-base",
        layer: int = 9,
        sample_rate: int = 16_000,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.codec = codec
        self.layer = layer
        self.sample_rate = sample_rate
        self.model = WavLMModel.from_pretrained(checkpoint)
        if not 0 <= layer <= self.model.config.num_hidden_layers:
            raise ValueError("WavLM teacher layer is outside hidden_states")
        self.model.requires_grad_(False)
        self.model.eval()
        if device is not None:
            self.model.to(device)

    @property
    def feature_dim(self) -> int:
        return self.model.config.hidden_size

    def train(self, mode: bool = True) -> WavLMTeacher:
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(
        self,
        semantic_ids: Tensor,
        acoustic_ids: Tensor,
        mask: Tensor,
    ) -> Tensor:
        if semantic_ids.shape[:2] != acoustic_ids.shape[:2]:
            raise ValueError("teacher semantic and acoustic codes must align")
        if mask.shape != semantic_ids.shape[:2]:
            raise ValueError("teacher mask must align with codec frames")

        waveforms = [
            self._waveform(
                torch.cat((semantic_ids[row, valid], acoustic_ids[row, valid]), dim=-1)
            )
            for row, valid in enumerate(mask)
        ]
        lengths = torch.tensor(
            [waveform.numel() for waveform in waveforms],
            device=self._device,
        )
        inputs = pad_sequence(waveforms, batch_first=True).to(self._device)
        sample_mask = (
            torch.arange(inputs.size(1), device=self._device)[None] < lengths[:, None]
        )
        output = self.model(
            inputs,
            attention_mask=sample_mask,
            output_hidden_states=True,
        )
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("WavLM did not return hidden states")
        features = hidden_states[self.layer]
        feature_lengths = self._feature_lengths(lengths)
        aligned = features.new_zeros(mask.shape + (self.feature_dim,))
        for row, (feature_length, frame_count) in enumerate(
            zip(feature_lengths.tolist(), mask.sum(dim=1).tolist())
        ):
            source = features[row, :feature_length].transpose(0, 1)[None]
            value = F.interpolate(
                source,
                size=frame_count,
                mode="linear",
                align_corners=False,
            )[0].transpose(0, 1)
            aligned[row, :frame_count] = value
        return aligned

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _waveform(self, codes: Tensor) -> Tensor:
        waveform = self.codec.decode(codes[None]).float()
        while waveform.dim() > 1 and waveform.size(0) == 1:
            waveform = waveform.squeeze(0)
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if waveform.dim() != 1:
            raise ValueError("codec teacher decode must produce mono waveform")
        if self.codec.sample_rate != self.sample_rate:
            from torchaudio.functional import resample

            waveform = resample(waveform, self.codec.sample_rate, self.sample_rate)
        return (waveform - waveform.mean()) / torch.sqrt(
            waveform.var(unbiased=False) + 1e-7
        )

    def _feature_lengths(self, lengths: Tensor) -> Tensor:
        output = lengths
        kernels = cast(list[int], self.model.config.conv_kernel)
        strides = cast(list[int], self.model.config.conv_stride)
        for kernel, stride in zip(kernels, strides):
            output = torch.div(output - kernel, stride, rounding_mode="floor") + 1
        return output


class RepaLoss(nn.Module):
    """Align a selected DiT layer to detached teacher frame features."""

    def forward(
        self,
        representation: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> LossItem:
        if representation.shape != target.shape:
            raise ValueError("REPA representation and teacher shapes must match")
        if mask.shape != target.shape[:2]:
            raise ValueError("REPA mask must align with teacher frames")
        prediction = F.normalize(representation.float(), dim=-1)
        teacher = F.normalize(target.detach().to(representation.device).float(), dim=-1)
        frame_loss = 1 - (prediction * teacher).sum(dim=-1)
        weights = mask.to(dtype=frame_loss.dtype)
        frame_count = weights.sum(dim=1)
        loss = (frame_loss * weights).sum(dim=1) / frame_count.clamp_min(1)
        return LossItem(loss=loss, details={"cosine": 1 - loss})
