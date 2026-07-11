from __future__ import annotations

from dataclasses import dataclass

import torch
from anydataset.types import Modality
from anytrain.idspace import Layout
from anytrain.optim.llm import create_optimizer
from lightning.pytorch import LightningModule
from torch import Tensor

from ..datamodule.types import ModelBatch, Task
from ..loss.module import Loss
from ..loss.types import Outputs
from ..model.acoustic import SpeechToSpeechFlowModel
from ..runtime.audio_tokenizer import semantic_ids_from_audio_tokens
from ..runtime.types import AudioTokenizer, Codec


@torch.no_grad()
def generate_batch(
    batch: ModelBatch,
    model: SpeechToSpeechFlowModel,
    *,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[Tensor]:
    """Generate one variable-length semantic response for every batch row."""
    audio_tasks = {Task.AUDIO_AR, Task.S2ST, Task.T2ST, Task.TTS}
    results: list[Tensor] = []
    for index, task in enumerate(batch.tasks):
        sequence = batch.input_ids[index]
        prompt_end = int((batch.labels[index] != -100).nonzero()[0].item())
        prompt = sequence[:prompt_end]

        acoustic_ids = None
        acoustic_positions = None
        acoustic_mask = None
        if batch.acoustic_input_ids is not None:
            acoustic_ids = batch.acoustic_input_ids[index : index + 1]
            acoustic_positions = batch.acoustic_input_positions[index : index + 1]
            acoustic_mask = batch.acoustic_input_mask[index : index + 1]

        modality = Modality.AUDIO if task in audio_tasks else Modality.TEXT
        results.append(
            model.generate_semantic(
                prompt[None],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                acoustic_input_ids=acoustic_ids,
                acoustic_input_positions=acoustic_positions,
                acoustic_input_mask=acoustic_mask,
                stop_token_id=(
                    model.runtime.eoa_token_id
                    if modality == Modality.AUDIO
                    else model.runtime.eos_token_id
                ),
                token_range=model.runtime.layout.blocks[
                    "audio" if modality == Modality.AUDIO else "text"
                ],
            )[0]
        )
    return results


def decode_generated_audio(
    audio_token_ids: Tensor,
    acoustic_features: Tensor | None = None,
    *,
    acoustic_codes: Tensor | None = None,
    codec: Codec,
    audio_tokenizer: AudioTokenizer,
    layout: Layout,
) -> Tensor:
    """Decode generated audio tokens and acoustic output into waveforms.

    FM models provide ``acoustic_features`` directly. RVQ models provide
    ``acoustic_codes`` that are dequantized by the codec before decoding.
    """
    if (acoustic_features is None) == (acoustic_codes is None):
        raise ValueError("provide exactly one of acoustic_features or acoustic_codes.")
    if acoustic_codes is not None:
        acoustic_features = codec.acoustic_codes_to_features(acoustic_codes)
    if acoustic_features is None:
        raise RuntimeError("acoustic features were not created.")

    local_start, local_end = layout.blocks["audio"]
    if bool((audio_token_ids < local_start).any()) or bool(
        (audio_token_ids >= local_end).any()
    ):
        raise ValueError(
            "audio token ids must be global ids from the audio layout block."
        )
    local_ids = audio_token_ids - local_start
    rows: list[Tensor] = []
    for row in local_ids:
        rows.append(semantic_ids_from_audio_tokens(audio_tokenizer, row))
    if not rows or len({tuple(row.shape) for row in rows}) != 1:
        raise ValueError(
            "audio token rows must expand to the same frame and codebook shape."
        )
    semantic_ids = torch.stack(rows)
    if semantic_ids.shape[:2] != acoustic_features.shape[:2]:
        raise ValueError(
            "semantic ids and acoustic features must align on [batch, frame]."
        )
    return codec.decode_features(semantic_ids, acoustic_features)


@torch.no_grad()
def generate_waveforms(
    batch: ModelBatch,
    model: SpeechToSpeechFlowModel,
    *,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[Tensor]:
    """Generate and decode one waveform per audio-target batch row."""
    audio_tasks = {Task.AUDIO_AR, Task.S2ST, Task.T2ST, Task.TTS}
    start, end = model.runtime.layout.blocks["audio"]
    waveforms: list[Tensor] = []
    for index, task in enumerate(batch.tasks):
        if task not in audio_tasks:
            raise ValueError("waveform generation requires an audio-target task.")
        labels = batch.labels[index]
        label_audio = (labels >= start) & (labels < end)
        acoustic_labels = batch.acoustic_labels
        label_positions = batch.acoustic_label_positions
        if (
            acoustic_labels is not None
            and label_positions is not None
            and bool(label_audio.any())
        ):
            row_input = batch.input_ids[index : index + 1]
            output = model(
                row_input,
                attention_mask=batch.attention_mask[index : index + 1],
                acoustic_input_ids=None
                if batch.acoustic_input_ids is None
                else batch.acoustic_input_ids[index : index + 1],
                acoustic_input_positions=None
                if batch.acoustic_input_positions is None
                else batch.acoustic_input_positions[index : index + 1],
                acoustic_input_mask=None
                if batch.acoustic_input_mask is None
                else batch.acoustic_input_mask[index : index + 1],
                output_hidden_states=True,
            )
            if output.hidden_states is None:
                raise RuntimeError("model did not return hidden states.")
            condition = model.target_frame_condition(
                output.hidden_states[-1], label_positions[index : index + 1] - 1
            )
            features = model.sample_acoustic(condition)[0]
            frame_mask = batch.acoustic_target_mask
            if frame_mask is None:
                raise RuntimeError("acoustic target mask is required with labels.")
            features = features[frame_mask[index]]
            semantic = labels[label_audio][None]
        else:
            prompt_end = int((labels != -100).nonzero()[0].item())
            prompt = batch.input_ids[index : index + 1, :prompt_end]
            acoustic_ids = (
                None
                if batch.acoustic_input_ids is None
                else batch.acoustic_input_ids[index : index + 1]
            )
            acoustic_positions = (
                None
                if batch.acoustic_input_positions is None
                else batch.acoustic_input_positions[index : index + 1]
            )
            acoustic_mask = (
                None
                if batch.acoustic_input_mask is None
                else batch.acoustic_input_mask[index : index + 1]
            )
            generated, features, _ = model.generate_audio(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                acoustic_input_ids=acoustic_ids,
                acoustic_input_positions=acoustic_positions,
                acoustic_input_mask=acoustic_mask,
            )
            semantic = generated[:, prompt.size(1) :]
            semantic = semantic[semantic != model.runtime.eoa_token_id].reshape(1, -1)
            features = features[0]
        waveforms.append(
            decode_generated_audio(
                semantic,
                features[None],
                codec=model.runtime.codec,
                audio_tokenizer=model.runtime.audio_tokenizer,
                layout=model.runtime.layout,
            )[0]
        )
    return waveforms


@dataclass(frozen=True)
class Config:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01


class SpeechToSpeech(LightningModule):
    def __init__(
        self,
        config: Config,
        *,
        model: SpeechToSpeechFlowModel,
        loss: Loss,
    ) -> None:
        super().__init__()

        self.config = config

        self.model = model
        self.loss = loss
        self._current_loss_outputs: Outputs | None = None

    def training_step(self, batch: ModelBatch, batch_idx: int = 0):
        del batch_idx
        outputs = self.loss.forward(batch, self.model)
        self._current_loss_outputs = outputs
        self.log("train/loss", outputs["loss"], prog_bar=True, on_step=True)
        return outputs

    def current_loss_outputs(self) -> Outputs:
        """Return loss outputs kept alive until the backward pass completes."""
        if self._current_loss_outputs is None:
            raise RuntimeError("loss outputs are unavailable outside a training step")
        return self._current_loss_outputs

    def on_after_backward(self) -> None:
        self._current_loss_outputs = None

    @torch.no_grad()
    def generate_batch(
        self,
        batch: ModelBatch,
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[torch.Tensor]:
        """Generate responses while preserving variable prompt lengths."""
        was_training = self.training
        self.eval()
        try:
            return generate_batch(
                batch,
                self.model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def generate_waveforms(
        self,
        batch: ModelBatch,
        *,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> list[torch.Tensor]:
        was_training = self.training
        self.eval()
        try:
            return generate_waveforms(
                batch,
                self.model,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        finally:
            self.train(was_training)

    def configure_optimizers(self):
        return create_optimizer(
            self.model,
            preset="sft",
            optimizer="adamw",
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
