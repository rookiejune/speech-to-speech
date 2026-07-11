from __future__ import annotations

import torch
from anydataset.types import Modality
from torch import Tensor

from ..datamodule.types import ModelBatch, Task
from ..model.acoustic import SpeechToSpeechFlowModel
from .decode import decode_generated_audio


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
