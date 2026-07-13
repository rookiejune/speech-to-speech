from typing import TypedDict

from anydataset.types import Modality
from anytrain.idspace import Layout
from torch import nn

from ..datamodule.types import ModelBatch
from ..model.protocol import FlowMatching, RVQMatching
from .causal_lm import CausalAcousticLoss
from .flow_matching import AcousticFlowLoss, FlowRuntime
from .repa import RepaLoss, Teacher
from .semantic import SemanticLoss
from .types import Outputs


class RepaConfig(TypedDict):
    weight: float
    teacher: Teacher


class Loss(nn.Module):
    def __init__(
        self,
        layout: Layout,
        flow_runtime: FlowRuntime,
        *,
        repa: RepaConfig | None = None,
    ) -> None:
        super().__init__()
        if repa is not None and repa["weight"] <= 0:
            raise ValueError("REPA weight must be positive")
        self.layout = layout
        self.semantic = SemanticLoss(layout)
        self.flow_matching = AcousticFlowLoss()
        self.repa_loss = RepaLoss()
        self.repa_teacher = None if repa is None else repa["teacher"]
        self.flow_runtime = flow_runtime
        self.repa_weight = None if repa is None else repa["weight"]

    def forward(self, batch: ModelBatch, model: FlowMatching) -> Outputs:
        if model.layout.blocks != self.layout.blocks:
            raise ValueError("model and loss must use the same runtime layout.")
        audio_target = batch.tasks[0].target_modality is Modality.AUDIO
        output = model(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            acoustic_input_ids=batch.acoustic_input_ids,
            acoustic_input_positions=batch.acoustic_input_positions,
            acoustic_input_mask=batch.acoustic_input_mask,
            output_hidden_states=audio_target,
        )
        semantic = self.semantic(output.logits, batch.labels)
        result: Outputs = {"loss": semantic.loss.mean(), "semantic": semantic}

        if audio_target:
            if batch.acoustic_labels is None or batch.acoustic_label_positions is None:
                raise ValueError("audio-target tasks require acoustic target fields.")
            if batch.acoustic_target_mask is None:
                raise RuntimeError(
                    "model batch did not produce an acoustic target mask."
                )
            if output.hidden_states is None:
                raise RuntimeError("model did not return acoustic condition states.")
            condition = model.target_frame_condition(
                output.hidden_states[-1], batch.acoustic_label_positions
            )
            target = model.acoustic_target_latent(batch.acoustic_labels)
            if self.repa_weight is None:
                acoustic = self.flow_matching(
                    model.acoustic_decoder,
                    condition,
                    target,
                    batch.acoustic_target_mask,
                    self.flow_runtime,
                )
            else:
                if self.repa_teacher is None or batch.semantic_frame_labels is None:
                    raise RuntimeError(
                        "REPA requires a teacher and semantic frame labels"
                    )
                acoustic, representation = self.flow_matching.forward_with_features(
                    model.acoustic_decoder,
                    condition,
                    target,
                    batch.acoustic_target_mask,
                    self.flow_runtime,
                )
                teacher = self.repa_teacher(
                    batch.semantic_frame_labels,
                    batch.acoustic_labels,
                    batch.acoustic_target_mask,
                )
                repa = self.repa_loss(
                    representation,
                    teacher,
                    batch.acoustic_target_mask,
                )
                result["repa"] = repa
                result["loss"] = result["loss"] + self.repa_weight * repa.loss.mean()
            result["flow_matching"] = acoustic
            result["loss"] = result["loss"] + acoustic.loss.mean()
        return result


class RVQLoss(nn.Module):
    def __init__(self, layout: Layout) -> None:
        super().__init__()
        self.layout = layout
        self.semantic = SemanticLoss(layout)
        self.causal_lm = CausalAcousticLoss()

    def forward(self, batch: ModelBatch, model: RVQMatching) -> Outputs:
        if model.layout.blocks != self.layout.blocks:
            raise ValueError("model and loss must use the same runtime layout.")
        audio_target = batch.tasks[0].target_modality is Modality.AUDIO
        output = model(
            batch.input_ids,
            attention_mask=batch.attention_mask,
            acoustic_input_ids=batch.acoustic_input_ids,
            acoustic_input_positions=batch.acoustic_input_positions,
            acoustic_input_mask=batch.acoustic_input_mask,
            output_hidden_states=audio_target,
        )
        semantic = self.semantic(output.logits, batch.labels)
        result: Outputs = {"loss": semantic.loss.mean(), "semantic": semantic}

        if audio_target:
            if batch.acoustic_labels is None or batch.acoustic_label_positions is None:
                raise ValueError("audio-target tasks require acoustic target fields.")
            if batch.acoustic_target_mask is None:
                raise RuntimeError("model batch did not produce an acoustic target mask.")
            if output.hidden_states is None:
                raise RuntimeError("model did not return acoustic condition states.")
            labels = batch.acoustic_labels
            logits = model.acoustic_logits(
                output.hidden_states[-1],
                batch.acoustic_label_positions,
                labels.masked_fill(~batch.acoustic_target_mask[..., None], 0),
            )
            acoustic = self.causal_lm(
                logits,
                labels,
                batch.acoustic_target_mask,
            )
            result["causal_lm"] = acoustic
            result["loss"] = result["loss"] + acoustic.loss.mean()
        return result
