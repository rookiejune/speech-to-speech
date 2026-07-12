from anydataset.types import Modality
from anytrain.idspace import Layout
from torch import nn

from ..datamodule.types import ModelBatch
from ..model.protocol import FlowMatching
from .flow_matching import AcousticFlowLoss, FlowRuntime
from .semantic import SemanticLoss
from .types import Outputs


class Loss(nn.Module):
    def __init__(self, layout: Layout, flow_runtime: FlowRuntime) -> None:
        super().__init__()
        self.layout = layout
        self.semantic = SemanticLoss(layout)
        self.flow = AcousticFlowLoss()
        self.flow_runtime = flow_runtime

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
            acoustic = self.flow(
                model.acoustic_decoder,
                condition,
                model.acoustic_target_latent(batch.acoustic_labels),
                batch.acoustic_target_mask,
                self.flow_runtime,
            )
            result["flow_matching"] = acoustic
            result["loss"] = result["loss"] + acoustic.loss.mean()
        return result
