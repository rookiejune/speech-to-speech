from dataclasses import dataclass

from torch import nn

from ..datamodule.types import ModelBatch
from ..model.acoustic import SpeechToSpeechFlowModel
from ..runtime import runtime
from .flow_matching import AcousticFlowLoss
from .semantic import SemanticLoss
from .types import Outputs


@dataclass
class Config:
    semantic: bool = True
    flow_matching: bool = False
    acoustic_oracle: bool = False
    autoregression: bool = False
    repa: bool = False


class Loss(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.config = config
        self.semantic = SemanticLoss(runtime().layout) if config.semantic else None
        self.flow = AcousticFlowLoss()

    def forward(self, batch: ModelBatch, model: SpeechToSpeechFlowModel) -> Outputs:
        if self.config.acoustic_oracle:
            if self.config.semantic:
                raise ValueError(
                    "acoustic_oracle requires semantic=False to bypass the backbone."
                )
            if not self.config.flow_matching:
                raise ValueError("acoustic_oracle requires flow_matching=True.")
            output = None
            result: Outputs = {"loss": batch.input_ids.new_zeros(())}
        else:
            if self.semantic is None:
                raise NotImplementedError("P1 requires the semantic objective.")
            output = model(
                batch.input_ids,
                attention_mask=batch.attention_mask,
                acoustic_input_ids=batch.acoustic_input_ids,
                acoustic_input_positions=batch.acoustic_input_positions,
                acoustic_input_mask=batch.acoustic_input_mask,
                output_hidden_states=self.config.flow_matching
                and batch.acoustic_labels is not None,
            )
            semantic = self.semantic(output.logits, batch.labels)
            result = {"loss": semantic.loss.mean(), "semantic": semantic}

        total = result["loss"]
        if self.config.flow_matching:
            if batch.acoustic_labels is None or batch.acoustic_label_positions is None:
                raise ValueError("flow matching requires acoustic target fields.")
            if batch.acoustic_target_mask is None:
                raise RuntimeError(
                    "model batch did not produce an acoustic target mask."
                )
            if self.config.acoustic_oracle:
                condition = model.target_frame_label_condition(
                    batch.labels, batch.acoustic_label_positions
                )
            else:
                if output is None or output.hidden_states is None:
                    raise RuntimeError(
                        "model does not provide an acoustic flow decoder."
                    )
                condition = model.target_frame_condition(
                    output.hidden_states[-1], batch.acoustic_label_positions - 1
                )
            acoustic = self.flow(
                model.acoustic_decoder,
                condition,
                model.acoustic_target_latent(batch.acoustic_labels),
                batch.acoustic_target_mask,
                model.runtime.flow_matching,
            )
            total = total + acoustic.loss.mean()
            result["flow_matching"] = acoustic
            result["loss"] = total
        return result
