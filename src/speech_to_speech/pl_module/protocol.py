from typing import Optional, Protocol

from peft import LoraConfig

from ..generation.protocol import AcousticFeatureGenerator
from ..loss.protocol import FlowObjectiveModel, RVQObjectiveModel


class FlowCompositionModel(
    FlowObjectiveModel,
    AcousticFeatureGenerator,
    Protocol,
):
    @property
    def lora_config(self) -> Optional[LoraConfig]: ...


class RVQCompositionModel(
    RVQObjectiveModel,
    AcousticFeatureGenerator,
    Protocol,
):
    @property
    def lora_config(self) -> Optional[LoraConfig]: ...
