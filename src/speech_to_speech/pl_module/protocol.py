from typing import Optional, Protocol

from peft import LoraConfig

from ..generation.protocol import AcousticFeatureGenerator
from ..loss.protocol import FlowObjectiveModel, RVQObjectiveModel
from ..model._contract import ModelCheckpointContract


class FlowCompositionModel(
    FlowObjectiveModel,
    AcousticFeatureGenerator,
    Protocol,
):
    @property
    def checkpoint_contract(self) -> ModelCheckpointContract: ...

    @property
    def lora_config(self) -> Optional[LoraConfig]: ...


class RVQCompositionModel(
    RVQObjectiveModel,
    AcousticFeatureGenerator,
    Protocol,
):
    @property
    def checkpoint_contract(self) -> ModelCheckpointContract: ...

    @property
    def lora_config(self) -> Optional[LoraConfig]: ...
