from typing import Protocol

from ..generation.protocol import AcousticFeatureGenerator
from ..loss.protocol import FlowObjectiveModel, RVQObjectiveModel
from ..model.lora import LoraModel


class FlowCompositionModel(
    FlowObjectiveModel,
    AcousticFeatureGenerator,
    LoraModel,
    Protocol,
):
    pass


class RVQCompositionModel(
    RVQObjectiveModel,
    AcousticFeatureGenerator,
    LoraModel,
    Protocol,
):
    pass
