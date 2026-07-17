from typing import Protocol

from ..generation.protocol import AcousticFeatureGenerator
from ..loss.protocol import FlowObjectiveModel, RVQObjectiveModel


class FlowCompositionModel(FlowObjectiveModel, AcousticFeatureGenerator, Protocol):
    pass


class RVQCompositionModel(RVQObjectiveModel, AcousticFeatureGenerator, Protocol):
    pass
