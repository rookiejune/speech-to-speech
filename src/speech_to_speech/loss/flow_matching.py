from semantic_acoustic_codec.loss.flow import (
    FeatureDecoder,
    FlowLoss,
    FlowRuntime,
    TrainingSample,
)


# The S2S objective keeps this name to describe the acoustic branch. The
# masked flow objective itself is shared with semantic-acoustic-codec.
AcousticFlowLoss = FlowLoss

__all__ = [
    "AcousticFlowLoss",
    "FeatureDecoder",
    "FlowRuntime",
    "TrainingSample",
]
