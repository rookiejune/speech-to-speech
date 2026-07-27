from anytrain.loss import MaskedCodebookCrossEntropyLoss


# Kept under the S2S name for the joint objective API. The implementation is
# task-agnostic and belongs to anytrain.
CausalAcousticLoss = MaskedCodebookCrossEntropyLoss

__all__ = ["CausalAcousticLoss"]
