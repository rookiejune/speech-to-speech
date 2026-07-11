import torch
from torch import Tensor

from ..datamodule.types import Task
from ..runtime import runtime
from .types import LossItem


def _mask_loss(loss: Tensor, mask_or_weight: Tensor):
    if mask_or_weight.dtype is torch.bool:
        mask_or_weight = mask_or_weight.to(dtype=loss.dtype)
    return (loss * mask_or_weight).mean()


def _mask(labels: Tensor):
    return labels == runtime().pad_token_id
