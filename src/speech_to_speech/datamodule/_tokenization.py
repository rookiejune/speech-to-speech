import torch
from torch import Tensor

from ..runtime.types import TextTokenizer


def token_ids(text: str, tokenizer: TextTokenizer) -> Tensor:
    values = torch.as_tensor(
        tokenizer.encode(text, add_special_tokens=False),
        dtype=torch.long,
    )
    if values.dim() != 1:
        raise ValueError("text tokenizer must return a 1D token sequence.")
    return values
