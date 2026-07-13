from .._compat import StrEnum


class Qwen3SpecialToken(StrEnum):
    PAD = "<|endoftext|>"
    BOS = "<|im_start|>"
    EOS = "<|im_end|>"
