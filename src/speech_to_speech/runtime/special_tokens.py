from .._compat import StrEnum, auto


class AudioSpecialToken(StrEnum):
    BOA = auto()
    EOA = auto()


class TextSpecialToken(StrEnum):
    # Qwen3
    PAD = "<|endoftext|>"
    BOS = "<|im_start|>"
    EOS = "<|im_end|>"
    USER = "user"
    ASSISTANT = "assistant"
    SEP = "\n"
    BOT = "<think>"
    EOT = "</think>"
