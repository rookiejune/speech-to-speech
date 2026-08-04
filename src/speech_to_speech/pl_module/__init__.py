from typing import TYPE_CHECKING

from .mimo import MimoModel, MimoModule, MimoPretrainingModule, MimoStepOutput

if TYPE_CHECKING:
    from .module import Config, SpeechToSpeechModule

__all__ = [
    "Config",
    "MimoModel",
    "MimoModule",
    "MimoPretrainingModule",
    "MimoStepOutput",
    "SpeechToSpeechModule",
]


def __getattr__(name: str) -> object:
    if name in {"Config", "SpeechToSpeechModule"}:
        from .module import Config, SpeechToSpeechModule

        return {"Config": Config, "SpeechToSpeechModule": SpeechToSpeechModule}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
