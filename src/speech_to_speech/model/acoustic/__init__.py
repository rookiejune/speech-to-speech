from ._config import (
    AcousticConfig,
    AcousticNoneConfig,
    AcousticType,
    DecoderConfig,
    FlowConfig,
    FlowRepaConfig,
    RepaConfig,
    RVQConfig,
)

__all__ = [
    "AcousticFlow",
    "AcousticConfig",
    "AcousticNoneConfig",
    "AcousticType",
    "DecoderConfig",
    "FlowConfig",
    "FlowModel",
    "FlowRepaConfig",
    "HiddenConditionAdapter",
    "RepaConfig",
    "RVQConfig",
    "RVQModel",
]


def __getattr__(name: str) -> object:
    if name == "HiddenConditionAdapter":
        from .condition import HiddenConditionAdapter

        return HiddenConditionAdapter
    if name in {"AcousticFlow", "FlowModel"}:
        from .flow import AcousticFlow, FlowModel

        return {"AcousticFlow": AcousticFlow, "FlowModel": FlowModel}[name]
    if name == "RVQModel":
        from .rvq import RVQModel

        return RVQModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
