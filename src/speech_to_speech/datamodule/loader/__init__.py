"""Loader planning and multi-loader scheduling."""

from typing import TYPE_CHECKING

from .contract import (
    ARFraming,
    LoaderConfig,
    LoaderPlanConfig,
    LoaderStepMode,
    validate_ar_framing,
)

if TYPE_CHECKING:
    from .schedule import (
        LoaderSchedule,
        ScheduledDataLoader,
        SupervisedTokenBatch,
        SupervisedTokenCounter,
        count_supervised_tokens,
        supervised_token_count,
    )

__all__ = [
    "ARFraming",
    "LoaderConfig",
    "LoaderPlanConfig",
    "LoaderSchedule",
    "LoaderStepMode",
    "ScheduledDataLoader",
    "SupervisedTokenBatch",
    "SupervisedTokenCounter",
    "count_supervised_tokens",
    "supervised_token_count",
    "validate_ar_framing",
]


def __getattr__(name: str) -> object:
    if name in {
        "LoaderSchedule",
        "ScheduledDataLoader",
        "SupervisedTokenBatch",
        "SupervisedTokenCounter",
        "count_supervised_tokens",
        "supervised_token_count",
    }:
        from .schedule import (
            LoaderSchedule,
            ScheduledDataLoader,
            SupervisedTokenBatch,
            SupervisedTokenCounter,
            count_supervised_tokens,
            supervised_token_count,
        )

        return {
            "LoaderSchedule": LoaderSchedule,
            "ScheduledDataLoader": ScheduledDataLoader,
            "SupervisedTokenBatch": SupervisedTokenBatch,
            "SupervisedTokenCounter": SupervisedTokenCounter,
            "count_supervised_tokens": count_supervised_tokens,
            "supervised_token_count": supervised_token_count,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
