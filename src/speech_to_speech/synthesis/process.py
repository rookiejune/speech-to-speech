"""Stable factory entry point for the subprocess synthesis producer."""

from __future__ import annotations

from speech_to_speech.datamodule.streaming import SynthesisRequest

from .subprocess import SubprocessController, subprocess_controller


def controller(request: SynthesisRequest) -> SubprocessController:
    """Create the configured rank-zero subprocess producer controller."""

    return subprocess_controller(request)


__all__ = ["controller"]
