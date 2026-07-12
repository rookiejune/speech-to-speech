from __future__ import annotations

from ..types import Task
from .base import TaskBase, TaskFactory


@TaskFactory.register(Task.AUDIO_AR)
class AudioAR(TaskBase):
    template = "Continue the {language} speech."


@TaskFactory.register(Task.ASR)
class ASR(TaskBase):
    template = "Transcribe the {language} speech: {source}"


@TaskFactory.register(Task.S2ST)
class S2ST(TaskBase):
    template = "Translate the following speech into {language} speech: {source}"


@TaskFactory.register(Task.S2TT)
class S2TT(TaskBase):
    template = "Translate the following speech into {language} text: {source}"


@TaskFactory.register(Task.TEXT_AR)
class TextAR(TaskBase):
    template = "Continue the following text."


@TaskFactory.register(Task.T2ST)
class T2ST(TaskBase):
    template = "Translate the following text into {language} speech: {source}"


@TaskFactory.register(Task.T2TT)
class T2TT(TaskBase):
    template = "Translate the following text into {language}: {source}"


@TaskFactory.register(Task.TTS)
class TTS(TaskBase):
    template = "Synthesize speech from the following text: {source}"
