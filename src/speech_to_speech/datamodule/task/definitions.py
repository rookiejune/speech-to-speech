from __future__ import annotations

from ..types import Task
from .abc import TaskBase, TaskFactory


@TaskFactory.register(Task.AUDIO_AR)
class AudioAR(TaskBase):
    source = None
    target = "audio"
    template = "Continue the {language} speech."
    paired = False


@TaskFactory.register(Task.ASR)
class ASR(TaskBase):
    source = "audio"
    target = "text"
    template = "Transcribe the {language} speech: {source}"
    paired = False


@TaskFactory.register(Task.S2ST)
class S2ST(TaskBase):
    source = "audio"
    target = "audio"
    template = "Translate the following speech into {language} speech: {source}"
    paired = True


@TaskFactory.register(Task.S2TT)
class S2TT(TaskBase):
    source = "audio"
    target = "text"
    template = "Translate the following speech into {language} text: {source}"
    paired = True


@TaskFactory.register(Task.TEXT_AR)
class TextAR(TaskBase):
    source = None
    target = "text"
    template = "Continue the following text."
    paired = False


@TaskFactory.register(Task.T2ST)
class T2ST(TaskBase):
    source = "text"
    target = "audio"
    template = "Translate the following text into {language} speech: {source}"
    paired = True


@TaskFactory.register(Task.TTS)
class TTS(TaskBase):
    source = "text"
    target = "audio"
    template = "Synthesize speech from the following text: {source}"
    paired = False
