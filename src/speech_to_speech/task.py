from __future__ import annotations

from anydataset.types import Modality

from ._compat import StrEnum, auto


class Task(StrEnum):
    AUDIO_AR = auto()
    ASR = auto()
    MT = auto()
    S2ST = auto()
    S2TT = auto()
    TEXT_AR = auto()
    T2ST = auto()
    T2TT = auto()
    TTS = auto()

    @property
    def source_modality(self) -> Modality | None:
        if self in {Task.AUDIO_AR, Task.TEXT_AR}:
            return None
        if self in {Task.ASR, Task.S2ST, Task.S2TT}:
            return Modality.AUDIO
        return Modality.TEXT

    @property
    def target_modality(self) -> Modality:
        if self in {Task.ASR, Task.MT, Task.S2TT, Task.TEXT_AR, Task.T2TT}:
            return Modality.TEXT
        return Modality.AUDIO

    @property
    def uses_source_role(self) -> bool:
        return self in {Task.MT, Task.S2ST, Task.S2TT, Task.T2ST, Task.T2TT}

    @property
    def template(self) -> str:
        if self is Task.AUDIO_AR:
            return "Continue the {language} speech."
        if self is Task.ASR:
            return "Transcribe the {language} speech: {source}"
        if self is Task.MT:
            return "Translate the following text into {language}: {source}"
        if self is Task.S2ST:
            return "Translate the following speech into {language} speech: {source}"
        if self is Task.S2TT:
            return "Translate the following speech into {language} text: {source}"
        if self is Task.TEXT_AR:
            return "Continue the following text."
        if self is Task.T2ST:
            return "Translate the following text into {language} speech: {source}"
        if self is Task.T2TT:
            return "Translate the following text into {language}: {source}"
        if self is Task.TTS:
            return "Synthesize speech from the following text: {source}"
        raise AssertionError(f"unsupported task: {self}")
