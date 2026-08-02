from __future__ import annotations

from typing import TYPE_CHECKING

from anydataset.types import AudioView

from ...runtime import AudioSequenceLayout
from ...runtime.audio_tokenizer import BiCodecAudioTokenizer

if TYPE_CHECKING:
    from ..protocol import DataRuntime


def needs_reference_audio_context(runtime: DataRuntime) -> bool:
    return (
        runtime.audio_sequence_layout is AudioSequenceLayout.SEMANTIC
        and runtime.audio_view is AudioView.BICODEC
        and isinstance(runtime.audio_tokenizer, BiCodecAudioTokenizer)
    )
