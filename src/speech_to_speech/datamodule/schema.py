from __future__ import annotations

from anydataset import AudioReq, AudioView, Modality, Role, Schema

LONGCAT_AUDIO = AudioReq(views=frozenset({AudioView.LONGCAT}))

SOURCE_AUTOREGRESSION: Schema = {
    (Role.SOURCE, Modality.AUDIO): LONGCAT_AUDIO,
}

TARGET_AUTOREGRESSION: Schema = {
    (Role.TARGET, Modality.AUDIO): LONGCAT_AUDIO,
}

TRANSLATION: Schema = {
    (Role.SOURCE, Modality.AUDIO): LONGCAT_AUDIO,
    (Role.TARGET, Modality.AUDIO): LONGCAT_AUDIO,
}
