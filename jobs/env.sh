#!/usr/bin/env bash
set -euo pipefail

SPEECH_TO_SPEECH_ROOT="${SPEECH_TO_SPEECH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPOS_ROOT="${REPOS_ROOT:-$(cd "${SPEECH_TO_SPEECH_ROOT}/.." && pwd)}"

source "${REPOS_ROOT}/workspace/jobs/env.sh"

export SPEECH_TO_SPEECH_ROOT
export PYTHONPATH="${SPEECH_TO_SPEECH_ROOT}/src:${REPOS_ROOT}/UniCodec:${REPOS_ROOT}/third_party/LongCat-Audio-Codec:${REPOS_ROOT}/third_party/length-based-batching-adapter/src:${PYTHONPATH:-}"
export SPEECH_TO_SPEECH_PYTHON="${SPEECH_TO_SPEECH_PYTHON:-${WORKSPACE_PYTHON:-python}}"
if [[ -z "${SPEECH_TO_SPEECH_TRAIN_ROOT:-}" ]]; then
    : "${DYNAMIC_HOME:?Set DYNAMIC_HOME or SPEECH_TO_SPEECH_TRAIN_ROOT before sourcing jobs/env.sh}"
    SPEECH_TO_SPEECH_TRAIN_ROOT="${DYNAMIC_HOME}/train/speech-to-speech"
fi
export SPEECH_TO_SPEECH_TRAIN_ROOT
export SPEECH_TO_SPEECH_AUDIO_TOKENIZER="${SPEECH_TO_SPEECH_AUDIO_TOKENIZER:-${STATIC_HOME}/bpe/longcat/vocab_100k_minfreq_0_maxlen_none_codes_8192}"
