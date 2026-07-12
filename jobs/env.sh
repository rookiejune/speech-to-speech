#!/usr/bin/env bash
set -euo pipefail

SPEECH_TO_SPEECH_ROOT="${SPEECH_TO_SPEECH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPOS_ROOT="${REPOS_ROOT:-$(cd "${SPEECH_TO_SPEECH_ROOT}/.." && pwd)}"

source "${REPOS_ROOT}/workspace/jobs/env.sh"

export SPEECH_TO_SPEECH_ROOT
export PYTHONPATH="${SPEECH_TO_SPEECH_ROOT}/src:${PYTHONPATH:-}"
export SPEECH_TO_SPEECH_PYTHON="${SPEECH_TO_SPEECH_PYTHON:-${WORKSPACE_PYTHON:-python}}"
export SPEECH_TO_SPEECH_TRAIN_ROOT="${SPEECH_TO_SPEECH_TRAIN_ROOT:-${DYNAMIC_HOME}/train/speech-to-speech}"
export SPEECH_TO_SPEECH_AUDIO_TOKENIZER="${SPEECH_TO_SPEECH_AUDIO_TOKENIZER:-${STATIC_HOME}/bpe/longcat/vocab_100k_minfreq_0_maxlen_none_codes_8192}"
